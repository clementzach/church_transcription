import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import queue
import string
import threading
import time
import httpx
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from dotenv import load_dotenv
import websockets
from google.cloud import texttospeech as _texttospeech
import google.auth.api_key as _google_api_key
from filter import filter_text as _filter_text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(funcName)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

app = FastAPI()

GLADIA_API_KEY = os.getenv("GLADIA_API_KEY")

# ── Google Cloud TTS client (singleton, shared across all sessions) ─────────
# Explicit ApiKeyCredentials bypasses google.auth.default() credential
# discovery, which otherwise hangs ~30s on non-GCE machines trying to reach
# the GCE metadata server.
_tts_client = _texttospeech.TextToSpeechClient(
    credentials=_google_api_key.Credentials(os.getenv("GOOGLE_TTS_API_KEY"))
)

ALL_LANGS          = ['en', 'es', 'ht', 'pt', 'zh', 'fr', 'no']
TRANSLATION_LANGS  = ['es', 'ht', 'pt', 'zh', 'fr', 'no']

# Local Haitian Creole Whisper model (optional)
LOCAL_HAITIAN_PATH = os.getenv('LOCAL_HAITIAN_PATH')
_local_whisper_model = None
if LOCAL_HAITIAN_PATH:
    try:
        from faster_whisper import WhisperModel
        _local_whisper_model = WhisperModel(
            LOCAL_HAITIAN_PATH, device='cuda', compute_type='float16'
        )
        log.info("Loaded local HT Whisper model from %s", LOCAL_HAITIAN_PATH)
    except Exception as _e:
        log.error("Failed to load local Whisper model from %s: %s", LOCAL_HAITIAN_PATH, _e)

if LOCAL_HAITIAN_PATH:
    from google import genai as _google_genai
    google_llm_client = _google_genai.Client(api_key=os.getenv("GOOGLE_LLM_API_KEY"))

# ── Local HT audio chunking / translation constants ───────────────────────
_WHISPER_SR = 16000
_MAX_CHUNK_SAMPLES = 10 * _WHISPER_SR   # flush audio to Whisper every 10 s max
_MAX_TRANSLATION_WORDS = 40             # also flush text buffer at this word count

_TRANSLATION_LANG_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'pt': 'Portuguese',
    'zh': 'Mandarin Chinese',
    'fr': 'French',
    'no': 'Norwegian',
}

# Gladia live-session config — kept server-side so the browser never needs to
# know about Gladia's API surface.
GLADIA_CONFIG_BASE = {
    'encoding': 'wav/pcm',
    'sample_rate': 16000,
    'bit_depth': 16,
    'channels': 1,
    'realtime_processing': {
        'translation': True,
        'translation_config': {
            'model': 'enhanced',
            'lipsync': False,
            'match_original_utterances': False,
            'context_adaptation': True,
            'context': (
                'This is a Latter-day Saint (LDS/Mormon) church service. '
                'Speakers may reference LDS-specific scripture, theology, and '
                'organizational terms. Preserve proper nouns exactly as spoken.'
            ),
        },
    },
}

# ── OpenAI TTS config (gpt-4o-mini-tts) ───────────────────────────────────
OPENAI_TTS_VOICES = {
    'es': 'nova',
    'pt': 'shimmer',
    'ht': 'alloy',
    'zh': 'nova',
    'fr': 'echo',
    'no': 'alloy',
}
OPENAI_TTS_INSTRUCTIONS = {
    'es': 'Speak naturally and clearly in Spanish.',
    'pt': 'Fale de forma natural e clara em português.',
    'ht': 'Ou pale kreyòl tankou yon natif natal',
    'zh': '请用标准普通话自然流利地朗读，像母语人士一样说话，发音清晰，语调自然。',
    'fr': 'Parlez de manière naturelle et claire en français.',
    'no': 'Snakk naturlig og tydelig på norsk.',
}

# ── Google Gemini 2.5 Flash TTS config ────────────────────────────────────
GOOGLE_TTS_VOICES = {
    'en': 'Puck',
    'es': 'Charon',
    'pt': 'Aoede',
    'ht': 'Kore',
    'zh': 'Fenrir',
    'fr': 'Puck',
    'no': 'Charon',
}
GOOGLE_TTS_LOCALES = {
    'en': 'en-US',
    'es': 'es-US',
    'pt': 'pt-BR',
    'ht': 'fr-HT',
    'zh': 'cmn-CN',
    'fr': 'fr-FR',
    'no': 'nb-NO',
}
# After this many seconds of no new text, close the gRPC stream and reopen on demand.
# Keeps the streaming_synthesize call count well within Google's ~200/day rate limit.
TTS_INACTIVITY_TIMEOUT = 30.0

# ── Session state ─────────────────────────────────────────────────────────────
# sessions[session_id] = {
#   'start_time': float,
#   'listener_registry': {lang: [asyncio.Queue, ...]},
#   'tts_queues':        {lang: queue.Queue(maxsize=1)},   # stdlib Queue; used by TTS threads
#   'timer': threading.Timer,
# }
SESSION_TIMEOUT_SECS = 2 * 3600  # 2 hours
_lock = threading.Lock()
sessions = {}

# Captured at startup; used by TTS worker threads to schedule puts on the event loop.
_loop: asyncio.AbstractEventLoop | None = None


@app.on_event("startup")
async def _on_startup():
    global _loop
    _loop = asyncio.get_running_loop()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _force_put(q: queue.Queue, item):
    """Put item into a Queue(maxsize=1), replacing any stale pending item."""
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put_nowait(item)


def _broadcast(session_id, lang, message):
    """Send a JSON str or raw PCM bytes to every listener for (session, lang).

    Called from TTS worker threads. Each listener owns an asyncio.Queue;
    call_soon_threadsafe schedules the put on the main event loop.
    """
    with _lock:
        session = sessions.get(session_id)
        if not session:
            return
        listeners = list(session['listener_registry'].get(lang, []))
    for q in listeners:
        try:
            _loop.call_soon_threadsafe(q.put_nowait, message)
        except Exception:
            pass


def _teardown_session(session, reason='Session ended'):
    """Cancel the session timer, stop TTS workers, and notify all listeners."""
    session['timer'].cancel()
    error_msg = json.dumps({'type': 'error', 'message': reason})
    for lang in ALL_LANGS:
        _force_put(session['tts_queues'][lang], None)
        for q in list(session['listener_registry'].get(lang, [])):
            try:
                _loop.call_soon_threadsafe(q.put_nowait, error_msg)
                _loop.call_soon_threadsafe(q.put_nowait, None)
            except Exception:
                pass


def _translate_ht_to(ht_text, target_lang, prev_sentences):
    """Translate Haitian Creole text to target_lang using Gemini text model."""
    lang_name = _TRANSLATION_LANG_NAMES[target_lang]
    context_lines = list(prev_sentences)[-2:]

    prompt = (
        "You are translating a live Latter-day Saint (LDS/Mormon) church talk from Haitian Creole "
        f"to {lang_name}. Speakers may reference LDS-specific scripture, theology, and "
        "organizational terms — preserve proper nouns exactly. The transcription may have been "
        "garbled by speech recognition."
    )
    if context_lines:
        prompt += "\n\nPrevious sentences (for context):\n" + "\n".join(context_lines)
    prompt += (
        f"\n\nTranslate the following Haitian Creole text to {lang_name}. "
        "Output only the translation with no explanation:\n" + ht_text
    )

    resp = google_llm_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    return resp.text.strip()


def _transcribe_and_broadcast_ht(session_id, audio_np, text_buffer, prev_sentences, send_callback):
    """Transcribe audio with faster-whisper VAD, buffer text until a sentence boundary
    or max word count, then translate active languages in parallel.

    Returns (ht_text, updated_text_buffer). ht_text is empty string if Whisper
    produced nothing; text_buffer carries over across calls until flushed.
    """
    chunk_secs = len(audio_np) / _WHISPER_SR
    t0 = time.time()
    try:
        segments, _ = _local_whisper_model.transcribe(
            audio_np, language='ht', vad_filter=True,
            vad_parameters={'min_silence_duration_ms': 500},
        )
        ht_text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        log.error("[local_ht:%s] Whisper error: %s", session_id, e)
        return '', text_buffer
    elapsed = time.time() - t0
    log.info("[local_ht:%s] chunk=%.1fs whisper=%.2fs text=%.60s",
             session_id, chunk_secs, elapsed, ht_text or '(empty)')

    if not ht_text:
        return '', text_buffer

    ht_text = _filter_text(ht_text)

    # Send raw HT transcript to broadcaster display immediately.
    try:
        send_callback(json.dumps({
            'type': 'transcript',
            'data': {'utterance': {'text': ht_text, 'is_final': True, 'language': 'ht'}},
        }))
    except Exception:
        pass

    # HT listeners get audio on every chunk — no need to wait for a full sentence.
    _enqueue_tts(session_id, 'ht', ht_text)

    # Accumulate into the translation buffer and check for a flush trigger.
    text_buffer = (text_buffer + ' ' + ht_text).strip()
    ends_sentence = text_buffer[-1] in '.?!…' if text_buffer else False
    word_count = len(text_buffer.split())

    if not ends_sentence and word_count < _MAX_TRANSLATION_WORDS:
        return ht_text, text_buffer

    # Flush: translate the buffered text into every language that has a listener.
    text_to_translate = text_buffer
    text_buffer = ''

    with _lock:
        sess = sessions.get(session_id)
        display_langs = set(sess.get('display_langs', ALL_LANGS)) if sess else set()
        listener_langs = {lang for lang in ALL_LANGS if sess and sess['listener_registry'].get(lang)}
        active_langs = [
            lang for lang in ALL_LANGS
            if lang != 'ht' and lang in (display_langs | listener_langs)
        ]

    if not active_langs:
        return ht_text, text_buffer

    def translate_one(lang):
        try:
            translated = _translate_ht_to(text_to_translate, lang, prev_sentences)
            return lang, _filter_text(translated) if translated else None
        except Exception as e:
            log.error("[local_ht:%s] Translation to %s failed: %s", session_id, lang, e)
            return lang, None

    with ThreadPoolExecutor(max_workers=len(active_langs)) as executor:
        results = list(executor.map(translate_one, active_langs))

    for lang, translated in results:
        if not translated:
            continue
        try:
            send_callback(json.dumps({
                'type': 'translation',
                'data': {
                    'target_language': lang,
                    'translated_utterance': {'text': translated},
                },
            }))
        except Exception:
            pass
        _enqueue_tts(session_id, lang, translated)

    return ht_text, text_buffer


def _expire_session(session_id):
    """Called by the session timer after SESSION_TIMEOUT_SECS."""
    log.info("Session %s timed out after %ds", session_id, SESSION_TIMEOUT_SECS)
    with _lock:
        session = sessions.pop(session_id, None)
    if session is None:
        return
    _teardown_session(session, 'Session expired (2-hour limit reached)')


def _enqueue_tts(session_id, lang, text):
    """Enqueue translated text for TTS synthesis, dropping any stale pending item."""
    with _lock:
        session = sessions.get(session_id)
        if not session or not session['listener_registry'].get(lang):
            return
    _force_put(session['tts_queues'][lang], text)


def _tts_worker(session_id, lang):
    """Per-(session, lang) TTS worker.

    Maintains one long-lived streaming_synthesize call that accepts multiple
    utterances, keeping the call count well within API rate limits.  The stream
    is closed after TTS_INACTIVITY_TIMEOUT seconds of silence and reopened on
    the next utterance.

    request_gen (runs in gRPC's internal background thread)
      • Reads text from tts_queues[lang], broadcasts the caption, then yields
        the synthesis request.
      • Returns after TTS_INACTIVITY_TIMEOUT of silence or on shutdown sentinel.

    The main worker loop iterates the streaming responses directly, broadcasting
    each raw PCM chunk to listeners as it arrives.
    """
    log.info("TTS worker started: session=%s lang=%s", session_id, lang)

    with _lock:
        session = sessions.get(session_id)
        if not session:
            return
        tts_q = session['tts_queues'][lang]

    shutdown = threading.Event()

    locale = GOOGLE_TTS_LOCALES[lang]
    voice_name = f"{locale}-Standard-A"
    config_req = _texttospeech.StreamingSynthesizeRequest(
        streaming_config=_texttospeech.StreamingSynthesizeConfig(
            voice=_texttospeech.VoiceSelectionParams(
                language_code=locale,
                name=voice_name,
            ),
            streaming_audio_config=_texttospeech.StreamingAudioConfig(
                audio_encoding=_texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
            ),
        )
    )

    while not shutdown.is_set():
        # Block until there is actual text to synthesize before opening a gRPC
        # stream — avoids idle stream-open API calls when no listeners are present.
        first_text = None
        while not shutdown.is_set():
            try:
                first_text = tts_q.get(timeout=1.0)
                break
            except queue.Empty:
                continue

        if shutdown.is_set():
            break
        if first_text is None:  # shutdown sentinel
            break

        call_alive = threading.Event()
        call_alive.set()

        def request_gen(initial=first_text):
            """Yields synthesis requests; consumed by gRPC in its background thread."""
            yield config_req
            _broadcast(session_id, lang, json.dumps({'type': 'text', 'text': initial}))
            log.info("[tts:%s:%s] → TTS: %.60s", session_id, lang, initial)
            yield _texttospeech.StreamingSynthesizeRequest(
                input=_texttospeech.StreamingSynthesisInput(text=initial)
            )
            inactive_since = None
            while call_alive.is_set():
                try:
                    text = tts_q.get(timeout=0.2)
                    inactive_since = None
                except queue.Empty:
                    if inactive_since is None:
                        inactive_since = time.time()
                    elif time.time() - inactive_since >= TTS_INACTIVITY_TIMEOUT:
                        log.info("[tts:%s:%s] Inactivity timeout — closing stream", session_id, lang)
                        call_alive.clear()
                        return
                    continue
                if text is None:
                    shutdown.set()
                    call_alive.clear()
                    return
                _broadcast(session_id, lang, json.dumps({'type': 'text', 'text': text}))
                log.info("[tts:%s:%s] → TTS: %.60s", session_id, lang, text)
                yield _texttospeech.StreamingSynthesizeRequest(
                    input=_texttospeech.StreamingSynthesisInput(text=text)
                )

        try:
            for response in _tts_client.streaming_synthesize(request_gen()):
                if response.audio_content:
                    _broadcast(session_id, lang, response.audio_content)
        except Exception as e:
            if not shutdown.is_set():
                log.error("[tts:%s:%s] Stream error: %s", session_id, lang, e)
        finally:
            call_alive.clear()

    log.info("TTS worker stopped: session=%s lang=%s", session_id, lang)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/listen")
async def listen():
    return FileResponse("static/listen.html")


@app.get("/{code}")
async def index_with_code(code: str):
    from fastapi import HTTPException
    valid_chars = string.ascii_uppercase + string.digits
    if len(code) == 6 and all(c in valid_chars for c in code.upper()):
        return FileResponse("static/index.html")
    raise HTTPException(status_code=404)


@app.post("/init-session")
async def init_session(request: Request):
    body = await request.json()
    src_lang = body.get('language', 'auto')
    session_id = (body.get('session_id') or '').upper().strip()
    log.info("init-session called, session_id=%s language=%s", session_id, src_lang)

    valid_chars = string.ascii_uppercase + string.digits
    if not session_id or len(session_id) != 6 or not all(c in valid_chars for c in session_id):
        return Response(
            content=json.dumps({'error': 'Session code must be exactly 6 alphanumeric characters'}),
            status_code=400,
            media_type='application/json',
        )

    old_session = None
    with _lock:
        existing = sessions.get(session_id)
        if existing is not None:
            if existing.get('broadcaster_connected', False):
                return Response(
                    content=json.dumps({'error': 'This session code is already in use. Choose a different code.'}),
                    status_code=409,
                    media_type='application/json',
                )
            old_session = sessions.pop(session_id)

    if old_session is not None:
        log.info("init-session: reclaiming disconnected session %s", session_id)
        _teardown_session(old_session, 'Session replaced by broadcaster')

    # Local HT mode: skip Gladia when local Whisper is loaded and HT is selected
    if src_lang == 'ht' and _local_whisper_model is not None:
        timer = threading.Timer(SESSION_TIMEOUT_SECS, _expire_session, args=(session_id,))
        timer.daemon = True
        session = {
            'start_time': time.time(),
            'src_lang': 'ht',
            'gladia_url': '',
            'broadcaster_connected': False,
            'listener_registry': {lang: [] for lang in ALL_LANGS},
            'tts_queues': {lang: queue.Queue(maxsize=1) for lang in ALL_LANGS},
            'timer': timer,
            'local_ht': True,
        }
        with _lock:
            sessions[session_id] = session
        for lang in ALL_LANGS:
            threading.Thread(target=_tts_worker, args=(session_id, lang), daemon=True).start()
        timer.start()
        log.info("Session created (local HT): id=%s timeout=%ds", session_id, SESSION_TIMEOUT_SECS)
        return {'session_id': session_id, 'local_ht': True}

    # Build the Gladia config server-side.
    config = copy.deepcopy(GLADIA_CONFIG_BASE)
    if src_lang == 'auto':
        config['language_config'] = {'languages': ALL_LANGS[:], 'code_switching': True}
        config['realtime_processing']['translation_config']['target_languages'] = ALL_LANGS[:]
    else:
        config['language_config'] = {'languages': [src_lang], 'code_switching': False}
        config['realtime_processing']['translation_config']['target_languages'] = [
            l for l in ALL_LANGS if l != src_lang
        ]

    log.info("Gladia config: %s", json.dumps(config))
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.gladia.io/v2/live",
            headers={
                "X-Gladia-Key": GLADIA_API_KEY,
                "Content-Type": "application/json",
            },
            json=config,
            timeout=10,
        )
    log.info("Gladia /v2/live response: status=%d", resp.status_code)
    if resp.is_success:
        data = resp.json()
        gladia_url = data.get('url') or data.get('websocket_url') or data.get('ws_url', '')

        timer = threading.Timer(SESSION_TIMEOUT_SECS, _expire_session, args=(session_id,))
        timer.daemon = True
        session = {
            'start_time': time.time(),
            'src_lang': src_lang,
            'gladia_url': gladia_url,
            'broadcaster_connected': False,
            'listener_registry': {lang: [] for lang in ALL_LANGS},
            'tts_queues': {lang: queue.Queue(maxsize=1) for lang in ALL_LANGS},
            'timer': timer,
        }
        concurrent_old = None
        with _lock:
            existing = sessions.get(session_id)
            if existing is not None:
                if existing.get('broadcaster_connected', False):
                    return Response(
                        content=json.dumps({'error': 'This session code is already in use. Choose a different code.'}),
                        status_code=409,
                        media_type='application/json',
                    )
                concurrent_old = sessions.pop(session_id)
            sessions[session_id] = session

        if concurrent_old is not None:
            _teardown_session(concurrent_old, 'Session replaced by broadcaster')

        for lang in ALL_LANGS:
            threading.Thread(
                target=_tts_worker, args=(session_id, lang), daemon=True
            ).start()
        timer.start()

        log.info("Session created: id=%s timeout=%ds gladia_url=%s",
                 session_id, SESSION_TIMEOUT_SECS, gladia_url)
        return {'session_id': session_id}
    log.error("Gladia init failed: %d %s", resp.status_code, resp.text)
    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


@app.post("/end-session")
async def end_session(request: Request):
    body = await request.json()
    session_id = (body.get('session_id') or '').upper().strip()

    with _lock:
        session = sessions.pop(session_id, None)

    if not session:
        return {'ok': True}

    _teardown_session(session, 'Session ended by broadcaster')
    log.info("Session ended by broadcaster: %s", session_id)
    return {'ok': True}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()

    # First message from browser: JSON with session_id only.
    try:
        raw_first = await ws.receive_text()
    except WebSocketDisconnect:
        return

    try:
        first_msg = json.loads(raw_first)
        session_id = first_msg.get('session_id', '')
    except Exception as e:
        log.error("stream: failed to parse first message: %s", e)
        await ws.close()
        return

    if not session_id:
        log.error("stream: missing session_id in first message")
        await ws.close()
        return

    with _lock:
        session = sessions.get(session_id)
    if not session:
        log.error("stream: unknown session_id=%s", session_id)
        await ws.close()
        return

    gladia_url = session.get('gladia_url', '')
    src_lang = session.get('src_lang', 'auto')
    if not gladia_url or not gladia_url.startswith("wss://"):
        log.error("stream: invalid or missing Gladia URL in session: %r", gladia_url)
        await ws.close()
        return

    log.info("stream: session=%s src_lang=%s connecting to Gladia at %s", session_id, src_lang, gladia_url)
    with _lock:
        if sessions.get(session_id) is session:
            session['broadcaster_connected'] = True

    audio_chunks = 0
    gladia_msgs = 0
    try:
        async with websockets.connect(gladia_url) as gladia_ws:
            log.info("stream: Gladia connection established")

            async def browser_to_gladia():
                nonlocal audio_chunks
                try:
                    while True:
                        data = await ws.receive()
                        if data['type'] == 'websocket.disconnect':
                            log.info("stream/browser_to_gladia: browser closed connection")
                            return
                        msg_bytes = data.get('bytes')
                        if msg_bytes:
                            audio_chunks += 1
                            if audio_chunks % 50 == 1:
                                log.info("stream/forward_audio: relayed %d audio chunks", audio_chunks)
                            await gladia_ws.send(msg_bytes)
                except WebSocketDisconnect:
                    log.info("stream/browser_to_gladia: browser closed connection")
                except Exception as e:
                    log.warning("stream/browser_to_gladia: exception: %s", e)

            async def gladia_to_browser():
                nonlocal gladia_msgs
                try:
                    async for raw in gladia_ws:
                        gladia_msgs += 1
                        msg_type = None

                        # Parse, filter text fields, then forward to browser.
                        # Filtering here means both the recorder display and
                        # listeners see clean text.
                        to_send = raw
                        try:
                            msg = json.loads(raw)
                            msg_type = msg.get('type')

                            if msg_type == 'transcript':
                                data_obj = msg.get('data') or {}
                                utterance = data_obj.get('utterance') or {}
                                if utterance.get('text'):
                                    utterance['text'] = _filter_text(utterance['text'])
                                    to_send = json.dumps(msg)
                                # English TTS comes from transcripts when source is English.
                                # In auto/non-English-source mode it comes from translation msgs.
                                if src_lang == 'en':
                                    is_final = (utterance.get('is_final') is True) or (data_obj.get('is_final') is True)
                                    if is_final and utterance.get('text'):
                                        _enqueue_tts(session_id, 'en', utterance['text'])

                            elif msg_type == 'translation':
                                data_field = msg.get('data', {})
                                translated = data_field.get('translated_utterance') or {}
                                if translated.get('text'):
                                    translated['text'] = _filter_text(translated['text'])
                                    to_send = json.dumps(msg)
                                lang = (data_field.get('target_language') or '').lower()
                                text = translated.get('text', '')
                                if lang in ALL_LANGS and text:
                                    _enqueue_tts(session_id, lang, text)
                        except Exception:
                            pass  # on any parse error, forward the original raw message

                        # Log data messages throttled; always log everything else.
                        if msg_type in ('transcript', 'translation'):
                            if gladia_msgs <= 5 or gladia_msgs % 20 == 0:
                                log.info("stream: gladia msg #%d type=%s", gladia_msgs, msg_type)
                        else:
                            log.info("stream: gladia msg #%d type=%s", gladia_msgs, msg_type)

                        try:
                            await ws.send_text(to_send)
                        except Exception as e:
                            log.warning("stream: browser send error: %s", e)
                            return
                except Exception as e:
                    log.warning("stream/gladia_to_browser: %s", e)

            task1 = asyncio.create_task(browser_to_gladia())
            task2 = asyncio.create_task(gladia_to_browser())
            try:
                await asyncio.wait({task1, task2}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                task1.cancel()
                task2.cancel()
                await asyncio.gather(task1, task2, return_exceptions=True)

    except Exception as e:
        log.error("stream: failed to connect to Gladia: %s", e)
    finally:
        with _lock:
            if sessions.get(session_id) is session:
                session['broadcaster_connected'] = False

    log.info("stream: closing (audio_chunks=%d, gladia_msgs=%d)", audio_chunks, gladia_msgs)


@app.websocket("/listen-stream")
async def listen_stream(ws: WebSocket):
    await ws.accept()

    # First message: JSON with session_id and language.
    try:
        first = await ws.receive_text()
        msg = json.loads(first)
        session_id = msg.get('session_id', '')
        lang = msg.get('language', '').lower()
    except Exception as e:
        log.error("listen-stream: failed to parse first message: %s", e)
        await ws.close()
        return

    log.info("listen-stream: session_id=%s lang=%s", session_id, lang)

    q = asyncio.Queue()
    with _lock:
        session = sessions.get(session_id)
        if not session or lang not in ALL_LANGS:
            log.warning("listen-stream: invalid session_id=%s or lang=%s", session_id, lang)
            await ws.send_text(json.dumps({'type': 'error', 'message': 'Invalid session ID or language'}))
            await ws.close()
            return
        session['listener_registry'][lang].append(q)
        log.info("listen-stream: registered listener for session=%s lang=%s (total=%d)",
                 session_id, lang, len(session['listener_registry'][lang]))

    # Tell the client the raw-PCM format so it can decode chunks without WAV headers.
    await ws.send_text(json.dumps({
        'type': 'audio_config',
        'sample_rate': 24000,
        'channels': 1,
        'sample_width': 2,  # Int16
    }))

    async def drain_incoming():
        """Read from WebSocket so we detect browser disconnect promptly."""
        try:
            while True:
                data = await ws.receive()
                if data['type'] == 'websocket.disconnect':
                    return
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    async def forward_outgoing():
        """Drain the listener queue and forward items to the WebSocket."""
        try:
            while True:
                item = await q.get()
                if item is None:
                    await ws.close()
                    return
                if isinstance(item, bytes):
                    await ws.send_bytes(item)
                else:
                    await ws.send_text(item)
        except Exception:
            pass

    task_in = asyncio.create_task(drain_incoming())
    task_out = asyncio.create_task(forward_outgoing())
    try:
        await asyncio.wait({task_in, task_out}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        task_in.cancel()
        task_out.cancel()
        await asyncio.gather(task_in, task_out, return_exceptions=True)
        with _lock:
            session = sessions.get(session_id)
            if session:
                try:
                    session['listener_registry'][lang].remove(q)
                    log.info("listen-stream: removed listener for session=%s lang=%s", session_id, lang)
                except ValueError:
                    pass


@app.websocket("/stream-local")
async def stream_local(ws: WebSocket):
    """WebSocket endpoint for local Haitian Creole Whisper transcription.

    Accumulates PCM audio (int16, 16 kHz, mono), splits on 800 ms silence or
    30 s max, then transcribes with the local Whisper model and translates to
    all other languages via Gemini.
    """
    await ws.accept()

    try:
        raw_first = await ws.receive_text()
        first_msg = json.loads(raw_first)
        session_id = first_msg.get('session_id', '')
        display_langs = [l for l in first_msg.get('display_langs', ALL_LANGS) if l in ALL_LANGS]
    except Exception:
        await ws.close()
        return

    with _lock:
        session = sessions.get(session_id)
        if not session:
            log.error("stream_local: unknown session_id=%s", session_id)
            await ws.close()
            return
        session['broadcaster_connected'] = True
        session['display_langs'] = display_langs

    log.info("stream_local: session=%s", session_id)

    def send_callback(msg):
        asyncio.run_coroutine_threadsafe(ws.send_text(msg), _loop)

    prev_sentences = []
    prev_lock = threading.Lock()
    transcribe_q = queue.Queue()
    state = {'text_buffer': ''}  # mutable so transcription_worker can update it

    def transcription_worker():
        while True:
            audio_np = transcribe_q.get()
            if audio_np is None:
                break
            with prev_lock:
                prev = list(prev_sentences)
            ht_text, state['text_buffer'] = _transcribe_and_broadcast_ht(
                session_id, audio_np, state['text_buffer'], prev, send_callback
            )
            if ht_text:
                with prev_lock:
                    prev_sentences.append(ht_text)
                    if len(prev_sentences) > 2:
                        del prev_sentences[:-2]

    worker = threading.Thread(target=transcription_worker, daemon=True)
    worker.start()

    # Accumulate raw int16 samples; faster-whisper VAD handles silence internally.
    sample_buffer = []

    def flush():
        nonlocal sample_buffer
        if not sample_buffer:
            return
        audio_np = np.concatenate(sample_buffer).astype(np.float32) / 32768.0
        sample_buffer = []
        transcribe_q.put(audio_np)

    try:
        while True:
            try:
                data = await ws.receive()
            except WebSocketDisconnect:
                break
            if data['type'] == 'websocket.disconnect':
                break
            msg_bytes = data.get('bytes')
            if not msg_bytes:
                # May be a text control message (e.g. display_langs update).
                msg_text = data.get('text')
                if msg_text:
                    try:
                        ctrl = json.loads(msg_text)
                        if ctrl.get('type') == 'display_langs':
                            langs = [l for l in ctrl.get('langs', []) if l in ALL_LANGS]
                            with _lock:
                                if sessions.get(session_id) is session:
                                    session['display_langs'] = langs
                    except Exception:
                        pass
                continue

            chunk_int16 = np.frombuffer(msg_bytes, dtype=np.int16)
            if len(chunk_int16) == 0:
                continue
            sample_buffer.append(chunk_int16)

            if sum(len(b) for b in sample_buffer) >= _MAX_CHUNK_SAMPLES:
                flush()

    finally:
        with _lock:
            s = sessions.get(session_id)
            if s is session:
                session['broadcaster_connected'] = False
        flush()
        transcribe_q.put(None)
        worker.join(timeout=30)
        log.info("stream_local: closed session=%s", session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
