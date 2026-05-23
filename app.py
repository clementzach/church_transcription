import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import itertools
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
from google import genai as _google_genai
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

# Google Cloud TTS client (singleton, shared across all sessions).
# Explicit ApiKeyCredentials bypasses google.auth.default() credential
# discovery, which otherwise hangs ~30s on non-GCE machines.
_tts_client = _texttospeech.TextToSpeechClient(
    credentials=_google_api_key.Credentials(os.getenv("GOOGLE_TTS_API_KEY"))
)
# Chirp3 HD voices output MULAW at 24000 Hz natively (confirmed by test).
# LINEAR16 is not supported for streaming_synthesize; MULAW is used instead.
TTS_SAMPLE_RATE = 24000
# After this many seconds of no new text, close the gRPC stream and reopen.
TTS_INACTIVITY_TIMEOUT = 30.0

ALL_LANGS          = ['en', 'es', 'ht', 'pt', 'zh', 'fr']
TRANSLATION_LANGS  = ['es', 'ht', 'pt', 'zh', 'fr']

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

google_llm_client = None
if LOCAL_HAITIAN_PATH:
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

GOOGLE_TTS_LOCALES = {
    'en': 'en-US',
    'es': 'es-US',
    'pt': 'pt-BR',
    'ht': 'fr-FR',   # no fr-HT locale; Chirp3-HD voices handle Creole via fr-FR
    'zh': 'cmn-CN',
    'fr': 'fr-FR',
}
GOOGLE_TTS_VOICES = {
    'en': 'Puck',
    'es': 'Charon',
    'pt': 'Aoede',
    'ht': 'Kore',
    'zh': 'Fenrir',
    'fr': 'Puck',
}

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

# Unique id assigned to each /stream connection. The finally block only clears
# broadcaster_connected if the current conn_id still matches, so a slow-exiting
# previous connection can't clobber the flag set by a fresh reconnect.
_conn_id_counter = itertools.count(1)

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


def _build_gladia_config(src_lang):
    """Return the Gladia /v2/live POST body for the given source language."""
    config = copy.deepcopy(GLADIA_CONFIG_BASE)
    if src_lang == 'auto':
        config['language_config'] = {'languages': ALL_LANGS[:], 'code_switching': True}
        config['realtime_processing']['translation_config']['target_languages'] = ALL_LANGS[:]
    else:
        config['language_config'] = {'languages': [src_lang], 'code_switching': False}
        config['realtime_processing']['translation_config']['target_languages'] = [
            l for l in ALL_LANGS if l != src_lang
        ]
    return config


async def _create_gladia_session(src_lang):
    """POST to Gladia /v2/live and return the WebSocket URL for the new session.

    Raises RuntimeError on failure.
    """
    config = _build_gladia_config(src_lang)
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
    if not resp.is_success:
        raise RuntimeError(f"Gladia /v2/live failed: {resp.status_code} {resp.text}")
    data = resp.json()
    url = data.get('url') or data.get('websocket_url') or data.get('ws_url', '')
    if not url:
        raise RuntimeError("Gladia /v2/live response missing url")
    return url


async def _reconfigure_session_language(session, new_lang):
    """Update an existing session in-place for a new source language.

    For HT with local Whisper loaded, flips to local-HT mode (no Gladia call).
    Otherwise allocates a fresh Gladia /v2/live session URL. Caller is
    responsible for tearing down whatever backend was using the old config.
    """
    if new_lang == 'ht' and _local_whisper_model is not None:
        session['src_lang'] = 'ht'
        session['gladia_url'] = ''
        session['local_ht'] = True
        return
    gladia_url = await _create_gladia_session(new_lang)
    session['src_lang'] = new_lang
    session['gladia_url'] = gladia_url
    session.pop('local_ht', None)


async def _wait_for_broadcaster_disconnect(session_id, timeout=0.5):
    """Poll until the session's broadcaster_connected flag clears, up to timeout.

    Returns True if the slot is free (session absent or disconnected),
    False on timeout. Used by init-session so a broadcaster reconnecting
    immediately after closing their WebSocket isn't rejected with a 409
    just because the server hasn't run its finally block yet. The client
    awaits its own WebSocket close before calling us, so this is a safety
    margin rather than the primary synchronization mechanism.
    """
    deadline = time.monotonic() + timeout
    while True:
        with _lock:
            s = sessions.get(session_id)
            if s is None or not s.get('broadcaster_connected', False):
                return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.05)


def _enqueue_tts(session_id, lang, text):
    """Enqueue translated text for TTS synthesis, dropping any stale pending item."""
    with _lock:
        session = sessions.get(session_id)
        if not session or not session['listener_registry'].get(lang):
            return
    _force_put(session['tts_queues'][lang], text)


def _tts_worker(session_id, lang):
    """Per-(session, lang) TTS worker.

    Maintains one long-lived streaming_synthesize gRPC call across multiple
    utterances for voice continuity.  The stream closes after
    TTS_INACTIVITY_TIMEOUT seconds of silence and reopens on the next utterance.
    sample_rate_hertz is omitted so the API uses each voice's native rate
    (22050 Hz for Standard voices), matching TTS_SAMPLE_RATE.
    """
    log.info("TTS worker started: session=%s lang=%s", session_id, lang)

    with _lock:
        session = sessions.get(session_id)
        if not session:
            return
        tts_q = session['tts_queues'][lang]

    shutdown = threading.Event()

    locale = GOOGLE_TTS_LOCALES[lang]
    voice_name = f"{locale}-Chirp3-HD-{GOOGLE_TTS_VOICES[lang]}"
    config_req = _texttospeech.StreamingSynthesizeRequest(
        streaming_config=_texttospeech.StreamingSynthesizeConfig(
            voice=_texttospeech.VoiceSelectionParams(
                language_code=locale,
                name=voice_name,
            ),
            streaming_audio_config=_texttospeech.StreamingAudioConfig(
                audio_encoding=_texttospeech.AudioEncoding.MULAW,
                # sample_rate_hertz omitted — Chirp3 HD native rate is 24000 Hz
            ),
        )
    )

    while not shutdown.is_set():
        first_text = None
        while not shutdown.is_set():
            try:
                first_text = tts_q.get(timeout=1.0)
                break
            except queue.Empty:
                continue

        if shutdown.is_set():
            break
        if first_text is None:
            break

        call_alive = threading.Event()
        call_alive.set()

        def request_gen(initial=first_text):
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
            log.info("[tts:%s:%s] Opening stream: voice=%s", session_id, lang, voice_name)
            audio_chunks = 0
            for response in _tts_client.streaming_synthesize(request_gen()):
                if response.audio_content:
                    audio_chunks += 1
                    _broadcast(session_id, lang, response.audio_content)
            log.info("[tts:%s:%s] Stream closed: audio_chunks=%d", session_id, lang, audio_chunks)
        except Exception as e:
            if not shutdown.is_set():
                grpc_code = getattr(e, 'code', None)
                grpc_details = getattr(e, 'details', None)
                if callable(grpc_code) and callable(grpc_details):
                    log.error("[tts:%s:%s] Stream error: %s | grpc_code=%s | grpc_details=%s",
                              session_id, lang, type(e).__name__, grpc_code(), grpc_details())
                else:
                    log.error("[tts:%s:%s] Stream error (%s): %s", session_id, lang, type(e).__name__, e)
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


@app.get("/api/qr")
async def generate_qr(url: str):
    import qrcode
    img = qrcode.make(url, box_size=15)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


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

    busy_response = Response(
        content=json.dumps({'error': 'This session code is already in use. Choose a different code.'}),
        status_code=409,
        media_type='application/json',
    )

    with _lock:
        existing = sessions.get(session_id)
        is_busy = existing is not None and existing.get('broadcaster_connected', False)

    # A broadcaster reconnecting (e.g. after toggling language) may close its
    # WebSocket microseconds before this request lands; the /stream finally
    # block hasn't flipped broadcaster_connected yet. Wait briefly so a
    # language switch isn't rejected with a spurious 409.
    if is_busy and not await _wait_for_broadcaster_disconnect(session_id):
        return busy_response

    # Local HT mode: skip Gladia when local Whisper is loaded and HT is selected
    if src_lang == 'ht' and _local_whisper_model is not None:
        with _lock:
            existing = sessions.get(session_id)
            if existing is not None:
                if existing.get('broadcaster_connected', False):
                    return busy_response
                existing['src_lang'] = 'ht'
                existing['gladia_url'] = ''
                existing['local_ht'] = True
                log.info("init-session: updated to local HT for session %s (listeners preserved)", session_id)
                return {'session_id': session_id, 'local_ht': True}

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
            if session_id in sessions:
                # Another request claimed the slot during the wait above.
                return busy_response
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

    # Re-check the slot is still free before spending a Gladia /v2/live call.
    with _lock:
        existing = sessions.get(session_id)
        if existing is not None and existing.get('broadcaster_connected', False):
            return busy_response

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

        with _lock:
            existing = sessions.get(session_id)
            if existing is not None:
                if existing.get('broadcaster_connected', False):
                    return busy_response
                # Update existing session in-place to preserve listener connections
                existing['src_lang'] = src_lang
                existing['gladia_url'] = gladia_url
                existing.pop('local_ht', None)
                log.info("init-session: updated Gladia config for session %s (listeners preserved)", session_id)
                return {'session_id': session_id}

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
            sessions[session_id] = session

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
    """Broadcaster /stream WebSocket.

    A single router task reads from the browser WS and routes audio chunks
    onto an asyncio.Queue, while control messages are handled inline (or
    surface a signal — switch / disconnect — to the outer loop).

    The outer loop owns the current "backend" (Gladia or local HT) as a
    cancellable task that consumes from the audio queue. On a
    switch_source_language control message, the loop tears down the current
    backend, calls _reconfigure_session_language to swap Gladia URLs (or
    flip to local HT), and re-enters the loop — all without the broadcaster
    needing to close the WebSocket or stop capturing audio.
    """
    await ws.accept()

    try:
        raw_first = await ws.receive_text()
        first_msg = json.loads(raw_first)
        session_id = first_msg.get('session_id', '')
        display_langs = [l for l in first_msg.get('display_langs', ALL_LANGS) if l in ALL_LANGS]
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

    my_conn_id = next(_conn_id_counter)
    with _lock:
        if sessions.get(session_id) is session:
            session['broadcaster_connected'] = True
            session['conn_id'] = my_conn_id
            session['display_langs'] = display_langs

    # Bounded so a stalled backend can't grow audio memory without bound;
    # chunks that arrive while no backend is consuming get dropped.
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=64)
    switch_event = asyncio.Event()
    disconnect_event = asyncio.Event()
    pending_switch: dict = {}

    async def router():
        try:
            while True:
                try:
                    data = await ws.receive()
                except WebSocketDisconnect:
                    return
                if data.get('type') == 'websocket.disconnect':
                    return
                msg_bytes = data.get('bytes')
                if msg_bytes:
                    try:
                        audio_q.put_nowait(msg_bytes)
                    except asyncio.QueueFull:
                        try:
                            audio_q.get_nowait()
                            audio_q.put_nowait(msg_bytes)
                        except (asyncio.QueueEmpty, asyncio.QueueFull):
                            pass
                    continue
                msg_text = data.get('text')
                if not msg_text:
                    continue
                try:
                    ctrl = json.loads(msg_text)
                except Exception:
                    continue
                ctrl_type = ctrl.get('type')
                if ctrl_type == 'display_langs':
                    langs = [l for l in ctrl.get('langs', []) if l in ALL_LANGS]
                    with _lock:
                        if sessions.get(session_id) is session:
                            session['display_langs'] = langs
                elif ctrl_type == 'switch_source_language':
                    new_lang = ctrl.get('language', 'auto')
                    pending_switch['language'] = new_lang
                    switch_event.set()
        finally:
            disconnect_event.set()

    router_task = asyncio.create_task(router())
    backend_task: asyncio.Task | None = None

    try:
        while not disconnect_event.is_set():
            # Drain any audio buffered between backends — playing it through
            # a freshly-opened Gladia stream would be out of sync with the
            # speaker by however long the switch took.
            while True:
                try:
                    audio_q.get_nowait()
                except asyncio.QueueEmpty:
                    break

            if session.get('local_ht'):
                backend_task = asyncio.create_task(
                    _run_local_ht_backend(ws, session_id, session, audio_q)
                )
            else:
                backend_task = asyncio.create_task(
                    _run_gladia_backend(ws, session_id, session, audio_q)
                )

            switch_wait = asyncio.create_task(switch_event.wait())
            disc_wait = asyncio.create_task(disconnect_event.wait())
            done, pending = await asyncio.wait(
                {backend_task, switch_wait, disc_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

            if disconnect_event.is_set():
                break

            if switch_event.is_set():
                switch_event.clear()
                new_lang = pending_switch.pop('language', 'auto')
                log.info("stream: switching session=%s to src_lang=%s", session_id, new_lang)

                if not backend_task.done():
                    backend_task.cancel()
                    try:
                        await backend_task
                    except (asyncio.CancelledError, Exception):
                        pass

                try:
                    await ws.send_text(json.dumps({
                        'type': 'language_switching', 'language': new_lang,
                    }))
                except Exception:
                    pass

                try:
                    await _reconfigure_session_language(session, new_lang)
                except Exception as e:
                    log.error("stream: language switch failed: %s", e)
                    try:
                        await ws.send_text(json.dumps({
                            'type': 'language_switch_failed',
                            'language': new_lang,
                            'error': str(e),
                        }))
                    except Exception:
                        pass
                    # Loop to recreate a backend with the still-current session config
                    continue

                try:
                    await ws.send_text(json.dumps({
                        'type': 'language_switched', 'language': new_lang,
                    }))
                except Exception:
                    pass
                continue

            # Backend exited on its own (Gladia disconnect, network error, …).
            # Without a fresh signal we'd loop forever; let the client reconnect.
            log.warning("stream: backend exited unexpectedly (session=%s)", session_id)
            break
    finally:
        if backend_task is not None and not backend_task.done():
            backend_task.cancel()
            try:
                await backend_task
            except (asyncio.CancelledError, Exception):
                pass
        router_task.cancel()
        try:
            await router_task
        except (asyncio.CancelledError, Exception):
            pass
        with _lock:
            if sessions.get(session_id) is session and session.get('conn_id') == my_conn_id:
                session['broadcaster_connected'] = False
        log.info("stream: closed session=%s", session_id)


async def _run_local_ht_backend(ws: WebSocket, session_id: str, session: dict,
                                audio_q: asyncio.Queue):
    """Consume PCM chunks from audio_q, run them through local Whisper, and
    push transcripts/translations back over ws. Returns on cancellation."""
    log.info("backend: session=%s (local HT)", session_id)

    def send_callback(msg):
        asyncio.run_coroutine_threadsafe(ws.send_text(msg), _loop)

    prev_sentences: list = []
    prev_lock = threading.Lock()
    transcribe_q: queue.Queue = queue.Queue()
    state = {'text_buffer': ''}

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

    sample_buffer: list = []

    def flush():
        nonlocal sample_buffer
        if not sample_buffer:
            return
        audio_np = np.concatenate(sample_buffer).astype(np.float32) / 32768.0
        sample_buffer = []
        transcribe_q.put(audio_np)

    try:
        while True:
            chunk_bytes = await audio_q.get()
            chunk_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
            if len(chunk_int16) == 0:
                continue
            sample_buffer.append(chunk_int16)
            if sum(len(b) for b in sample_buffer) >= _MAX_CHUNK_SAMPLES:
                flush()
    except asyncio.CancelledError:
        pass
    finally:
        flush()
        transcribe_q.put(None)
        worker.join(timeout=30)
        log.info("backend: local HT exited session=%s", session_id)


async def _run_gladia_backend(ws: WebSocket, session_id: str, session: dict,
                              audio_q: asyncio.Queue):
    """Open a Gladia live session, forward audio from audio_q into it, and
    relay transcripts back to ws. Returns when Gladia closes or is cancelled."""
    src_lang = session.get('src_lang', 'auto')
    gladia_url = session.get('gladia_url', '')
    if not gladia_url or not gladia_url.startswith("wss://"):
        log.error("backend: invalid Gladia URL in session: %r", gladia_url)
        return

    log.info("backend: session=%s src_lang=%s connecting to Gladia", session_id, src_lang)
    audio_chunks = 0
    gladia_msgs = 0

    try:
        async with websockets.connect(gladia_url) as gladia_ws:
            log.info("backend: Gladia connection established (session=%s)", session_id)

            async def forward_audio():
                nonlocal audio_chunks
                while True:
                    chunk = await audio_q.get()
                    audio_chunks += 1
                    if audio_chunks % 50 == 1:
                        log.info("backend: relayed %d audio chunks", audio_chunks)
                    await gladia_ws.send(chunk)

            async def forward_transcripts():
                nonlocal gladia_msgs
                async for raw in gladia_ws:
                    gladia_msgs += 1
                    msg_type = None
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
                            # Gladia excludes the source language from translation
                            # targets, so same-language listeners only get audio
                            # via this transcript path. TTS for whatever language
                            # this utterance is in (detected in auto mode, fixed
                            # src_lang otherwise).
                            is_final = (utterance.get('is_final') is True) or (data_obj.get('is_final') is True)
                            if is_final and utterance.get('text'):
                                utt_lang = (utterance.get('language') or '').lower()
                                if utt_lang not in ALL_LANGS:
                                    utt_lang = src_lang if src_lang in ALL_LANGS else None
                                if utt_lang:
                                    _enqueue_tts(session_id, utt_lang, utterance['text'])

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
                        pass

                    if msg_type in ('transcript', 'translation'):
                        if gladia_msgs <= 5 or gladia_msgs % 20 == 0:
                            log.info("backend: gladia msg #%d type=%s", gladia_msgs, msg_type)
                    else:
                        log.info("backend: gladia msg #%d type=%s", gladia_msgs, msg_type)

                    try:
                        await ws.send_text(to_send)
                    except Exception as e:
                        log.warning("backend: browser send error: %s", e)
                        return

            task1 = asyncio.create_task(forward_audio())
            task2 = asyncio.create_task(forward_transcripts())
            try:
                await asyncio.wait({task1, task2}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                task1.cancel()
                task2.cancel()
                await asyncio.gather(task1, task2, return_exceptions=True)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.error("backend: Gladia error (session=%s): %s", session_id, e)
    log.info("backend: Gladia exited session=%s (audio=%d, gladia_msgs=%d)",
             session_id, audio_chunks, gladia_msgs)


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

    # Tell the client the audio format so it can decode chunks without WAV headers.
    await ws.send_text(json.dumps({
        'type': 'audio_config',
        'sample_rate': TTS_SAMPLE_RATE,
        'channels': 1,
        'encoding': 'mulaw',  # MULAW 8-bit; Chirp3 HD streaming only supports MULAW/OGG_OPUS
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
