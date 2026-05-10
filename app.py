from gevent import monkey
monkey.patch_all()

import io
import os
import json
import logging
import queue
import secrets
import string
import threading
import time
import wave
import requests
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_sock import Sock
from dotenv import load_dotenv
import websocket
from openai import OpenAI
from filter import filter_text as _filter_text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(funcName)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

app = Flask(__name__)
sock = Sock(app)

GLADIA_API_KEY = os.getenv("GLADIA_API_KEY")

# ── TTS Provider ───────────────────────────────────────────────────────────
# Set TTS_PROVIDER=openai in .env to switch back to OpenAI TTS.
# Default is 'google' (Gemini 2.5 Flash TTS).
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'google').lower()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

if TTS_PROVIDER == 'google' or LOCAL_HAITIAN_PATH:
    from google import genai as _google_genai
    from google.genai import types as _google_types
    google_tts_client = _google_genai.Client(api_key=os.getenv("GOOGLE_TTS_API_KEY"))
    google_llm_client = _google_genai.Client(api_key=os.getenv("GOOGLE_LLM_API_KEY"))

# ── Local HT audio chunking constants ─────────────────────────────────────
_WHISPER_SR = 16000
_MAX_CHUNK_SAMPLES = 30 * _WHISPER_SR       # 30 s hard cap
_SILENCE_MIN_SAMPLES = int(0.8 * _WHISPER_SR)  # 800 ms silence triggers split
_SILENCE_THRESH_RMS = 0.01                  # ≈ −40 dBFS

_TRANSLATION_LANG_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'pt': 'Portuguese',
    'zh': 'Mandarin Chinese',
    'fr': 'French',
    'no': 'Norwegian',
}

TRANSLATION_LANGS = ['es', 'ht', 'pt', 'zh', 'fr', 'no']

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
    'es': 'Charon',
    'pt': 'Aoede',
    'ht': 'Kore',
    'zh': 'Fenrir',
    'fr': 'Puck',
    'no': 'Charon',
}

# ── Session state ────────────────────────────────────────────────────────────
# sessions[session_id] = {
#   'start_time': float,
#   'listener_registry': {lang: [ws, ...]},
#   'tts_queues': {lang: Queue(maxsize=1)},
#   'timer': threading.Timer | None,
# }
SESSION_TIMEOUT_SECS = 2 * 3600  # 2 hours
_lock = threading.Lock()
sessions = {}


def _pcm_to_wav(pcm_data, sample_rate=24000, channels=1, sample_width=2):
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def _tts_generate_audio(lang, text):
    """Generate TTS audio using the configured provider.

    Returns (audio_bytes, mime_type). Switch providers by setting
    TTS_PROVIDER=openai or TTS_PROVIDER=google in .env.
    """
    if TTS_PROVIDER == 'openai':
        response = openai_client.audio.speech.create(
            model='gpt-4o-mini-tts',
            voice=OPENAI_TTS_VOICES[lang],
            input=text,
            instructions=OPENAI_TTS_INSTRUCTIONS[lang],
            response_format='mp3',
        )
        return response.content, 'audio/mpeg'

    # Google Gemini 2.5 Flash TTS
    response = google_tts_client.models.generate_content(
        model='gemini-2.5-flash-preview-tts',
        contents=text,
        config=_google_types.GenerateContentConfig(
            response_modalities=['AUDIO'],
            speech_config=_google_types.SpeechConfig(
                voice_config=_google_types.VoiceConfig(
                    prebuilt_voice_config=_google_types.PrebuiltVoiceConfig(
                        voice_name=GOOGLE_TTS_VOICES[lang],
                    )
                ),
            ),
        ),
    )
    candidates = response.candidates
    if not candidates:
        raise ValueError("Gemini TTS returned no candidates")
    candidate = candidates[0]
    if candidate.content is None:
        finish_reason = getattr(candidate, 'finish_reason', 'unknown')
        raise ValueError(f"Gemini TTS candidate has no content (finish_reason={finish_reason})")
    pcm_data = candidate.content.parts[0].inline_data.data
    return _pcm_to_wav(pcm_data), 'audio/wav'


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


def _transcribe_and_broadcast_ht(session_id, audio_np, prev_sentences, browser_ws):
    """Transcribe a float32 audio array with local Whisper, translate to all target
    languages via Gemini, then push text+TTS to listeners.  Returns the HT text or None."""
    try:
        segments, _ = _local_whisper_model.transcribe(audio_np, language='ht')
        ht_text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        log.error("[local_ht:%s] Whisper error: %s", session_id, e)
        return None

    if not ht_text:
        return None

    ht_text = _filter_text(ht_text)
    log.info("[local_ht:%s] Whisper: %.80s", session_id, ht_text)

    # Forward final HT transcript to broadcaster display
    transcript_msg = json.dumps({
        'type': 'transcript',
        'data': {'utterance': {'text': ht_text, 'is_final': True, 'language': 'ht'}},
    })
    try:
        browser_ws.send(transcript_msg)
    except Exception:
        pass

    # Queue HT text+TTS for Haitian Creole listeners
    _enqueue_tts(session_id, 'ht', ht_text)

    # Translate to every other target language
    for lang in TRANSLATION_LANGS:
        if lang == 'ht':
            continue
        try:
            translated = _translate_ht_to(ht_text, lang, prev_sentences)
            if not translated:
                continue
            translated = _filter_text(translated)
            translation_msg = json.dumps({
                'type': 'translation',
                'data': {
                    'target_language': lang,
                    'translated_utterance': {'text': translated},
                },
            })
            try:
                browser_ws.send(translation_msg)
            except Exception:
                pass
            _enqueue_tts(session_id, lang, translated)
        except Exception as e:
            log.error("[local_ht:%s] Translation to %s failed: %s", session_id, lang, e)

    return ht_text


def _generate_session_id():
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(6))


def _expire_session(session_id):
    """Called by each session's timer after SESSION_TIMEOUT_SECS."""
    log.info("Session %s timed out after %ds", session_id, SESSION_TIMEOUT_SECS)
    with _lock:
        session = sessions.pop(session_id, None)
    if session is None:
        return
    # Stop this session's TTS workers
    for lang in TRANSLATION_LANGS:
        q = session['tts_queues'][lang]
        try:
            q.put_nowait(None)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
    # Notify and close listener WebSockets
    for lang in TRANSLATION_LANGS:
        for ws_conn in list(session['listener_registry'].get(lang, [])):
            try:
                ws_conn.send(json.dumps({'type': 'error', 'message': 'Session expired (2-hour limit reached)'}))
                ws_conn.close()
            except Exception:
                pass
        session['listener_registry'][lang] = []


def _enqueue_tts(session_id, lang, text):
    with _lock:
        session = sessions.get(session_id)
        if not session:
            return
        if not session['listener_registry'].get(lang):
            return
    q = session['tts_queues'][lang]
    try:
        q.put_nowait(text)
    except queue.Full:
        # Drop the stale pending item and replace with the latest
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(text)
        except queue.Full:
            pass


def _tts_worker(session_id, lang):
    """Background thread per (session, language): calls TTS provider and broadcasts to listeners."""
    log.info("TTS worker started for session=%s lang=%s provider=%s", session_id, lang, TTS_PROVIDER)
    # Hold a reference to this session's queue so we can block on it even after
    # the session is removed from the dict (expiry will enqueue None to unblock us).
    with _lock:
        session = sessions.get(session_id)
        if not session:
            return
        q = session['tts_queues'][lang]

    while True:
        text = q.get()
        if text is None:
            break

        log.info("[tts:%s:%s] Sending text to listeners: %.60s", session_id, lang, text)

        # Send text so the listener can display it immediately
        with _lock:
            session = sessions.get(session_id)
            if not session:
                break
            listeners = list(session['listener_registry'].get(lang, []))
        dead = []
        for ws_conn in listeners:
            try:
                ws_conn.send(json.dumps({'type': 'text', 'text': text}))
            except Exception:
                dead.append(ws_conn)
        if dead:
            with _lock:
                session = sessions.get(session_id)
                if session:
                    for d in dead:
                        try:
                            session['listener_registry'][lang].remove(d)
                        except ValueError:
                            pass

        # Skip API call if all listeners disconnected while text was queued
        with _lock:
            session = sessions.get(session_id)
            if not session or not session['listener_registry'].get(lang):
                log.info("[tts:%s:%s] No listeners, skipping TTS API call", session_id, lang)
                continue

        # Generate audio via the configured provider
        try:
            log.info("[tts:%s:%s] Calling %s TTS", session_id, lang, TTS_PROVIDER)
            audio_bytes, mime_type = _tts_generate_audio(lang, text)
            log.info("[tts:%s:%s] Got %d bytes of audio (%s)", session_id, lang, len(audio_bytes), mime_type)
        except Exception as e:
            log.error("[tts:%s:%s] TTS error: %s", session_id, lang, e)
            continue

        with _lock:
            session = sessions.get(session_id)
            if not session:
                continue
            listeners = list(session['listener_registry'].get(lang, []))
        dead = []
        for ws_conn in listeners:
            try:
                ws_conn.send(json.dumps({'type': 'audio_info', 'mime_type': mime_type}))
                ws_conn.send(audio_bytes)
            except Exception:
                dead.append(ws_conn)
        if dead:
            with _lock:
                session = sessions.get(session_id)
                if session:
                    for d in dead:
                        try:
                            session['listener_registry'][lang].remove(d)
                        except ValueError:
                            pass

    log.info("TTS worker stopped for session=%s lang=%s", session_id, lang)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/listen")
def listen():
    return send_from_directory("static", "listen.html")


def _create_session(local_ht=False):
    """Create a new session entry, start TTS workers, and schedule expiry. Returns session_id."""
    session_id = _generate_session_id()
    session = {
        'start_time': time.time(),
        'listener_registry': {lang: [] for lang in TRANSLATION_LANGS},
        'tts_queues': {lang: queue.Queue(maxsize=1) for lang in TRANSLATION_LANGS},
        'timer': None,
        'local_ht': local_ht,
    }
    with _lock:
        sessions[session_id] = session
    for lang in TRANSLATION_LANGS:
        threading.Thread(target=_tts_worker, args=(session_id, lang), daemon=True).start()
    timer = threading.Timer(SESSION_TIMEOUT_SECS, _expire_session, args=(session_id,))
    timer.daemon = True
    timer.start()
    session['timer'] = timer
    return session_id


@app.route("/init-session", methods=["POST"])
def init_session():
    config = request.get_json()
    log.info("init-session called, config=%s", json.dumps(config))

    # Local HT mode: Whisper model loaded AND broadcaster pinned to Haitian Creole
    lang_config = (config or {}).get('language_config', {})
    is_local_ht = (
        _local_whisper_model is not None
        and lang_config.get('languages') == ['ht']
    )

    if is_local_ht:
        session_id = _create_session(local_ht=True)
        log.info("Session created (local HT): id=%s timeout=%ds", session_id, SESSION_TIMEOUT_SECS)
        return jsonify({'session_id': session_id, 'local_ht': True})

    resp = requests.post(
        "https://api.gladia.io/v2/live",
        headers={
            "X-Gladia-Key": GLADIA_API_KEY,
            "Content-Type": "application/json",
        },
        json=config,
        timeout=10,
    )
    log.info("Gladia /v2/live response: status=%d", resp.status_code)
    if resp.ok:
        session_id = _create_session(local_ht=False)
        data = resp.json()
        data['session_id'] = session_id
        log.info("Session created: id=%s timeout=%ds gladia_url=%s",
                 session_id, SESSION_TIMEOUT_SECS, data.get('url') or data.get('websocket_url'))
        return jsonify(data)
    log.error("Gladia init failed: %d %s", resp.status_code, resp.text)
    return (resp.content, resp.status_code, {"Content-Type": "application/json"})


@sock.route("/stream")
def stream(browser_ws):
    # First message from browser: JSON with Gladia URL and session_id
    raw_first = browser_ws.receive()
    gladia_url = None
    session_id = None
    try:
        first_msg = json.loads(raw_first)
        gladia_url = first_msg.get('url', '')
        session_id = first_msg.get('session_id', '')
    except Exception:
        # Legacy fallback: plain URL string (no session routing)
        gladia_url = raw_first
        session_id = None

    if not gladia_url or not gladia_url.startswith("wss://"):
        log.error("stream: invalid or missing Gladia URL: %r", gladia_url)
        browser_ws.close()
        return

    if not session_id:
        log.error("stream: missing session_id in first message")
        browser_ws.close()
        return

    with _lock:
        if session_id not in sessions:
            log.error("stream: unknown session_id=%s", session_id)
            browser_ws.close()
            return

    log.info("stream: session=%s connecting to Gladia at %s", session_id, gladia_url)
    try:
        gladia_ws = websocket.create_connection(gladia_url)
    except Exception as e:
        log.error("stream: failed to connect to Gladia: %s", e)
        browser_ws.close()
        return

    log.info("stream: Gladia connection established")
    stop_event = threading.Event()
    audio_chunks = 0

    def forward_audio():
        nonlocal audio_chunks
        try:
            while not stop_event.is_set():
                data = browser_ws.receive()
                if data is None:
                    log.info("stream/forward_audio: browser closed connection")
                    break
                if isinstance(data, bytes):
                    audio_chunks += 1
                    if audio_chunks % 50 == 1:
                        log.info("stream/forward_audio: relayed %d audio chunks", audio_chunks)
                    gladia_ws.send_binary(data)
        except Exception as e:
            log.warning("stream/forward_audio: exception: %s", e)
        finally:
            stop_event.set()

    audio_thread = threading.Thread(target=forward_audio, daemon=True)
    audio_thread.start()

    gladia_msgs = 0
    try:
        while not stop_event.is_set():
            try:
                raw = gladia_ws.recv()
            except Exception as e:
                log.warning("stream: gladia recv error: %s", e)
                break
            if raw is None:
                log.info("stream: Gladia closed connection")
                break
            gladia_msgs += 1

            # Parse the message, filter any text fields, then forward to browser.
            # Filtering here means the recorder display and listeners all see clean text.
            to_send = raw
            try:
                msg = json.loads(raw)
                msg_type = msg.get('type')
                if gladia_msgs <= 5 or msg_type in ('transcript', 'translation'):
                    log.info("stream: gladia msg #%d type=%s", gladia_msgs, msg_type)

                if msg_type == 'transcript':
                    utterance = (msg.get('data') or {}).get('utterance') or {}
                    if utterance.get('text'):
                        utterance['text'] = _filter_text(utterance['text'])
                        to_send = json.dumps(msg)

                elif msg_type == 'translation':
                    data = msg.get('data', {})
                    translated = data.get('translated_utterance') or {}
                    if translated.get('text'):
                        translated['text'] = _filter_text(translated['text'])
                        to_send = json.dumps(msg)
                    # Enqueue filtered text for TTS/listener delivery
                    lang = (data.get('target_language') or '').lower()
                    text = translated.get('text', '')
                    if lang in TRANSLATION_LANGS and text:
                        _enqueue_tts(session_id, lang, text)
            except Exception:
                pass  # on any parse error, forward the original raw message

            try:
                browser_ws.send(to_send)
            except Exception as e:
                log.warning("stream: browser send error: %s", e)
                break

    finally:
        log.info("stream: closing (audio_chunks=%d, gladia_msgs=%d)", audio_chunks, gladia_msgs)
        stop_event.set()
        try:
            gladia_ws.close()
        except Exception:
            pass


@sock.route("/listen-stream")
def listen_stream(ws_conn):
    # First message: JSON with session_id and language
    try:
        first = ws_conn.receive()
        msg = json.loads(first)
        session_id = msg.get('session_id', '')
        lang = msg.get('language', '').lower()
    except Exception as e:
        log.error("listen-stream: failed to parse first message: %s", e)
        ws_conn.close()
        return

    log.info("listen-stream: session_id=%s lang=%s", session_id, lang)

    with _lock:
        session = sessions.get(session_id)
        if not session or lang not in TRANSLATION_LANGS:
            log.warning("listen-stream: invalid session_id=%s or lang=%s", session_id, lang)
            ws_conn.send(json.dumps({'type': 'error', 'message': 'Invalid session ID or language'}))
            ws_conn.close()
            return
        session['listener_registry'][lang].append(ws_conn)
        log.info("listen-stream: registered listener for session=%s lang=%s (total=%d)",
                 session_id, lang, len(session['listener_registry'][lang]))

    # Hold connection open; remove on disconnect
    try:
        while True:
            data = ws_conn.receive()
            if data is None:
                break
    finally:
        with _lock:
            session = sessions.get(session_id)
            if session:
                try:
                    session['listener_registry'][lang].remove(ws_conn)
                    log.info("listen-stream: removed listener for session=%s lang=%s", session_id, lang)
                except ValueError:
                    pass


@sock.route("/stream-local")
def stream_local(browser_ws):
    """WebSocket endpoint for local Haitian Creole Whisper transcription.

    Accumulates PCM audio (int16, 16 kHz, mono), splits on 800 ms silence or
    30 s max, then transcribes with the local Whisper model and translates to
    all other languages via Gemini.
    """
    raw_first = browser_ws.receive()
    session_id = None
    try:
        first_msg = json.loads(raw_first)
        session_id = first_msg.get('session_id', '')
    except Exception:
        browser_ws.close()
        return

    with _lock:
        if session_id not in sessions:
            log.error("stream_local: unknown session_id=%s", session_id)
            browser_ws.close()
            return

    log.info("stream_local: session=%s", session_id)

    # Rolling context of the last two HT sentences, shared with the worker thread.
    prev_sentences = []
    prev_lock = threading.Lock()

    transcribe_q = queue.Queue()

    def transcription_worker():
        while True:
            audio_np = transcribe_q.get()
            if audio_np is None:
                break
            with prev_lock:
                prev = list(prev_sentences)
            ht_text = _transcribe_and_broadcast_ht(session_id, audio_np, prev, browser_ws)
            if ht_text:
                with prev_lock:
                    prev_sentences.append(ht_text)
                    if len(prev_sentences) > 2:
                        del prev_sentences[:-2]

    worker = threading.Thread(target=transcription_worker, daemon=True)
    worker.start()

    sample_buffer = []   # list of np.ndarray (float32)
    silence_run = 0      # consecutive silent samples
    has_content = False  # seen at least one non-silent chunk

    def flush():
        nonlocal sample_buffer, silence_run, has_content
        if not sample_buffer:
            return
        audio_np = np.concatenate(sample_buffer)
        sample_buffer = []
        silence_run = 0
        has_content = False
        transcribe_q.put(audio_np)

    try:
        while True:
            data = browser_ws.receive()
            if data is None:
                break
            if not isinstance(data, bytes):
                continue  # ignore stop_recording or other text frames

            chunk_int16 = np.frombuffer(data, dtype=np.int16)
            if len(chunk_int16) == 0:
                continue
            chunk_float = chunk_int16.astype(np.float32) / 32768.0

            rms = float(np.sqrt(np.mean(chunk_float ** 2)))
            if rms < _SILENCE_THRESH_RMS:
                silence_run += len(chunk_float)
            else:
                silence_run = 0
                has_content = True

            sample_buffer.append(chunk_float)
            total = sum(len(b) for b in sample_buffer)

            split_on_silence = has_content and silence_run >= _SILENCE_MIN_SAMPLES
            if split_on_silence or total >= _MAX_CHUNK_SAMPLES:
                flush()

    finally:
        if has_content and sample_buffer:
            flush()
        transcribe_q.put(None)
        worker.join(timeout=30)
        log.info("stream_local: closed session=%s", session_id)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
