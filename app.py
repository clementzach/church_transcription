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

if TTS_PROVIDER == 'google':
    from google import genai as _google_genai
    from google.genai import types as _google_types
    google_client = _google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

TRANSLATION_LANGS = ['es', 'ht', 'pt', 'zh']

# ── OpenAI TTS config (gpt-4o-mini-tts) ───────────────────────────────────
OPENAI_TTS_VOICES = {
    'es': 'nova',
    'pt': 'shimmer',
    'ht': 'alloy',
    'zh': 'nova',
}
OPENAI_TTS_INSTRUCTIONS = {
    'es': 'Speak naturally and clearly in Spanish.',
    'pt': 'Fale de forma natural e clara em português.',
    'ht': 'Ou pale kreyòl tankou yon natif natal',
    'zh': '请用标准普通话自然流利地朗读，像母语人士一样说话，发音清晰，语调自然。',
}

# ── Google Gemini 2.5 Flash TTS config ────────────────────────────────────
GOOGLE_TTS_VOICES = {
    'es': 'Charon',
    'pt': 'Aoede',
    'ht': 'Kore',
    'zh': 'Fenrir',
}
GOOGLE_TTS_INSTRUCTIONS = {
    'es': (
        'You are a native Spanish speaker from Latin America. '
        'Speak naturally and fluently with authentic Latin American Spanish '
        'pronunciation, rhythm, and intonation. Sound warm, clear, and conversational.'
    ),
    'pt': (
        'Você é um falante nativo de português brasileiro. '
        'Fale de forma natural e fluente com pronúncia, ritmo e entonação '
        'autênticos do português brasileiro. Soe caloroso, claro e conversacional.'
    ),
    'ht': (
        'Ou se yon moun ki pale kreyòl ayisyen kòm lang manman ou. '
        'Pale natirèlman ak aksan, ritem ak entònasyon otantik kreyòl ayisyen. '
        'Sone cho, klè epi konvèsasyonèl.'
    ),
    'zh': (
        '你是一位普通话母语者。请用自然流利的标准普通话朗读，'
        '发音标准，语调自然，声音温暖清晰，像日常对话一样亲切。'
    ),
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
    response = google_client.models.generate_content(
        model='gemini-2.5-flash-tts',
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
            system_instruction=GOOGLE_TTS_INSTRUCTIONS[lang],
        ),
    )
    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    return _pcm_to_wav(pcm_data), 'audio/wav'


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


@app.route("/init-session", methods=["POST"])
def init_session():
    config = request.get_json()
    log.info("init-session called, config=%s", json.dumps(config))
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
        session_id = _generate_session_id()
        session = {
            'start_time': time.time(),
            'listener_registry': {lang: [] for lang in TRANSLATION_LANGS},
            'tts_queues': {lang: queue.Queue(maxsize=1) for lang in TRANSLATION_LANGS},
            'timer': None,
        }
        with _lock:
            sessions[session_id] = session

        # Start TTS workers for this session
        for lang in TRANSLATION_LANGS:
            threading.Thread(
                target=_tts_worker, args=(session_id, lang), daemon=True
            ).start()

        # Schedule automatic expiry tied to this session
        timer = threading.Timer(SESSION_TIMEOUT_SECS, _expire_session, args=(session_id,))
        timer.daemon = True
        timer.start()
        session['timer'] = timer

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
