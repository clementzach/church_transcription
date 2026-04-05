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
from google.cloud import texttospeech as _texttospeech
from google.api_core import client_options as _client_options
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

# ── Google Cloud TTS client (singleton, shared across all sessions) ─────────
_tts_client = _texttospeech.TextToSpeechClient(
    client_options=_client_options.ClientOptions(
        api_key=os.getenv("GOOGLE_API_KEY")
    )
)

TRANSLATION_LANGS = ['es', 'ht', 'pt', 'zh', 'fr']

# ── Google Cloud TTS config (Gemini 2.5 Flash, streaming) ──────────────────
GOOGLE_TTS_MODEL = 'gemini-2.5-flash-tts'
GOOGLE_TTS_VOICES = {
    'es': 'charon',
    'pt': 'aoede',
    'ht': 'kore',
    'zh': 'fenrir',
    'fr': 'puck',
}
GOOGLE_TTS_LOCALES = {
    'es': 'es-US',
    'pt': 'pt-BR',
    'ht': 'fr-HT',
    'zh': 'cmn-CN',
    'fr': 'fr-FR',
}

# After this many seconds with no new text, close the stream and reopen on demand.
TTS_INACTIVITY_TIMEOUT = 30.0
# Silence gap (seconds) between audio chunks that signals an utterance boundary.
TTS_UTTERANCE_GAP = 0.2

# ── Session state ─────────────────────────────────────────────────────────────
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


def _broadcast(session_id, lang, message):
    """Send a JSON str or audio bytes to every current listener for (session, lang).

    Dead connections are pruned from the registry.
    """
    with _lock:
        session = sessions.get(session_id)
        if not session:
            return
        listeners = list(session['listener_registry'].get(lang, []))

    dead = []
    for ws_conn in listeners:
        try:
            ws_conn.send(message)
        except Exception:
            dead.append(ws_conn)

    if dead:
        with _lock:
            session = sessions.get(session_id)
            if session:
                reg = session['listener_registry'].get(lang, [])
                for d in dead:
                    try:
                        reg.remove(d)
                    except ValueError:
                        pass


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
    # Send None sentinel to each TTS worker so they shut down cleanly.
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
    # Notify and close listener WebSockets.
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
        # Drop the stale pending item and replace with the latest.
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(text)
        except queue.Full:
            pass


def _tts_worker(session_id, lang):
    """Maintains one long-lived Google Cloud TTS streaming connection per (session, lang).

    Architecture
    ────────────
    Each iteration of the outer while loop owns one gRPC streaming call:

      request_gen (gRPC background thread)
        • Reads text items from tts_queues[lang] (maxsize=1, drops stale utterances).
        • Broadcasts the caption to listeners before yielding to TTS so text appears
          immediately while audio is still being synthesised.
        • Returns (closing the stream) after TTS_INACTIVITY_TIMEOUT seconds of silence
          or when the None shutdown sentinel is received.

      response_reader (daemon thread)
        • Iterates the streaming responses and puts each PCM chunk into audio_q.
        • Puts a None sentinel when the stream ends (normally or on error).

      _tts_worker greenlet (main loop)
        • Reads audio_q with a TTS_UTTERANCE_GAP timeout.
        • A timeout means no audio arrived for that long → utterance boundary.
          Accumulated PCM chunks are wrapped in a single WAV and broadcast.
        • A None sentinel means the stream ended; flushes any remaining audio,
          then loops back to open a new stream (or exits if shutdown is set).

    The gRPC channel is reused across calls because _tts_client is a singleton.
    """
    log.info("TTS worker started: session=%s lang=%s", session_id, lang)

    with _lock:
        session = sessions.get(session_id)
        if not session:
            return
        tts_q = session['tts_queues'][lang]

    config_req = _texttospeech.StreamingSynthesizeRequest(
        streaming_config=_texttospeech.StreamingSynthesizeConfig(
            voice=_texttospeech.VoiceSelectionParams(
                name=GOOGLE_TTS_VOICES[lang],
                language_code=GOOGLE_TTS_LOCALES[lang],
                model_name=GOOGLE_TTS_MODEL,
            )
        )
    )

    shutdown = threading.Event()

    while not shutdown.is_set():
        # Per-stream-attempt state.  Default-argument binding (e.g. _alive=call_alive)
        # ensures each closure captures *this* iteration's objects, not a later one.
        call_alive = threading.Event()
        call_alive.set()
        audio_q = queue.Queue()

        def request_gen(_alive=call_alive):
            """Yields TTS requests; runs in gRPC's background thread."""
            yield config_req
            inactive_since = None
            while _alive.is_set():
                try:
                    text = tts_q.get(timeout=0.2)
                    inactive_since = None
                except queue.Empty:
                    if inactive_since is None:
                        inactive_since = time.time()
                    elif time.time() - inactive_since >= TTS_INACTIVITY_TIMEOUT:
                        log.info("[tts:%s:%s] Inactivity timeout — closing stream", session_id, lang)
                        _alive.clear()
                        return
                    continue
                if text is None:
                    shutdown.set()
                    _alive.clear()
                    return
                # Broadcast caption before audio so text appears immediately.
                _broadcast(session_id, lang, json.dumps({'type': 'text', 'text': text}))
                log.info("[tts:%s:%s] → TTS: %.60s", session_id, lang, text)
                yield _texttospeech.StreamingSynthesizeRequest(
                    input=_texttospeech.StreamingSynthesisInput(text=text)
                )

        def response_reader(_rg=request_gen, _aq=audio_q):
            """Iterates the TTS stream; runs in its own daemon thread."""
            try:
                for response in _tts_client.streaming_synthesize(_rg()):
                    if response.audio_content:
                        _aq.put(response.audio_content)
            except Exception as e:
                if not shutdown.is_set():
                    log.error("[tts:%s:%s] Stream error: %s", session_id, lang, e)
            finally:
                _aq.put(None)  # always signal end-of-stream

        reader_thread = threading.Thread(target=response_reader, daemon=True)
        reader_thread.start()

        # Accumulate PCM chunks and flush as one WAV per utterance.
        # A TTS_UTTERANCE_GAP timeout on the queue means no audio arrived for
        # that long — the synthesiser is idle between utterances — so we flush.
        current_chunks = []
        try:
            while True:
                try:
                    chunk = audio_q.get(timeout=TTS_UTTERANCE_GAP)
                except queue.Empty:
                    # Utterance boundary: flush accumulated audio.
                    if current_chunks:
                        log.info("[tts:%s:%s] Flushing %d PCM chunks as WAV",
                                 session_id, lang, len(current_chunks))
                        _broadcast(session_id, lang, _pcm_to_wav(b''.join(current_chunks)))
                        current_chunks = []
                    continue
                if chunk is None:
                    break  # stream ended
                current_chunks.append(chunk)
        finally:
            # Flush any tail audio that arrived before the None sentinel.
            if current_chunks:
                _broadcast(session_id, lang, _pcm_to_wav(b''.join(current_chunks)))
            call_alive.clear()

        reader_thread.join(timeout=2.0)

        if not shutdown.is_set():
            time.sleep(0.2)  # brief pause before reconnecting

    log.info("TTS worker stopped: session=%s lang=%s", session_id, lang)


# ── Routes ────────────────────────────────────────────────────────────────────

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

        # Start one TTS worker per language for this session.
        for lang in TRANSLATION_LANGS:
            threading.Thread(
                target=_tts_worker, args=(session_id, lang), daemon=True
            ).start()

        # Schedule automatic expiry.
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
    # First message from browser: JSON with Gladia URL and session_id.
    raw_first = browser_ws.receive()
    gladia_url = None
    session_id = None
    try:
        first_msg = json.loads(raw_first)
        gladia_url = first_msg.get('url', '')
        session_id = first_msg.get('session_id', '')
    except Exception:
        # Legacy fallback: plain URL string (no session routing).
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
                    # Enqueue filtered text for TTS/listener delivery.
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
    # First message: JSON with session_id and language.
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

    # Hold connection open; remove on disconnect.
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
