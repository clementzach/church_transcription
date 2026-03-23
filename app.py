import os
import json
import queue
import secrets
import string
import threading
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_sock import Sock
from dotenv import load_dotenv
import websocket
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
sock = Sock(app)

GLADIA_API_KEY = os.getenv("GLADIA_API_KEY")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TRANSLATION_LANGS = ['es', 'ht', 'pt']

# Voice and instructions per language for gpt-4o-mini-tts
TTS_VOICES = {
    'es': 'nova',
    'pt': 'shimmer',
    'ht': 'alloy',
}
TTS_INSTRUCTIONS = {
    'es': 'Speak naturally and clearly in Spanish.',
    'pt': 'Fale de forma natural e clara em português.',
    'ht': 'Ou pale kreyòl tankou yon natif natal',
}

# ── Session state (one active recording session at a time) ─────────────────
_lock = threading.Lock()
current_session_id = None
listener_registry = {lang: [] for lang in TRANSLATION_LANGS}  # lang -> [ws, ...]

# One TTS queue per language; maxsize=1 so we drop stale segments
tts_queues = {lang: queue.Queue(maxsize=1) for lang in TRANSLATION_LANGS}


def _generate_session_id():
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(6))


def _enqueue_tts(lang, text):
    # Skip entirely if no listeners are connected for this language
    with _lock:
        if not listener_registry.get(lang):
            return
    q = tts_queues[lang]
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


def _tts_worker(lang):
    """Background thread per language: calls OpenAI TTS and sends audio to listeners."""
    while True:
        text = tts_queues[lang].get()
        if text is None:
            break

        # First send the text so the listener can display it immediately
        with _lock:
            listeners = list(listener_registry.get(lang, []))
        dead = []
        for ws_conn in listeners:
            try:
                ws_conn.send(json.dumps({'type': 'text', 'text': text}))
            except Exception:
                dead.append(ws_conn)
        if dead:
            with _lock:
                for d in dead:
                    try:
                        listener_registry[lang].remove(d)
                    except ValueError:
                        pass

        # Skip API call if all listeners disconnected while text was queued
        with _lock:
            if not listener_registry.get(lang):
                continue

        # Then generate and send audio
        try:
            response = openai_client.audio.speech.create(
                model='gpt-4o-mini-tts',
                voice=TTS_VOICES[lang],
                input=text,
                instructions=TTS_INSTRUCTIONS[lang],
                response_format='mp3',
            )
            audio_bytes = response.content
        except Exception as e:
            print(f"TTS error [{lang}]: {e}")
            continue

        with _lock:
            listeners = list(listener_registry.get(lang, []))
        dead = []
        for ws_conn in listeners:
            try:
                ws_conn.send(audio_bytes)
            except Exception:
                dead.append(ws_conn)
        if dead:
            with _lock:
                for d in dead:
                    try:
                        listener_registry[lang].remove(d)
                    except ValueError:
                        pass


# Start one TTS worker thread per translation language
for _lang in TRANSLATION_LANGS:
    threading.Thread(target=_tts_worker, args=(_lang,), daemon=True).start()


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/listen")
def listen():
    return send_from_directory("static", "listen.html")


@app.route("/init-session", methods=["POST"])
def init_session():
    global current_session_id
    config = request.get_json()
    resp = requests.post(
        "https://api.gladia.io/v2/live",
        headers={
            "X-Gladia-Key": GLADIA_API_KEY,
            "Content-Type": "application/json",
        },
        json=config,
        timeout=10,
    )
    if resp.status_code == 200:
        session_id = _generate_session_id()
        with _lock:
            current_session_id = session_id
            for lang in TRANSLATION_LANGS:
                listener_registry[lang] = []
        data = resp.json()
        data['session_id'] = session_id
        return jsonify(data)
    return (resp.content, resp.status_code, {"Content-Type": "application/json"})


@sock.route("/stream")
def stream(browser_ws):
    # First message from browser: Gladia WebSocket URL
    gladia_url = browser_ws.receive()
    if not gladia_url or not gladia_url.startswith("wss://"):
        browser_ws.close()
        return

    gladia_ws = websocket.create_connection(gladia_url)
    stop_event = threading.Event()

    def forward_audio():
        """Read binary audio from browser and relay to Gladia."""
        try:
            while not stop_event.is_set():
                data = browser_ws.receive()
                if data is None:
                    break
                if isinstance(data, bytes):
                    gladia_ws.send_binary(data)
        except Exception:
            pass
        finally:
            stop_event.set()

    audio_thread = threading.Thread(target=forward_audio, daemon=True)
    audio_thread.start()

    try:
        while not stop_event.is_set():
            try:
                raw = gladia_ws.recv()
            except Exception:
                break
            if raw is None:
                break
            # Relay to recorder browser (unchanged)
            try:
                browser_ws.send(raw)
            except Exception:
                break
            # Intercept committed translation messages to trigger TTS
            try:
                msg = json.loads(raw)
                if msg.get('type') == 'translation':
                    data = msg.get('data', {})
                    lang = (data.get('target_language') or '').lower()
                    text = (data.get('translated_utterance') or {}).get('text', '')
                    if lang in tts_queues and text:
                        _enqueue_tts(lang, text)
            except Exception:
                pass
    finally:
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
    except Exception:
        ws_conn.close()
        return

    with _lock:
        if session_id != current_session_id or lang not in listener_registry:
            ws_conn.send(json.dumps({'type': 'error', 'message': 'Invalid session ID or language'}))
            ws_conn.close()
            return
        listener_registry[lang].append(ws_conn)

    # Hold connection open; remove on disconnect
    try:
        while True:
            data = ws_conn.receive()
            if data is None:
                break
    finally:
        with _lock:
            try:
                listener_registry[lang].remove(ws_conn)
            except ValueError:
                pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
