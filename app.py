import os
import json
import threading
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_sock import Sock
from dotenv import load_dotenv
import websocket

load_dotenv()

app = Flask(__name__)
sock = Sock(app)

GLADIA_API_KEY = os.getenv("GLADIA_API_KEY")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/init-session", methods=["POST"])
def init_session():
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
    return (resp.content, resp.status_code, {"Content-Type": "application/json"})


@sock.route("/stream")
def stream(browser_ws):
    # First message from browser is the Gladia WebSocket URL
    gladia_url = browser_ws.receive()
    if not gladia_url or not gladia_url.startswith("wss://"):
        browser_ws.close()
        return

    gladia_ws = websocket.create_connection(gladia_url)

    stop_event = threading.Event()

    def forward_audio():
        """Read binary audio from browser and send to Gladia."""
        try:
            while not stop_event.is_set():
                data = browser_ws.receive()
                if data is None:
                    break
                if isinstance(data, bytes):
                    gladia_ws.send_binary(data)
                # Ignore unexpected text frames from browser during streaming
        except Exception:
            pass
        finally:
            stop_event.set()

    audio_thread = threading.Thread(target=forward_audio, daemon=True)
    audio_thread.start()

    # Main thread: read JSON from Gladia and forward to browser
    try:
        while not stop_event.is_set():
            try:
                raw = gladia_ws.recv()
            except Exception:
                break
            if raw is None:
                break
            try:
                browser_ws.send(raw)
            except Exception:
                break
    finally:
        stop_event.set()
        try:
            gladia_ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
