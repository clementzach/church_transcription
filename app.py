import asyncio
import json
import logging
import os
import queue
import secrets
import string
import threading
import time

import httpx
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
    credentials=_google_api_key.Credentials(os.getenv("GOOGLE_API_KEY"))
)

TRANSLATION_LANGS = ['es', 'ht', 'pt', 'zh', 'fr', 'no']

# ── Google Cloud TTS config (Gemini 2.5 Flash, streaming) ──────────────────
GOOGLE_TTS_MODEL = 'gemini-2.5-flash-tts'
GOOGLE_TTS_VOICES = {
    'es': 'charon',
    'pt': 'aoede',
    'ht': 'kore',
    'zh': 'fenrir',
    'fr': 'puck',
    'no': 'Charon',
}
GOOGLE_TTS_LOCALES = {
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


def _generate_session_id():
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(6))


def _expire_session(session_id):
    """Called by the session timer after SESSION_TIMEOUT_SECS."""
    log.info("Session %s timed out after %ds", session_id, SESSION_TIMEOUT_SECS)
    with _lock:
        session = sessions.pop(session_id, None)
    if session is None:
        return

    error_msg = json.dumps({'type': 'error', 'message': 'Session expired (2-hour limit reached)'})
    for lang in TRANSLATION_LANGS:
        # Signal TTS worker to shut down.
        _force_put(session['tts_queues'][lang], None)
        # Notify listeners: send error then close sentinel.
        for q in list(session['listener_registry'].get(lang, [])):
            try:
                _loop.call_soon_threadsafe(q.put_nowait, error_msg)
                _loop.call_soon_threadsafe(q.put_nowait, None)
            except Exception:
                pass
        session['listener_registry'][lang] = []


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
        call_alive = threading.Event()
        call_alive.set()

        def request_gen():
            """Yields synthesis requests; consumed by gRPC in its background thread."""
            yield config_req
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
                # Caption first so text appears before audio arrives.
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

        if not shutdown.is_set():
            time.sleep(0.2)  # brief pause before reopening

    log.info("TTS worker stopped: session=%s lang=%s", session_id, lang)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/listen")
async def listen():
    return FileResponse("static/listen.html")


@app.post("/init-session")
async def init_session(request: Request):
    config = await request.json()
    log.info("init-session called, config=%s", json.dumps(config))
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
        session_id = _generate_session_id()
        timer = threading.Timer(SESSION_TIMEOUT_SECS, _expire_session, args=(session_id,))
        timer.daemon = True
        session = {
            'start_time': time.time(),
            'listener_registry': {lang: [] for lang in TRANSLATION_LANGS},
            'tts_queues': {lang: queue.Queue(maxsize=1) for lang in TRANSLATION_LANGS},
            'timer': timer,
        }
        with _lock:
            sessions[session_id] = session

        for lang in TRANSLATION_LANGS:
            threading.Thread(
                target=_tts_worker, args=(session_id, lang), daemon=True
            ).start()
        timer.start()

        data = resp.json()
        data['session_id'] = session_id
        log.info("Session created: id=%s timeout=%ds gladia_url=%s",
                 session_id, SESSION_TIMEOUT_SECS, data.get('url') or data.get('websocket_url'))
        return data
    log.error("Gladia init failed: %d %s", resp.status_code, resp.text)
    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()

    # First message from browser: JSON with Gladia URL and session_id.
    try:
        raw_first = await ws.receive_text()
    except WebSocketDisconnect:
        return

    try:
        first_msg = json.loads(raw_first)
        gladia_url = first_msg.get('url', '')
        session_id = first_msg.get('session_id', '')
    except Exception as e:
        log.error("stream: failed to parse first message: %s", e)
        await ws.close()
        return

    if not gladia_url or not gladia_url.startswith("wss://"):
        log.error("stream: invalid or missing Gladia URL: %r", gladia_url)
        await ws.close()
        return

    if not session_id:
        log.error("stream: missing session_id in first message")
        await ws.close()
        return

    with _lock:
        if session_id not in sessions:
            log.error("stream: unknown session_id=%s", session_id)
            await ws.close()
            return

    log.info("stream: session=%s connecting to Gladia at %s", session_id, gladia_url)
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
                                utterance = (msg.get('data') or {}).get('utterance') or {}
                                if utterance.get('text'):
                                    utterance['text'] = _filter_text(utterance['text'])
                                    to_send = json.dumps(msg)

                            elif msg_type == 'translation':
                                data_field = msg.get('data', {})
                                translated = data_field.get('translated_utterance') or {}
                                if translated.get('text'):
                                    translated['text'] = _filter_text(translated['text'])
                                    to_send = json.dumps(msg)
                                lang = (data_field.get('target_language') or '').lower()
                                text = translated.get('text', '')
                                if lang in TRANSLATION_LANGS and text:
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
        if not session or lang not in TRANSLATION_LANGS:
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
