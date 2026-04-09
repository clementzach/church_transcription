# Church Transcription — CLAUDE.md

## What this project is

A live transcription and translation web app for church services. A broadcaster records audio; Gladia transcribes and translates it in real time; Google Cloud TTS (Gemini 2.5 Flash) converts translations to speech; remote listeners hear audio and see captions in their chosen language (Spanish, Haitian Creole, Portuguese, Mandarin, French, or Norwegian).

## Architecture

```
Browser (broadcaster)
  └─ WebSocket /stream ──► app.py ──► Gladia WebSocket (transcription + translation)
                                 │
                                 └─ _enqueue_tts(lang, text)
                                        │
                              queue.Queue(maxsize=1) per language   ← drops stale items
                                        │
                              _tts_worker(lang) thread ──► Google Cloud TTS gRPC stream
                                        │
                              asyncio.Queue per listener
                                        │
                              WebSocket /listen-stream ──► Browser (listener)
```

**Multiple concurrent sessions are supported.** All session state lives in the `sessions` dict (keyed by session ID), protected by `_lock`. Each session has its own `listener_registry`, `tts_queues`, and expiry timer — sessions are fully isolated.

## Key files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI backend — session management, WebSocket relay, TTS workers |
| `static/index.html` | Broadcaster UI — mic capture, live caption grid (6 languages), how-to guide modal |
| `static/listen.html` | Listener UI — session login, caption display, audio playback, how-to guide modal |
| `filter.py` | Text filter applied to transcripts and translations before forwarding |
| `church-transcription.service` | Systemd unit for production (uvicorn, single process) |

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in GLADIA_API_KEY, GOOGLE_API_KEY
python app.py          # listens on :5001
```

## Session lifecycle

1. Broadcaster hits `POST /init-session` → gets a Gladia WebSocket URL and a 6-char session ID.
2. Broadcaster opens `WebSocket /stream`, sends the Gladia URL and session ID as the first JSON message, then streams PCM audio.
3. Listeners open `/listen`, enter session ID + language, connect via `WebSocket /listen-stream`.
4. After **2 hours** the session timer fires (`_expire_session`): all listener WebSockets receive an error message and are closed; the session ID is invalidated.

## TTS worker (`_tts_worker`)

One thread per `(session_id, lang)`. Keeps a **single long-lived** `streaming_synthesize` gRPC call open across multiple utterances to stay within Google's ~200 calls/day rate limit. The stream closes after `TTS_INACTIVITY_TIMEOUT` (30 s) of silence and reopens on the next utterance.

- Text arrives on a `queue.Queue(maxsize=1)`. If a new translation arrives while one is already pending, the stale item is replaced (`_force_put`). This keeps audio in sync with the live speaker.
- The caption JSON is broadcast to listeners **before** synthesis begins, so text appears immediately while audio is being generated.
- Each raw PCM chunk (24 kHz / Int16 / mono) is broadcast to listeners as it arrives from gRPC — no buffering until utterance end.

## Listener delivery

Each `listen_stream` WebSocket handler owns an `asyncio.Queue`. TTS worker threads push items via `_loop.call_soon_threadsafe(q.put_nowait, item)`, where `_loop` is the main event loop captured at startup. The handler awaits items and forwards them; a `None` sentinel signals close (session expiry). This keeps all WebSocket sends on the event loop.

## Audio playback in the listener UI

On connect, the server sends an `audio_config` JSON message (sample rate, channels, bit depth). All subsequent binary messages are raw PCM chunks. The browser uses `AudioContext` with scheduled playback: each chunk is converted Int16→Float32, loaded into an `AudioBuffer`, and scheduled to start exactly when the previous chunk ends — giving gapless streaming with low latency.

Browsers block autoplay without a prior user gesture. A **"▶ Start Audio"** button gates playback (`audioUnlocked`). Chunks that arrive before the button is clicked are held in `audioQueue` (capped at 30 to prevent backlog).

## TTS queue behaviour

`queue.Queue(maxsize=1)` per language. `_force_put` drains any stale pending item before inserting the new one. This keeps listeners in sync with the live speaker rather than falling behind.

## Deployment notes

- Single uvicorn process required — session state is in-process.
- Reverse proxy (nginx) should forward `/church_transcription/` prefix and upgrade WebSocket headers.
- The `_loop` global is set in the `startup` event hook; all TTS thread→asyncio communication goes through it.
