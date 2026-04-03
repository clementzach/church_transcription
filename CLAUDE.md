# Church Transcription — CLAUDE.md

## What this project is

A live transcription and translation web app for church services. A broadcaster records audio; Gladia transcribes and translates it in real time; OpenAI TTS converts translations to speech; remote listeners hear audio and see captions in their chosen language (Spanish, Haitian Creole, or Portuguese).

## Architecture

```
Browser (broadcaster)
  └─ WebSocket /stream ──► app.py ──► Gladia WebSocket (transcription + translation)
                                 │
                                 └─ _enqueue_tts(lang, text)
                                        │
                              threading.Queue per language (maxsize=1)
                                        │
                              _tts_worker(lang) ──► OpenAI TTS API
                                        │
                              WebSocket /listen-stream ──► Browser (listener)
```

**Multiple concurrent sessions are supported.** All session state lives in the `sessions` dict (keyed by session ID), protected by `_lock`. Each session has its own `listener_registry`, `tts_queues`, and expiry timer — sessions are fully isolated.

## Key files

| File | Purpose |
|------|---------|
| `app.py` | Flask backend — session management, WebSocket relay, TTS workers |
| `static/index.html` | Broadcaster UI — mic capture, live caption grid (4 languages) |
| `static/listen.html` | Listener UI — session login, caption display, audio playback |
| `church-transcription.service` | Systemd unit for production (gunicorn + gevent, single worker) |

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in GLADIA_API_KEY and OPENAI_API_KEY
python app.py          # listens on :5001
```

## Session lifecycle

1. Broadcaster hits `POST /init-session` → gets a Gladia WebSocket URL and a 6-char session ID.
2. Broadcaster opens `WebSocket /stream`, sends the Gladia URL as first message, then streams PCM audio.
3. Listeners open `/listen`, enter session ID + language, connect via `WebSocket /listen-stream`.
4. After **2 hours** the session timer fires (`_expire_session`): all listener WebSockets receive an error message and are closed; the session ID is invalidated.

## Audio playback in the listener UI

Browsers block autoplay without a prior user gesture. `listen.html` gates all playback behind `audioUnlocked` (set to `false` on load). A **"Start Audio"** button is shown in the header; clicking it sets `audioUnlocked = true` and drains any queued audio. Audio that arrives before the button is clicked is held in `audioQueue`.

## TTS queue behaviour

Each language has a `queue.Queue(maxsize=1)`. If a new translation arrives while one is already queued, the stale item is dropped and replaced. This keeps listeners in sync with the live speaker rather than falling further behind.

## Deployment notes

- Single gunicorn worker (`-w 1`) required — session state lives in-process.
- gevent monkey-patching at the top of `app.py` is mandatory before any other imports.
- Reverse proxy (nginx) should forward `/church_transcription/` prefix and upgrade WebSocket headers.
