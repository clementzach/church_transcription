#!/usr/bin/env python3
"""End-to-end load test for the church transcription server.

Spins up N broadcaster WebSockets (each streaming a looped speech clip at
real-time pace through the real Gladia + TTS pipeline) and M listener
WebSockets spread across languages, then reports client-side delivery metrics
and — if the server runs on this host — the server process's CPU / RSS / thread
count sampled from /proc.

The goal is to answer one question: can the single uvicorn event loop sustain
the audio fan-out to all listeners without lagging or dropping them?

Usage
-----
    # 5 broadcasters, 200 listeners, 3-minute run, against a local server,
    # streaming a recorded speech clip, watching the server pid for CPU/RSS:
    python tools/loadtest.py \
        --audio sample_speech.wav \
        --broadcasters 5 --listeners 200 --duration 180 \
        --base-url http://127.0.0.1:5001 \
        --server-pid $(pgrep -f 'uvicorn app:app' | head -1)

Notes
-----
* --audio should be a recording of *speech* (any format readable by the `wave`
  module: 16-bit PCM WAV). Non-16 kHz / non-mono input is resampled/downmixed
  automatically. Without real speech, Gladia produces no transcripts, so no
  TTS audio flows and only connection capacity (not fan-out) is exercised — the
  script warns and continues with silence in that case.
* This runs the *real* pipeline: it consumes Gladia and Google TTS quota.
  5 broadcasters = 5 Gladia sessions and up to 5*6 TTS streams. Both are within
  the documented limits (Gladia 30 concurrent; TTS 200 sessions/min).
* The client itself receives the full egress (~40 Mbit/s at full load). If the
  client machine is weak, run listeners from a second box and compare.
"""

import argparse
import asyncio
import json
import os
import random
import string
import time
import wave
from dataclasses import dataclass, field
from statistics import median

import httpx
import numpy as np
import websockets

TRANSLATION_AND_SOURCE = ['en', 'es', 'ht', 'pt', 'zh', 'fr']
SRC_SAMPLE_RATE = 16000  # Gladia expects 16 kHz / 16-bit / mono PCM


# ── Audio loading ──────────────────────────────────────────────────────────

def load_pcm_16k_mono(path: str) -> np.ndarray:
    """Read a WAV file and return int16 samples at 16 kHz mono."""
    with wave.open(path, 'rb') as w:
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    if sample_width != 2:
        raise SystemExit(f"{path}: need 16-bit PCM WAV, got sample width {sample_width}")
    samples = np.frombuffer(frames, dtype=np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    if rate != SRC_SAMPLE_RATE:
        # Simple linear resample — good enough for driving the recognizer.
        duration = len(samples) / rate
        new_len = int(duration * SRC_SAMPLE_RATE)
        x_old = np.linspace(0, 1, num=len(samples), endpoint=False)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.int16)
    return samples


# ── Server resource sampler (Linux /proc, no dependencies) ───────────────────

class ProcSampler:
    """Samples %CPU (per-core; >100 means multiple cores/threads busy), RSS, and
    thread count for a pid from /proc. Returns (cpu_pct, rss_mb, threads)."""

    def __init__(self, pid: int):
        self.pid = pid
        self.clk_tck = os.sysconf('SC_CLK_TCK')
        self._last_cpu = None
        self._last_t = None

    def _read_cpu_jiffies(self) -> int:
        with open(f'/proc/{self.pid}/stat') as f:
            # Fields after the (comm) parenthesised name: utime=14, stime=15.
            parts = f.read().rsplit(')', 1)[1].split()
        utime, stime = int(parts[11]), int(parts[12])
        return utime + stime

    def sample(self):
        now = time.monotonic()
        try:
            cpu = self._read_cpu_jiffies()
            rss_mb = threads = None
            with open(f'/proc/{self.pid}/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        rss_mb = int(line.split()[1]) / 1024.0
                    elif line.startswith('Threads:'):
                        threads = int(line.split()[1])
        except (FileNotFoundError, ProcessLookupError, IndexError):
            return None
        cpu_pct = None
        if self._last_cpu is not None:
            dt = now - self._last_t
            if dt > 0:
                cpu_pct = (cpu - self._last_cpu) / self.clk_tck / dt * 100.0
        self._last_cpu, self._last_t = cpu, now
        return cpu_pct, rss_mb, threads


# ── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class ListenerStat:
    lang: str
    session: str
    connected: bool = False
    error: str | None = None
    audio_bytes: int = 0
    audio_chunks: int = 0
    text_msgs: int = 0
    connect_t: float = 0.0
    first_audio_t: float | None = None
    last_audio_t: float | None = None
    _prev_bytes: int = 0   # snapshot at last report, for interval deltas


@dataclass
class Stats:
    listeners: list = field(default_factory=list)
    broadcaster_chunks: int = 0
    broadcaster_errors: int = 0
    broadcaster_keepalive_deaths: int = 0


# ── Broadcaster ────────────────────────────────────────────────────────────

async def run_broadcaster(http_base, ws_base, session_id, pcm, chunk_ms,
                          stop_evt, stats: Stats):
    # Allocate the session.
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{http_base}/init-session",
                                  json={'session_id': session_id, 'language': 'auto'},
                                  timeout=15)
        if r.status_code != 200:
            print(f"[broadcaster {session_id}] init-session failed: "
                  f"{r.status_code} {r.text[:200]}")
            stats.broadcaster_errors += 1
            return
    except Exception as e:
        print(f"[broadcaster {session_id}] init-session error: {e}")
        stats.broadcaster_errors += 1
        return

    chunk_samples = int(SRC_SAMPLE_RATE * chunk_ms / 1000)
    try:
        # ping_interval=None disables this client's own keepalive pings. When
        # the generator and server share a box, a momentarily busy client loop
        # can otherwise fail to process pongs within the timeout and kill its
        # own connections — a rig artifact, not a server failure.
        async with websockets.connect(f"{ws_base}/stream", max_size=None,
                                      ping_interval=None, open_timeout=15) as ws:
            await ws.send(json.dumps({'session_id': session_id,
                                      'display_langs': TRANSLATION_AND_SOURCE}))
            start = time.monotonic()
            pos = 0
            n = 0
            while not stop_evt.is_set():
                end = pos + chunk_samples
                if end <= len(pcm):
                    chunk = pcm[pos:end]
                    pos = end
                else:  # wrap the clip
                    chunk = np.concatenate([pcm[pos:], pcm[:end - len(pcm)]])
                    pos = end - len(pcm)
                await ws.send(chunk.tobytes())
                n += 1
                stats.broadcaster_chunks += 1
                target = start + n * chunk_ms / 1000.0
                await asyncio.sleep(max(0.0, target - time.monotonic()))
    except Exception as e:
        if not stop_evt.is_set():
            print(f"[broadcaster {session_id}] stream error: {e}")
            stats.broadcaster_errors += 1
            if 'keepalive' in str(e).lower():
                stats.broadcaster_keepalive_deaths += 1


# ── Listener ─────────────────────────────────────────────────────────────────

async def run_listener(ws_base, session_id, lang, stop_evt, st: ListenerStat):
    try:
        async with websockets.connect(f"{ws_base}/listen-stream", max_size=None,
                                      ping_interval=None, open_timeout=15) as ws:
            await ws.send(json.dumps({'session_id': session_id, 'language': lang}))
            st.connected = True
            st.connect_t = time.monotonic()
            while not stop_evt.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                now = time.monotonic()
                if isinstance(msg, bytes):
                    st.audio_bytes += len(msg)
                    st.audio_chunks += 1
                    if st.first_audio_t is None:
                        st.first_audio_t = now
                    st.last_audio_t = now
                else:
                    st.text_msgs += 1
                    try:
                        if json.loads(msg).get('type') == 'error':
                            st.error = 'server error msg'
                            return
                    except Exception:
                        pass
    except Exception as e:
        if not stop_evt.is_set():
            st.error = str(e)[:80]


# ── Orchestration ──────────────────────────────────────────────────────────

def rand_session_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))


async def reporter(stats: Stats, sampler, interval, stop_evt, t0):
    prev_total = 0
    last_t = t0
    while not stop_evt.is_set():
        await asyncio.sleep(interval)
        ls = stats.listeners
        now = time.monotonic()
        dt = now - last_t
        elapsed = now - t0
        connected = sum(1 for s in ls if s.connected and not s.error)
        errored = sum(1 for s in ls if s.error)
        # Instantaneous (this-interval) egress and how many listeners actually
        # received data this interval — a stall shows up here even though the
        # cumulative counts keep looking healthy.
        total_bytes = sum(s.audio_bytes for s in ls)
        receiving_now = sum(1 for s in ls if s.audio_bytes > s._prev_bytes)
        for s in ls:
            s._prev_bytes = s.audio_bytes
        inst_mbit = (total_bytes - prev_total) * 8 / 1e6 / dt if dt else 0
        prev_total = total_bytes
        last_t = now
        line = (f"[{elapsed:6.1f}s] listeners {connected}/{len(ls)} ok "
                f"({errored} err), {receiving_now} receiving now | "
                f"egress {total_bytes / 1e6:7.1f} MB ({inst_mbit:5.1f} Mbit/s now) | "
                f"bcast {stats.broadcaster_chunks}")
        if sampler:
            s = sampler.sample()
            if s:
                cpu, rss, threads = s
                cpu_s = f"{cpu:5.1f}%" if cpu is not None else "  ?  "
                line += f" | server cpu {cpu_s} rss {rss:6.0f}MB thr {threads}"
        print(line, flush=True)


async def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-url', default='http://127.0.0.1:5001',
                    help='HTTP base URL of the server (may include path prefix)')
    ap.add_argument('--ws-url', default=None,
                    help='WebSocket base URL (derived from --base-url if omitted)')
    ap.add_argument('--broadcasters', type=int, default=5)
    ap.add_argument('--listeners', type=int, default=200)
    ap.add_argument('--audio', default=None, help='speech WAV to stream (16-bit PCM)')
    ap.add_argument('--duration', type=float, default=180, help='seconds to run')
    ap.add_argument('--langs', default=','.join(TRANSLATION_AND_SOURCE),
                    help='comma-separated listener languages to spread across')
    ap.add_argument('--chunk-ms', type=int, default=100, help='audio chunk size')
    ap.add_argument('--ramp', type=float, default=10,
                    help='seconds over which to stagger listener connects')
    ap.add_argument('--report-interval', type=float, default=5)
    ap.add_argument('--server-pid', type=int, default=None,
                    help='pid of the uvicorn process for /proc CPU+RSS sampling')
    args = ap.parse_args()

    http_base = args.base_url.rstrip('/')
    ws_base = args.ws_url or ('ws' + http_base[len('http'):])
    ws_base = ws_base.rstrip('/')
    langs = [l.strip() for l in args.langs.split(',') if l.strip()]

    if args.audio:
        pcm = load_pcm_16k_mono(args.audio)
        print(f"Loaded {len(pcm) / SRC_SAMPLE_RATE:.1f}s of audio from {args.audio}")
    else:
        print("WARNING: no --audio given. Streaming silence; Gladia will produce "
              "no transcripts, so NO TTS audio will flow. This tests connection "
              "capacity only, not fan-out.")
        pcm = np.zeros(SRC_SAMPLE_RATE * 5, dtype=np.int16)

    sampler = ProcSampler(args.server_pid) if args.server_pid else None
    if args.server_pid:
        sampler.sample()  # prime the CPU delta

    stats = Stats()
    stop_evt = asyncio.Event()
    t0 = time.monotonic()

    # Broadcasters first, so sessions exist before listeners join.
    session_ids = [rand_session_id() for _ in range(args.broadcasters)]
    bcast_tasks = [
        asyncio.create_task(
            run_broadcaster(http_base, ws_base, sid, pcm, args.chunk_ms, stop_evt, stats))
        for sid in session_ids
    ]
    await asyncio.sleep(2.0)  # let init-session + Gladia connect settle

    # Listeners spread round-robin across sessions and languages, ramped in.
    listener_tasks = []
    for i in range(args.listeners):
        sid = session_ids[i % len(session_ids)]
        lang = langs[i % len(langs)]
        st = ListenerStat(lang=lang, session=sid)
        stats.listeners.append(st)
        listener_tasks.append(
            asyncio.create_task(run_listener(ws_base, sid, lang, stop_evt, st)))
        if args.ramp:
            await asyncio.sleep(args.ramp / args.listeners)

    rep = asyncio.create_task(reporter(stats, sampler, args.report_interval, stop_evt, t0))

    await asyncio.sleep(max(0.0, args.duration - (time.monotonic() - t0)))
    stop_evt.set()
    rep.cancel()
    await asyncio.gather(*bcast_tasks, *listener_tasks, rep, return_exceptions=True)

    # ── Final summary ────────────────────────────────────────────────────
    ls = stats.listeners
    elapsed = time.monotonic() - t0
    ok = [s for s in ls if s.connected and not s.error]
    errored = [s for s in ls if s.error]
    receiving = [s for s in ls if s.audio_chunks > 0]
    ttfa = sorted(s.first_audio_t - s.connect_t for s in receiving if s.first_audio_t)
    total_mb = sum(s.audio_bytes for s in ls) / 1e6

    def pct(p, xs):
        return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float('nan')

    print("\n" + "=" * 70)
    print("LOAD TEST SUMMARY")
    print("=" * 70)
    print(f"  duration:            {elapsed:.1f}s")
    print(f"  broadcasters:        {args.broadcasters} "
          f"({stats.broadcaster_chunks} chunks sent, {stats.broadcaster_errors} errors)")
    if stats.broadcaster_keepalive_deaths:
        print(f"    NOTE: {stats.broadcaster_keepalive_deaths} broadcaster(s) died of "
              f"keepalive timeout — usually the generator loop starving when\n"
              f"          co-located with the server, not a server failure. "
              f"Re-run from a separate host to confirm.")
    print(f"  listeners connected: {len(ok)}/{args.listeners}  ({len(errored)} errored)")
    print(f"  listeners w/ audio:  {len(receiving)}/{args.listeners}")
    print(f"  total egress:        {total_mb:.1f} MB  "
          f"({total_mb * 8 / elapsed:.1f} Mbit/s sustained)")
    if ttfa:
        print(f"  time-to-first-audio: median {median(ttfa):.1f}s  "
              f"p95 {pct(0.95, ttfa):.1f}s  max {ttfa[-1]:.1f}s")
    # Per-language fan-out balance — large divergence flags a struggling loop.
    per_lang = {}
    for s in receiving:
        per_lang.setdefault(s.lang, []).append(s.audio_bytes)
    for lang in sorted(per_lang):
        vals = per_lang[lang]
        print(f"    {lang}: {len(vals):3d} receiving, "
              f"avg {sum(vals) / len(vals) / 1e6:.2f} MB/listener")
    if errored:
        from collections import Counter
        common = Counter(s.error for s in errored).most_common(5)
        print("  top errors:")
        for msg, c in common:
            print(f"    {c:4d} x {msg}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
