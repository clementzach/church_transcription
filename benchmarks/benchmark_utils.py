"""
benchmark_utils.py — helpers for streaming transcription benchmarks.

Typical usage (Jupyter notebook):
    from benchmarks.benchmark_utils import run_benchmark, run_all_benchmarks, print_scores, dry_run
    from benchmarks.registry import STRATEGIES

    # Run all strategies in parallel
    results = run_all_benchmarks(STRATEGIES)

    # Or run a single strategy
    from functools import partial
    from benchmarks.benchmark_utils import transcribe_gladia
    df = run_benchmark(partial(transcribe_gladia, vocabulary=[], language_hint=True), label="gladia_lang_auto")

Transcripts are cached in benchmarks/results/<label>/<lang>/<title>.txt.
Re-running reuses cached transcripts and recomputes all metrics.
"""

import asyncio
import concurrent.futures
import csv
import json
import os
import pathlib
import re
import ssl
import tempfile
import threading
import time
from collections import defaultdict
from typing import Callable, Optional
import time

import anthropic
import certifi
import httpx
import jiwer
import openai
import pandas as pd
import websockets
from dotenv import load_dotenv
from pydub import AudioSegment

# SSL context using certifi's CA bundle.
# Python installed from python.org on macOS does not trust the system keychain,
# so wss:// connections fail without this.
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

load_dotenv()

GLADIA_API_KEY    = os.getenv("GLADIA_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

BENCHMARK_ROOT = pathlib.Path(__file__).parent.parent / "conference_data" / "benchmark"
METADATA_CSV   = BENCHMARK_ROOT / "benchmark_metadata.csv"
RESULTS_ROOT   = pathlib.Path(__file__).parent / "results"

# Directory name → BCP-47 language code used by transcription APIs
LANG_CODES: dict[str, str] = {
    "eng": "en",
    "spa": "es",
    "nor": "no",
    "por": "pt",
    "hat": "ht",
    "zho": "zh",
}

CHUNK_MS = 250          # audio chunk size for real-time simulation
_PRINT_LOCK = threading.Lock()   # serialise verbose output across parallel threads

# ── Church vocabulary ──────────────────────────────────────────────────────────
# Applied by default to both Gladia (custom_vocabulary) and AssemblyAI
# (keyterms_prompt) to boost recognition of proper names and LDS-specific terms.
# All entries are ≤50 characters to satisfy AssemblyAI's keyterms limit.
# Override with vocabulary=[] / keyterms=[] to run a no-vocabulary baseline.

CHURCH_VOCABULARY: list[str] = [
    # ── First Presidency & Quorum of the Twelve (active in 2021–2026) ─────────
    "Nelson",
    "Oaks",
    "Eyring",
    "Holland",
    "Uchtdorf",
    "Bednar",
    "Cook",
    "Christofferson",
    "Andersen",
    "Rasband",
    "Stevenson",
    "Renlund",
    "Gong",
    "Soares",
    "Kearon",
    "Ballard",
    "Gilbert",

    # ── Book of Mormon figures ─────────────────────────────────────────────────
    "Nephi",
    "Lehi",
    "Laman",
    "Lemuel",
    "Sariah",
    "Jacob",
    "Enos",
    "Mosiah",
    "Abinadi",
    "Alma",
    "Amulek",
    "Ammon",
    "Helaman",
    "Mormon",
    "Moroni",
    "Ether",
    "Jaredite",
    "Gadianton",
    "Zeezrom",
    "Korihor",
    "Teancum",
    "Pahoran",
    "Limhi",
    "Zeniff",
    "Coriantumr",

    # ── LDS-specific terms ─────────────────────────────────────────────────────
    "Atonement",
    "Testimony",
    "Restoration",
    "Priesthood",
    "Sacrament",
    "Endowment",
    "Covenant",
    "Covenants",
    "Celestial",
    "Melchizedek",
    "Aaronic",
    "Tithing",
    "Consecration",
    "Exaltation",
    "Liahona",
    "Tabernacle",
    "Dispensation",
    "Apostasy",
    "Stewardship",
    "Ministering",
    "Sealing",
    "Seer",
    "Revelator",
    "Nephite",
    "Lamanite",
    "Zion",
    "General Conference",
    "Latter-day Saints",
    "First Presidency",
    "Urim and Thummim",
    "Plan of Salvation",
    "Word of Wisdom",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_benchmark_pairs(langs: Optional[list[str]] = None) -> list[dict]:
    """Load (mp3_path, text_path) pairs by scanning the benchmark filesystem.

    The metadata CSV only covers English rows, so we scan the directory tree
    directly and match each mp3 to its corresponding text file by relative path.
    Silently skips any mp3 that has no matching text file.

    Directory structure expected:
        benchmark/{lang}/mp3/y_{year}/m_{month}/{title}.mp3
        benchmark/{lang}/text/y_{year}/m_{month}/{title}.txt

    Returns:
        List of dicts with keys: lang, title, mp3_path, text_path.
        Sorted by (lang, year, month, title) for deterministic ordering.
    """
    target_langs = set(langs) if langs else set(LANG_CODES.keys())
    pairs = []
    for lang_dir in sorted(BENCHMARK_ROOT.iterdir()):
        lang = lang_dir.name
        if lang not in target_langs:
            continue
        mp3_root = lang_dir / "mp3"
        txt_root = lang_dir / "text"
        if not mp3_root.exists():
            continue
        for mp3_path in sorted(mp3_root.rglob("*.mp3")):
            # Derive the expected text path from the same relative sub-path
            rel = mp3_path.relative_to(mp3_root)
            txt_path = txt_root / rel.with_suffix(".txt")
            if not txt_path.exists():
                continue
            title = mp3_path.stem.replace("_", " ").strip('"')
            pairs.append({
                "lang":      lang,
                "title":     title,
                "mp3_path":  mp3_path,
                "text_path": txt_path,
            })
    return pairs


def load_ground_truth(text_path: pathlib.Path) -> str:
    """Read a talk text file, stripping the title / author / calling header.

    Header format (consistent across all languages):
        <title>
        <blank>
        <author>
        <blank>
        <calling>
        <blank>
        <talk body ...>

    We skip the first 3 groups of non-blank lines (title, author, calling)
    and all the interleaved blank lines, returning only the talk body.
    """
    lines = text_path.read_text(encoding="utf-8").splitlines()
    i = 0
    for _ in range(3):
        while i < len(lines) and lines[i].strip():   # skip non-blank (header group)
            i += 1
        while i < len(lines) and not lines[i].strip(): # skip blank separator
            i += 1
    return "\n".join(lines[i:]).strip()


# ── Text normalisation ─────────────────────────────────────────────────────────

def normalize_text(text: str, lang: Optional[str] = None) -> str:
    """Lowercase, strip punctuation, and collapse whitespace.

    For Chinese (zho) we insert a space between every character so that
    jiwer.wer() tokenises at the character level rather than treating the
    whole string as one word.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    if lang == "zho":
        # Remove spaces first, then re-insert between every character
        text = " ".join(text.replace(" ", ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Audio streaming ────────────────────────────────────────────────────────────

def stream_audio_chunks(
    audio_path: pathlib.Path,
    chunk_ms: int = CHUNK_MS,
    max_duration_secs: Optional[float] = None,
):
    """Generator yielding raw PCM (16-bit, 16 kHz, mono) byte chunks.

    Uses pydub so it handles mp3, wav, and any other ffmpeg-supported format.

    Args:
        audio_path:        Path to audio file.
        chunk_ms:          Chunk size in milliseconds.
        max_duration_secs: If set, stop after this many seconds of audio.
    """
    # Pass format explicitly so pydub skips the ffprobe detection step.
    fmt = pathlib.Path(audio_path).suffix.lstrip(".").lower()
    audio = AudioSegment.from_file(str(audio_path), format=fmt)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    if max_duration_secs is not None:
        audio = audio[: int(max_duration_secs * 1000)]
    for start in range(0, len(audio), chunk_ms):
        yield audio[start : start + chunk_ms].raw_data


def _paced_stream(audio_path: pathlib.Path, chunk_ms: int = CHUNK_MS):
    """Yield PCM chunks with real-time pacing delays.

    Suitable for passing directly to assemblyai's client.stream() or any
    synchronous consumer that needs chunks at the natural audio rate.
    """
    for chunk in stream_audio_chunks(audio_path, chunk_ms=chunk_ms):
        yield chunk
        time.sleep(chunk_ms / 1000)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_word_metrics(
    hypothesis: str, reference: str, lang: Optional[str] = None
) -> dict:
    """Return WER, recall, and accuracy for a hypothesis/reference pair.

    Both strings are normalised before comparison. For Chinese, metrics are
    computed at the character level (see normalize_text).

    Returns:
        dict with keys:
            wer      — Word Error Rate (lower is better)
            recall   — hits / (hits + substitutions + deletions); proportion of
                       reference words correctly captured
            accuracy — hits / (hits + substitutions + insertions); precision of
                       the hypothesis words
    """
    norm_ref = normalize_text(reference, lang)
    norm_hyp = normalize_text(hypothesis, lang)
    output = jiwer.process_words(norm_ref, norm_hyp)
    hits = output.hits
    subs = output.substitutions
    dels = output.deletions
    ins  = output.insertions
    n    = hits + subs + dels  # total reference words
    wer      = (subs + dels + ins) / n if n > 0 else 0.0
    recall   = hits / n                if n > 0 else 0.0
    accuracy = hits / (hits + subs + ins) if (hits + subs + ins) > 0 else 0.0
    return {"wer": wer, "recall": recall, "accuracy": accuracy}


def compute_llm_score(
    hypothesis: str, reference: str, lang: Optional[str] = None
) -> float:
    """Use Claude Sonnet as a judge to score transcription quality 1–5.

    8 = perfect transcription, 1 = completely unusable.
    Returns float("nan") on any error.
    """
    prompt = (
        "You are evaluating the quality of a speech transcription. Note that the ground truth text may contain additional footnotes or reference not spoken in the transcription. You may ignore those while making your evaluation.\n\n"
        
        "Rate the transcription quality on a scale from 1 to 8:\n"
        "8 = Perfect transcription. There are no errors and the transcription matches the ground truth perfectly besides any footnotes existing in the ground truth.\n"
        "7 = Nearly Perfect transcription. There may be minor inconsistencies in spelling or formatting but a human would be able to understand the full meaning of every part of the transcription perfectly.\n"
        "6 = Very Good transcription. Some words in the transcription may differ, but a human would be able to understand the general content of each part of the transcription .\n"
         "5 = Good transcription. Many words in the transcription may differ, but a human would be able to understand the general content of most parts of the transcription .\n"
        "4 = Fair. A human may be able to understand the content of most sections, but many sections are garbled and difficult to understand.\n"
        "3 = Poor. A human may be able to guess at the general subject of the transcription based on some of the words, and they may be able to understand the content of some sections, but most sections are garbled and impossible to understand.\n"
        "2 = Very Poor. A human may be able to guess at the general subject of the transcription based on some of the words, but would not be able to understand any of the points the speaker is trying to convey.\n"
        "1 = Completely unusable. No or very few words match between the ground truth and transcription. Words do not follow grammatical structure.\n\n"
        f"Ground truth: ```\n{reference}\n```\n"
        f"Transcription: ```\n{hypothesis}\n```\n"
        "Respond with only a single integer (1-8). Do not include anything else in your response. \n"
        "Correct example responses are:\n```5```\n```1```\n```3``` . \n"
    )
    try:
        time.sleep(60)
        response = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return float(response.content[0].text.strip())
    except Exception as e:
        error_str = "Error with llm! " + str(e)
        try:
            error_str = error_str + " " + str(response.content[0].text)
        except:
            pass
        print(error_str)
        return float("nan")


# ── Gladia ─────────────────────────────────────────────────────────────────────

def _run_async(coro):
    """Run an async coroutine safely from both plain Python and Jupyter.

    Jupyter already has a running event loop, so asyncio.run() would raise
    'This event loop is already running'. Running in a fresh thread avoids
    that without requiring nest_asyncio.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


async def _gladia_transcribe_async(
    audio_path: pathlib.Path,
    *,
    vocabulary: Optional[list] = None,
    custom_spelling: Optional[dict] = None,
    audio_enhancer: bool = False,
    language: Optional[str] = None,
) -> str:
    if not GLADIA_API_KEY:
        raise RuntimeError("GLADIA_API_KEY is not set in .env")
    config: dict = {
        "sample_rate": 16000,
        "bit_depth":   16,
        "channels":    1,
        "encoding":    "wav/pcm",
        "model":       "solaria-1",
        # Nested sub-objects required by Gladia v2 schema
        "pre_processing": {
            "audio_enhancer": audio_enhancer,
        },
    }
    if language:
        config["language_config"] = {"languages": [language]}
    realtime: dict = {}
    if vocabulary:
        realtime["custom_vocabulary"] = True
        realtime["custom_vocabulary_config"] = {
            "vocabulary":        vocabulary,
            "default_intensity": 0.1,
        }
    if custom_spelling:
        realtime["custom_spelling"] = True
        realtime["custom_spelling_config"] = {"spelling_dictionary": custom_spelling}
    if realtime:
        config["realtime_processing"] = realtime

    # Pre-load audio before opening the WebSocket so the blocking ffmpeg decode
    # does not stall the event loop after the connection is established.
    audio_chunks = list(stream_audio_chunks(audio_path))

    async with httpx.AsyncClient(verify=certifi.where()) as client:
        resp = await client.post(
            "https://api.gladia.io/v2/live",
            headers={
                "X-Gladia-Key":  GLADIA_API_KEY,
                "Content-Type":  "application/json",
            },
            json=config,
            timeout=10,
        )
        if not resp.is_success:
            raise RuntimeError(
                f"Gladia init failed {resp.status_code}: {resp.text}"
            )
        ws_url = resp.json()["url"]

    transcript_parts: list[str] = []

    async with websockets.connect(ws_url, ssl=_SSL_CTX) as ws:
        async def _send():
            for chunk in audio_chunks:
                await ws.send(chunk)
                await asyncio.sleep(CHUNK_MS / 1000)
            await ws.send(json.dumps({"type": "stop_recording"}))

        send_task = asyncio.create_task(_send())
        try:
            async for raw in ws:
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "transcript":
                    data = msg.get("data") or {}
                    if data.get("is_final"):
                        text = (data.get("utterance") or {}).get("text", "").strip()
                        if text:
                            transcript_parts.append(text)
                elif msg_type == "post_final_transcript":
                    break
        except websockets.exceptions.ConnectionClosedOK:
            pass
        finally:
            send_task.cancel()

    return " ".join(transcript_parts)


def transcribe_gladia(
    audio_path: pathlib.Path,
    lang: str,
    *,
    vocabulary: Optional[list] = None,
    custom_spelling: Optional[dict] = None,
    audio_enhancer: bool = False,
    language_hint: bool = False,
) -> str:
    """Transcribe audio using Gladia streaming (solaria-1).

    Args:
        audio_path:      Path to audio file.
        lang:            Benchmark language directory name (e.g. 'eng', 'spa').
        vocabulary:      Custom vocabulary list sent to Gladia. Pass CHURCH_VOCABULARY
                         to boost LDS terms, or [] for a no-vocabulary baseline.
        custom_spelling: Optional dict mapping canonical forms to pronunciations.
        audio_enhancer:  Apply Gladia's audio quality preprocessing.
        language_hint:   If True, pass the BCP-47 code via language_config.languages
                         so Gladia skips auto-detection.

    Returns:
        Full transcript as a single string.
    """
    return _run_async(
        _gladia_transcribe_async(
            audio_path,
            vocabulary=vocabulary,
            custom_spelling=custom_spelling,
            audio_enhancer=audio_enhancer,
            language=LANG_CODES[lang] if language_hint else None,
        )
    )


# ── AssemblyAI Universal Streaming ─────────────────────────────────────────────
# Uses the raw WebSocket API directly (no SDK) to avoid SSL / dependency issues.
#
# Valid speech_model values:
#   "universal-streaming-multilingual"  — Universal-2, all languages incl. Haitian Creole
#   "universal-streaming-english"       — Universal-2, English only
#   "u3-rt-pro"                         — Universal-3 Pro (highest accuracy, fewer languages)
#   "whisper-rt"                        — Whisper real-time
#
# u3-rt-pro only supports a subset of languages; passing an unsupported language_code
# causes a 1011 internal error.  language_code is silently dropped when the lang is
# not in _AAI_U3_PRO_LANGS.
_AAI_U3_PRO_LANGS = {"en", "es", "fr", "de", "pt", "it", "nl", "no", "sv", "da",
                     "fi", "pl", "ru", "uk", "ko", "ja", "zh", "hi", "tr"}

import urllib.parse as _urlparse


async def _assemblyai_transcribe_async(
    audio_path: pathlib.Path,
    *,
    keyterms: Optional[list[str]] = None,
    speech_model: str = "u3-rt-pro",
    language: Optional[str] = None,
) -> str:
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set in .env")

    # Pre-load audio before opening the WebSocket so the blocking ffmpeg decode
    # does not stall the event loop after the connection is established.
    audio_chunks = list(stream_audio_chunks(audio_path))

    params: list[tuple[str, str]] = [
        ("speech_model", speech_model),
        ("encoding",     "pcm_s16le"),
        ("sample_rate",  "16000"),
        ("token",        ASSEMBLYAI_API_KEY),
    ]
    if language:
        params.append(("language_code", language))
    for term in (keyterms or []):
        params.append(("keyterms_prompt", term))

    url = "wss://streaming.assemblyai.com/v3/ws?" + _urlparse.urlencode(params)
    transcript_parts: list[str] = []

    async with websockets.connect(url, ssl=_SSL_CTX) as ws:
        async def _send():
            for chunk in audio_chunks:
                await ws.send(chunk)
                await asyncio.sleep(CHUNK_MS / 1000)
            await ws.send(json.dumps({"type": "Terminate"}))

        send_task = asyncio.create_task(_send())
        try:
            async for raw in ws:
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "Turn" and msg.get("end_of_turn"):
                    text = msg.get("transcript", "").strip()
                    if text:
                        transcript_parts.append(text)
                elif msg_type == "Termination":
                    break
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            send_task.cancel()

    return " ".join(transcript_parts)


def transcribe_assemblyai(
    audio_path: pathlib.Path,
    lang: str,
    *,
    keyterms: Optional[list[str]] = None,
    speech_model: str = "u3-rt-pro",
    language_hint: bool = False,
) -> str:
    """Transcribe audio using AssemblyAI Universal Streaming (raw WebSocket).

    Args:
        audio_path:    Path to audio file.
        lang:          Benchmark language directory name (e.g. 'eng', 'spa').
        keyterms:      Up to 100 terms (≤50 chars each) to boost recognition of.
                       Defaults to None. Pass [] for a baseline run.
        speech_model:  Model identifier:
                         "universal-streaming-multilingual" — Universal-2 (all languages,
                             including Haitian Creole)
                         "u3-rt-pro" — Universal-3 Pro (highest accuracy, fewer languages)
        language_hint: If True, pass the BCP-47 language code so the model skips
                       auto-detection.

    Returns:
        Full transcript as a single string.
    """
    lang_code = LANG_CODES[lang] if language_hint else None
    # u3-rt-pro only handles a subset of languages; drop the hint for others to
    # avoid a 1011 internal error from the AssemblyAI server.
    if lang_code and speech_model == "u3-rt-pro" and lang_code not in _AAI_U3_PRO_LANGS:
        lang_code = None
    return _run_async(
        _assemblyai_transcribe_async(
            audio_path,
            keyterms=keyterms,
            speech_model=speech_model,
            language=lang_code,
        )
    )


# ── Whisper (OpenAI API) ───────────────────────────────────────────────────────

def transcribe_whisper(
    audio_path: pathlib.Path,
    lang: str,
    *,
    local: bool = False,
) -> str:
    """Transcribe audio using OpenAI Whisper.

    Args:
        audio_path: Path to audio file (any format OpenAI accepts).
        lang:       Benchmark language directory name.
        local:      If True, run a locally-hosted Whisper model instead of
                    the OpenAI API. Currently not implemented — see stub below.

    Returns:
        Full transcript as a single string.
    """
    if local:
        # To implement: pip install openai-whisper, then:
        #   import whisper
        #   model = whisper.load_model("large-v3")
        #   result = model.transcribe(str(audio_path), language=LANG_CODES[lang])
        #   return result["text"]
        raise NotImplementedError(
            "Local Whisper is not yet wired up. Set local=False to use the OpenAI API."
        )

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    lang_code = LANG_CODES[lang]
    with open(audio_path, "rb") as f:
        try:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=lang_code,
                response_format="text",
            )
        except openai.BadRequestError as exc:
            if "unsupported_language" in str(exc):
                # Fall back to auto-detection for languages Whisper doesn't support
                f.seek(0)
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                )
            else:
                raise
    return result


# ── Benchmark runner ───────────────────────────────────────────────────────────

def _safe_label(label: str) -> str:
    """Return a filesystem-safe version of *label*."""
    if label:
        return re.sub(r"[^\w\-]", "_", label).strip("_")
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _transcript_path(safe_label: str, lang: str, title: str) -> pathlib.Path:
    safe_title = re.sub(r"[^\w\-]", "_", title)
    return RESULTS_ROOT / safe_label / lang / f"{safe_title}.txt"


def _load_cached_transcript(safe_label: str, lang: str, title: str) -> Optional[str]:
    path = _transcript_path(safe_label, lang, title)
    return path.read_text(encoding="utf-8") if path.exists() else None


def _save_transcript(safe_label: str, lang: str, title: str, hypothesis: str) -> None:
    path = _transcript_path(safe_label, lang, title)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(hypothesis, encoding="utf-8")


def run_benchmark(
    transcribe_fn: Callable,
    langs: Optional[list[str]] = None,
    label: str = "",
    verbose: bool = True,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Run transcription benchmark across all available (lang, talk) pairs.

    Transcripts are cached in ``benchmarks/results/<label>/``. If a cached
    transcript exists for a talk it is loaded from disk rather than calling
    the API again. All metrics (WER, recall, accuracy, LLM score) are always
    recomputed from the transcript text.

    Args:
        transcribe_fn: Callable with signature (audio_path, lang) -> str.
                       Use functools.partial to bind API-specific kwargs.
        langs:         Restrict to these language directory names. None = all.
        label:         Label used as the results subdirectory name and for
                       progress output. Should not contain spaces.
        verbose:       Print per-talk progress (output may interleave when
                       running in parallel).
        max_workers:   Max threads for parallel talk transcription.

    Returns:
        pd.DataFrame with columns:
            lang, title, wer, recall, accuracy, llm_score, hypothesis, reference
    """
    pairs = load_benchmark_pairs(langs=langs)
    slabel = _safe_label(label)

    if verbose and label:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")

    def _run_one(pair: dict) -> dict:
        lang  = pair["lang"]
        title = pair["title"]
        reference = load_ground_truth(pair["text_path"])

        cached = _load_cached_transcript(slabel, lang, title)
        from_cache = cached is not None

        try:
            if from_cache:
                hypothesis = cached
            else:
                hypothesis = transcribe_fn(pair["mp3_path"], lang)
                _save_transcript(slabel, lang, title, hypothesis)

            metrics   = compute_word_metrics(hypothesis, reference, lang=lang)
            llm_score = compute_llm_score(hypothesis, reference, lang=lang)

            if verbose:
                tag = "[cached] " if from_cache else ""
                with _PRINT_LOCK:
                    print(
                        f"  [{lang}] {title[:45]:<45} {tag}"
                        f"WER={metrics['wer']:.3f} "
                        f"recall={metrics['recall']:.3f} "
                        f"acc={metrics['accuracy']:.3f} "
                        f"llm={llm_score:.1f}"
                    )

            return {
                "lang":       lang,
                "title":      title,
                "wer":        metrics["wer"],
                "recall":     metrics["recall"],
                "accuracy":   metrics["accuracy"],
                "llm_score":  llm_score,
                "hypothesis": hypothesis,
                "reference":  reference,
            }

        except Exception as exc:
            if verbose:
                with _PRINT_LOCK:
                    print(f"  [{label}][{lang}] {title[:45]:<45} ERROR — {exc}")
            return {
                "lang":       lang,
                "title":      title,
                "wer":        float("nan"),
                "recall":     float("nan"),
                "accuracy":   float("nan"),
                "llm_score":  float("nan"),
                "hypothesis": None,
                "reference":  None,
                "error":      str(exc),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(_run_one, pairs))

    return pd.DataFrame(rows)


def run_all_benchmarks(
    strategies: list[dict],
    langs: Optional[list[str]] = None,
    max_workers: int = 5,
    verbose: bool = True,
) -> dict:
    """Run multiple benchmark strategies in parallel.

    Each strategy is a dict with keys:
        label — filesystem-safe name (no spaces), used as results subdirectory
        fn    — transcribe callable with signature (audio_path, lang) -> str

    Args:
        strategies:  List of strategy dicts (see benchmarks/registry.py).
        langs:       Restrict to these language codes. None = all.
        max_workers: Max concurrent strategies.
        verbose:     Print per-talk progress.

    Returns:
        Dict mapping label -> pd.DataFrame.
    """
    results: dict = {}

    def _run_strategy(strategy: dict):
        return strategy["label"], run_benchmark(
            strategy["fn"],
            langs=langs,
            label=strategy["label"],
            verbose=verbose,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for label, df in executor.map(_run_strategy, strategies):
            results[label] = df

    return results


# ── Score printing ─────────────────────────────────────────────────────────────

def print_scores(df: pd.DataFrame, label: str = "") -> None:
    """Print per-language metrics and overall weighted averages.

    Error rows (NaN wer) are excluded from averages.
    """
    if label:
        print(f"\n{'=' * 66}")
        print(f"  {label}")
        print(f"{'=' * 66}")

    valid = df.dropna(subset=["wer"])

    print(f"\n{'Language':<10} {'Talks':>6} {'WER':>8} {'Recall':>8} {'Accuracy':>10} {'LLM':>6}")
    print("─" * 52)
    for lang, group in valid.groupby("lang"):
        print(
            f"{lang:<10} {len(group):>6}"
            f" {group['wer'].mean():>8.3f}"
            f" {group['recall'].mean():>8.3f}"
            f" {group['accuracy'].mean():>10.3f}"
            f" {group['llm_score'].mean():>6.2f}"
        )
    print("─" * 52)
    if not valid.empty:
        print(
            f"{'Overall':<10} {len(valid):>6}"
            f" {valid['wer'].mean():>8.3f}"
            f" {valid['recall'].mean():>8.3f}"
            f" {valid['accuracy'].mean():>10.3f}"
            f" {valid['llm_score'].mean():>6.2f}"
        )


# ── Dry run ────────────────────────────────────────────────────────────────────

def dry_run(
    transcribe_fn: Callable,
    audio_path: pathlib.Path,
    duration_secs: float = 3.0,
) -> str:
    """Stream only the first `duration_secs` seconds of audio through transcribe_fn.

    Use this to verify that a transcription pipeline is wired up correctly
    before committing to a full benchmark run.

    The language is inferred from the file path (expects the benchmark directory
    structure: .../benchmark/<lang>/mp3/...). Falls back to 'eng' if not found.

    Prints the partial transcript and returns it.
    """
    # Infer language from path
    parts = pathlib.Path(audio_path).parts
    lang = next((p for p in parts if p in LANG_CODES), "eng")

    # Clip to duration_secs in memory and write to a temp WAV file.
    # Pass format explicitly so pydub skips the ffprobe detection step.
    fmt = pathlib.Path(audio_path).suffix.lstrip(".").lower()
    audio = AudioSegment.from_file(str(audio_path), format=fmt)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    clipped = audio[: int(duration_secs * 1000)]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        clipped.export(tmp.name, format="wav")
        tmp_path = pathlib.Path(tmp.name)

    print(
        f"Dry run: {duration_secs}s of '{pathlib.Path(audio_path).name}'"
        f" (lang={lang}, {len(clipped) / 1000:.1f}s)"
    )
    try:
        result = transcribe_fn(tmp_path, lang)
        print(f"Transcript: {result!r}")
        return result
    finally:
        tmp_path.unlink(missing_ok=True)
