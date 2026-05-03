import io
import json
import logging
import os
import re
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")  # root .env (GOOGLE_API_KEY etc.)
load_dotenv(Path(__file__).parent / ".env")  # fine_tuning/.env overrides

import numpy as np
from faster_whisper import WhisperModel
from google import genai as google_genai
from google.genai import types as google_types
from pydub import AudioSegment
from pydub.silence import split_on_silence

from fine_tuning.data_preprocessing import get_ipa_string, identify_phonetic_component_within_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).parent.parent
MP3_ROOT = REPO_ROOT / "conference_data" / "haitian_creole" / "mp3"
TXT_ROOT = REPO_ROOT / "conference_data" / "haitian_creole" / "text"
OUTPUT_ROOT = REPO_ROOT / "fine_tuning" / "output"
DATA_DIR = OUTPUT_ROOT / "data"
METADATA_FILE = OUTPUT_ROOT / "metadata.jsonl"

SILENCE_MIN_LEN_MS = 800
SILENCE_THRESH_DBFS = -40
MIN_CHUNK_MS = 2_000
MAX_SEGMENT_MS = 30_000
WINDOW_SIZE = 4
LANG_NAME = "Haitian Creole"
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_LANG = "ht"
API_MAX_RETRIES = 3
API_RETRY_DELAY = 2.0
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_TTS_VOICE = "Kore"  # same voice used for 'ht' in app.py

_whisper_model: WhisperModel | None = None
_google_client: google_genai.Client | None = None


def _get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    return _whisper_model


def _get_google_client() -> google_genai.Client:
    global _google_client
    if _google_client is None:
        _google_client = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _google_client


def synthesize_tts_audio(text: str) -> AudioSegment | None:
    for attempt in range(API_MAX_RETRIES):
        try:
            client = _get_google_client()
            response = client.models.generate_content(
                model=GEMINI_TTS_MODEL,
                contents=text,
                config=google_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=google_types.SpeechConfig(
                        voice_config=google_types.VoiceConfig(
                            prebuilt_voice_config=google_types.PrebuiltVoiceConfig(
                                voice_name=GEMINI_TTS_VOICE,
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
            wav_bytes = _pcm_to_wav(pcm_data)
            return AudioSegment.from_wav(io.BytesIO(wav_bytes))
        except Exception as e:
            if attempt < API_MAX_RETRIES - 1:
                sleep_s = (2 ** attempt) * API_RETRY_DELAY
                logger.warning("TTS error %s, retrying in %.1fs", e, sleep_s)
                time.sleep(sleep_s)
            else:
                logger.error("TTS synthesis failed after %d retries: %s", API_MAX_RETRIES, e)
    return None


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


@dataclass
class TalkPair:
    mp3_path: Path
    txt_path: Path
    year: str
    month: str
    stem: str


@dataclass
class Utterance:
    audio: AudioSegment
    whisper_text: str
    matched_text: str
    is_matched: bool
    duration_ms: int


def find_talk_pairs(mp3_root: Path = MP3_ROOT, txt_root: Path = TXT_ROOT) -> list[TalkPair]:
    mp3_index: dict[tuple, Path] = {}
    for mp3 in mp3_root.rglob("*.mp3"):
        parts = mp3.relative_to(mp3_root).parts
        if len(parts) == 3:
            year, month, filename = parts
            mp3_index[(year, month, mp3.stem)] = mp3

    pairs = []
    for txt in txt_root.rglob("*.txt"):
        parts = txt.relative_to(txt_root).parts
        if len(parts) == 3:
            year, month, _ = parts
            key = (year, month, txt.stem)
            if key in mp3_index:
                pairs.append(TalkPair(
                    mp3_path=mp3_index[key],
                    txt_path=txt,
                    year=year,
                    month=month,
                    stem=txt.stem,
                ))
            else:
                logger.debug("No matching mp3 for %s", txt)

    pairs.sort(key=lambda p: (p.year, p.month, p.stem))
    logger.info("Found %d matched talk pairs", len(pairs))
    return pairs


def chunk_audio_on_silence(mp3_path: Path) -> list[AudioSegment]:
    audio = AudioSegment.from_file(mp3_path)
    chunks = split_on_silence(
        audio,
        min_silence_len=SILENCE_MIN_LEN_MS,
        silence_thresh=SILENCE_THRESH_DBFS,
        keep_silence=100,
    )
    chunks = [c for c in chunks if len(c) >= MIN_CHUNK_MS]

    if len(chunks) <= 1:
        # Fallback: fixed-duration slices
        half = MAX_SEGMENT_MS // 2
        chunks = [audio[i:i + half] for i in range(0, len(audio), half)]
        chunks = [c for c in chunks if len(c) >= MIN_CHUNK_MS]

    return chunks


def transcribe_chunk(chunk: AudioSegment) -> str:
    mono_16k = chunk.set_frame_rate(16_000).set_channels(1)
    audio_array = np.array(mono_16k.get_array_of_samples()).astype(np.float32) / 32768.0
    segments, _ = _get_whisper_model().transcribe(audio_array, language=WHISPER_LANG)
    return " ".join(seg.text.strip() for seg in segments).strip()


def _call_with_retry(fn, *args, **kwargs) -> str:
    for attempt in range(API_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < API_MAX_RETRIES - 1:
                sleep_s = (2 ** attempt) * API_RETRY_DELAY
                logger.warning("%s error %s, retrying in %.1fs", fn.__name__, e, sleep_s)
                time.sleep(sleep_s)
            else:
                logger.error("%s failed after %d retries", fn.__name__, API_MAX_RETRIES)
    return "---UNMATCHED---"


def parse_paragraphs(txt_path: Path) -> list[str]:
    text = re.sub(r"\d", "", txt_path.read_text(encoding="utf-8"))
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", errors="ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    return " ".join(s.split())


def _find_in_window(
    matched_text: str, paragraphs: list[str], window_start: int
) -> tuple[int, int] | None:
    """Case/whitespace-insensitive exact match within the current window.

    Returns (new_window_start, para_idx), or None if not found.

    The window advances only when the match is beyond the first paragraph,
    which tells us the first paragraph is no longer relevant. The matched
    paragraph itself becomes the new first paragraph (anchor) on the next call.
    """
    window_end = min(window_start + WINDOW_SIZE, len(paragraphs))
    needle = _normalize(matched_text)

    for i in range(window_start, window_end):
        if needle in _normalize(paragraphs[i]):
            # Only slide window_start forward when we've confirmed paragraph
            # window_start is behind us (i.e. the match is in a later paragraph).
            new_start = i if i > window_start else window_start
            return (new_start, i)

    return None


def align_utterance_to_text(
    whisper_text: str,
    paragraphs: list[str],
    window_start: int,
) -> tuple[str, int, int]:
    """Returns (matched_text, new_window_start, matched_paragraph_index).

    matched_paragraph_index is -1 when no match is found.
    Claude is called at most twice: if the first response doesn't produce an
    exact match in the window, one retry is attempted before giving up.
    """
    if window_start >= len(paragraphs):
        logger.info("Skipping alignment: window_start=%d >= len(paragraphs)=%d", window_start, len(paragraphs))
        return ("", window_start, -1)
    if not whisper_text.strip():
        logger.info("Skipping alignment: empty whisper text")
        return ("", window_start, -1)

    ipa = _call_with_retry(get_ipa_string, LANG_NAME, whisper_text)
    if not ipa or "UNMATCHED" in ipa:
        logger.info("IPA generation failed or returned unmatched for: %.60s", whisper_text)
        return ("", window_start, -1)
    logger.info("IPA result: %s", ipa)

    window_text = "\n\n".join(paragraphs[window_start: window_start + WINDOW_SIZE])

    for _ in range(2):
        result = _call_with_retry(
            identify_phonetic_component_within_text, LANG_NAME, ipa, window_text
        )
        result = result.strip().strip("`")
        logger.info("Alignment result: %s", result)

        if "unmatched" in result.lower():
            logger.info("No match found for: %.60s", whisper_text)
            return ("", window_start, -1)

        match = _find_in_window(result, paragraphs, window_start)
        if match is not None:
            logger.info("Match found: %s", result)
            new_start, para_idx = match
            return (result, new_start, para_idx)

    return ("", window_start, -1)


def _finalize_segment(utterances: list[Utterance]) -> tuple[AudioSegment, str]:
    audio = utterances[0].audio
    for u in utterances[1:]:
        audio = audio + u.audio
    text = " ".join(u.matched_text for u in utterances if u.matched_text)
    return (audio, text)


def stitch_segments(utterances: list[Utterance]) -> list[tuple[AudioSegment, str]]:
    current: list[Utterance] = []
    output: list[tuple[AudioSegment, str]] = []

    for u in utterances:
        if not u.is_matched:
            if current:
                output.append(_finalize_segment(current))
                current = []
            continue

        current.append(u)

        # When the window overflows, emit everything except the newest utterance,
        # then slide forward by dropping the oldest — producing overlapping segments
        # (e.g. ABC → BCD → CDE).
        while sum(x.duration_ms for x in current) > MAX_SEGMENT_MS and len(current) > 1:
            output.append(_finalize_segment(current[:-1]))
            current = current[1:]

    if current:
        output.append(_finalize_segment(current))

    return output


def slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", errors="ignore").decode()
    slug = re.sub(r"[^\w]", "_", normalized)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:60]


def _talk_segment_prefix(pair: TalkPair) -> str:
    year_num = pair.year.replace("y_", "")
    month_num = pair.month.replace("m_", "")
    return f"seg_{year_num}_{month_num}_{slugify(pair.stem)}_"


def export_segment(
    audio: AudioSegment,
    text: str,
    pair: TalkPair,
    seg_index: int,
    metadata_handle,
    data_dir: Path = DATA_DIR,
    synthetic: bool = False,
) -> None:
    resampled = audio.set_frame_rate(16_000).set_channels(1).set_sample_width(2)
    filename = f"{_talk_segment_prefix(pair)}{seg_index:05d}.wav"
    wav_path = data_dir / filename
    resampled.export(wav_path, format="wav")

    record = {"file_name": f"data/{filename}", "transcription": text, "synthetic": synthetic}
    metadata_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata_handle.flush()


def _talk_already_processed(pair: TalkPair, data_dir: Path) -> bool:
    return any(data_dir.glob(f"{_talk_segment_prefix(pair)}*.wav"))


def process_talk(
    pair: TalkPair, seg_counter: list[int], metadata_handle, data_dir: Path = DATA_DIR
) -> tuple[int, int]:
    """Returns (n_real_segments, n_tts_segments)."""
    logger.info("Processing %s/%s/%s", pair.year, pair.month, pair.stem)

    chunks = chunk_audio_on_silence(pair.mp3_path)
    if not chunks:
        logger.warning("No audio chunks for %s", pair.stem)
        return (0, 0)

    paragraphs = parse_paragraphs(pair.txt_path)
    if not paragraphs:
        logger.warning("No paragraphs for %s", pair.stem)
        return (0, 0)

    utterances: list[Utterance] = []
    window_start = 0
    matched_para_indices: set[int] = set()

    for i, chunk in enumerate(chunks):
        whisper_text = transcribe_chunk(chunk)
        logger.info("Chunk %d whisper: %.80s", i, whisper_text)
        matched_text, new_window_start, para_idx = align_utterance_to_text(
            whisper_text, paragraphs, window_start
        )
        is_matched = bool(matched_text)
        if is_matched:
            window_start = new_window_start
            matched_para_indices.add(para_idx)
            logger.debug("Chunk %d matched: %s", i, matched_text[:60])
        else:
            logger.debug("Chunk %d unmatched (whisper: %s)", i, whisper_text[:40])

        utterances.append(Utterance(
            audio=chunk,
            whisper_text=whisper_text,
            matched_text=matched_text,
            is_matched=is_matched,
            duration_ms=len(chunk),
        ))

    segments = stitch_segments(utterances)
    for audio, text in segments:
        export_segment(audio, text, pair, seg_counter[0], metadata_handle, data_dir, synthetic=False)
        seg_counter[0] += 1

    tts_count = 0
    n_unmatched = len(paragraphs) - len(matched_para_indices)
    if len(paragraphs) > 0 and len(matched_para_indices) == 0:
        logger.warning(
            "Talk %s: 0/%d paragraphs matched — skipping TTS fallback to avoid all-synthetic output",
            pair.stem, len(paragraphs),
        )
        return (0, 0)

    for para_idx, para in enumerate(paragraphs):
        if para_idx not in matched_para_indices:
            logger.debug("Generating TTS for unmatched paragraph %d: %.60s", para_idx, para)
            tts_audio = synthesize_tts_audio(para)
            if tts_audio is not None:
                export_segment(tts_audio, para, pair, seg_counter[0], metadata_handle, data_dir, synthetic=True)
                seg_counter[0] += 1
                tts_count += 1

    n_real = len(segments)
    n_total_paras = len(paragraphs)
    n_matched_paras = len(matched_para_indices)
    pct_synthetic = 100.0 * n_unmatched / n_total_paras if n_total_paras else 0.0
    logger.info(
        "Talk %s: %d/%d paragraphs matched to real audio, %d/%d needed TTS (%.1f%% synthetic) "
        "→ %d real segments, %d TTS segments",
        pair.stem,
        n_matched_paras, n_total_paras,
        n_unmatched, n_total_paras, pct_synthetic,
        n_real, tts_count,
    )
    return (n_real, tts_count)


def run_pipeline(
    mp3_root: Path = MP3_ROOT,
    txt_root: Path = TXT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    limit: Optional[int] = None,
) -> None:
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_root / "metadata.jsonl"

    pairs = find_talk_pairs(mp3_root, txt_root)
    if limit is not None:
        pairs = pairs[:limit]

    seg_counter = [sum(1 for _ in data_dir.glob("*.wav"))]
    total_real = 0
    total_tts = 0

    with open(metadata_file, "a", encoding="utf-8") as meta_f:
        for pair in pairs:
            if _talk_already_processed(pair, data_dir):
                logger.info("Skipping already-processed talk: %s", pair.stem)
                continue
            try:
                n_real, n_tts = process_talk(pair, seg_counter, meta_f, data_dir)
                total_real += n_real
                total_tts += n_tts
            except Exception as e:
                logger.error("Failed to process %s: %s", pair.stem, e, exc_info=True)

    total = total_real + total_tts
    pct_synthetic = 100.0 * total_tts / total if total else 0.0
    logger.info(
        "Pipeline complete: %d real segments, %d TTS segments, %d total (%.1f%% synthetic)",
        total_real, total_tts, total, pct_synthetic,
    )


if __name__ == "__main__":
    run_pipeline()
