import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import numpy as np
from faster_whisper import WhisperModel
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

SILENCE_MIN_LEN_MS = 500
SILENCE_THRESH_DBFS = -40
MIN_CHUNK_MS = 300
MAX_SEGMENT_MS = 30_000
WINDOW_SIZE = 3
LANG_NAME = "Haitian Creole"
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_LANG = "ht"
API_MAX_RETRIES = 3
API_RETRY_DELAY = 2.0

_whisper_model: WhisperModel | None = None


def _get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    return _whisper_model


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
    text = txt_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def advance_window(matched_text: str, paragraphs: list[str], window_start: int) -> int:
    window_end = min(window_start + WINDOW_SIZE, len(paragraphs))
    needle = matched_text.strip()

    for i in range(window_start, window_end):
        if needle in paragraphs[i]:
            return min(i + 1, len(paragraphs))

    # Token overlap fallback
    matched_tokens = set(needle.lower().split())
    if not matched_tokens:
        return window_start

    best_i, best_score = window_start, 0.0
    for i in range(window_start, window_end):
        para_tokens = set(paragraphs[i].lower().split())
        union = matched_tokens | para_tokens
        score = len(matched_tokens & para_tokens) / len(union) if union else 0.0
        if score > best_score:
            best_score, best_i = score, i

    return min(best_i + 1, len(paragraphs))


def align_utterance_to_text(
    whisper_text: str,
    paragraphs: list[str],
    window_start: int,
) -> tuple[str, int]:
    if window_start >= len(paragraphs):
        return ("", window_start)
    if not whisper_text.strip():
        return ("", window_start)

    ipa = _call_with_retry(get_ipa_string, LANG_NAME, whisper_text)
    if not ipa:
        return ("", window_start)

    window_text = "\n\n".join(paragraphs[window_start: window_start + WINDOW_SIZE])
    result = _call_with_retry(
        identify_phonetic_component_within_text, LANG_NAME, ipa, window_text
    )
    result = result.strip().strip("`")

    if result == "---UNMATCHED---":
        return ("", window_start)

    new_start = advance_window(result, paragraphs, window_start)
    return (result, new_start)


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

        current_total = sum(x.duration_ms for x in current)
        if current_total + u.duration_ms <= MAX_SEGMENT_MS:
            current.append(u)
        else:
            if current:
                output.append(_finalize_segment(current))
            current = current[1:] + [u]
            while len(current) > 1 and sum(x.duration_ms for x in current) > MAX_SEGMENT_MS:
                current = current[1:]

    if current:
        output.append(_finalize_segment(current))

    return output


def slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", errors="ignore").decode()
    slug = re.sub(r"[^\w]", "_", normalized)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:60]


def export_segment(
    audio: AudioSegment,
    text: str,
    pair: TalkPair,
    seg_index: int,
    metadata_handle,
    data_dir: Path = DATA_DIR,
) -> None:
    resampled = audio.set_frame_rate(16_000).set_channels(1).set_sample_width(2)
    year_num = pair.year.replace("y_", "")
    month_num = pair.month.replace("m_", "")
    slug = slugify(pair.stem)
    filename = f"seg_{year_num}_{month_num}_{slug}_{seg_index:05d}.wav"
    wav_path = data_dir / filename
    resampled.export(wav_path, format="wav")

    record = {"file_name": f"data/{filename}", "transcription": text}
    metadata_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata_handle.flush()


def _talk_already_processed(pair: TalkPair, data_dir: Path) -> bool:
    year_num = pair.year.replace("y_", "")
    month_num = pair.month.replace("m_", "")
    slug = slugify(pair.stem)
    prefix = f"seg_{year_num}_{month_num}_{slug}_"
    return any(data_dir.glob(f"{prefix}*.wav"))


def process_talk(pair: TalkPair, seg_counter: list[int], metadata_handle, data_dir: Path = DATA_DIR) -> int:
    logger.info("Processing %s/%s/%s", pair.year, pair.month, pair.stem)

    chunks = chunk_audio_on_silence(pair.mp3_path)
    if not chunks:
        logger.warning("No audio chunks for %s", pair.stem)
        return 0

    paragraphs = parse_paragraphs(pair.txt_path)
    if not paragraphs:
        logger.warning("No paragraphs for %s", pair.stem)
        return 0

    utterances: list[Utterance] = []
    window_start = 0

    for i, chunk in enumerate(chunks):
        whisper_text = transcribe_chunk(chunk)
        matched_text, new_window_start = align_utterance_to_text(
            whisper_text, paragraphs, window_start
        )
        is_matched = bool(matched_text)
        if is_matched:
            window_start = new_window_start
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
        export_segment(audio, text, pair, seg_counter[0], metadata_handle, data_dir)
        seg_counter[0] += 1

    matched_count = sum(1 for u in utterances if u.is_matched)
    logger.info(
        "Talk %s: %d chunks, %d matched, %d segments written",
        pair.stem, len(chunks), matched_count, len(segments),
    )
    return len(segments)


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

    seg_counter = [0]
    total_segments = 0

    with open(metadata_file, "a", encoding="utf-8") as meta_f:
        for pair in pairs:
            if _talk_already_processed(pair, data_dir):
                logger.info("Skipping already-processed talk: %s", pair.stem)
                continue
            try:
                n = process_talk(pair, seg_counter, meta_f, data_dir)
                total_segments += n
            except Exception as e:
                logger.error("Failed to process %s: %s", pair.stem, e, exc_info=True)

    logger.info("Pipeline complete: %d total segments written", total_segments)


if __name__ == "__main__":
    run_pipeline()
