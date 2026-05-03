"""
Generates synthetic fine-tuning data using a HuggingFace SpeechT5 Haitian Creole TTS model.
All output has synthetic=True in metadata and uses the syn_ filename prefix.
Writes to synthetic_metadata.jsonl (separate from metadata.jsonl) so this script
is safe to run concurrently with the main pipeline.
"""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from pydub import AudioSegment
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).parent.parent
TXT_ROOT = REPO_ROOT / "conference_data" / "haitian_creole" / "text"
OUTPUT_ROOT = REPO_ROOT / "fine_tuning" / "output"
DATA_DIR = OUTPUT_ROOT / "data"
SYNTHETIC_METADATA_FILE = OUTPUT_ROOT / "synthetic_metadata.jsonl"

# HuggingFace model identifiers — update HF_MODEL_ID to the exact repo slug
HF_MODEL_ID = "idajikuu/SpeechT5_TTS_Haitian"
HF_VOCODER_ID = "microsoft/speecht5_hifigan"
SPEAKER_EMBEDDINGS_DATASET = "Matthijs/cmu-arctic-xvectors"
SPEAKER_IDX = 7306  # xvector index in the validation split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHARS = 600  # SpeechT5 input token limit safeguard

_processor: SpeechT5Processor | None = None
_model: SpeechT5ForTextToSpeech | None = None
_vocoder: SpeechT5HifiGan | None = None
_speaker_embeddings: torch.Tensor | None = None


def _get_tts_components() -> tuple:
    global _processor, _model, _vocoder, _speaker_embeddings
    if _model is None:
        logger.info("Loading SpeechT5 processor/model: %s", HF_MODEL_ID)
        _processor = SpeechT5Processor.from_pretrained(HF_MODEL_ID)
        _model = SpeechT5ForTextToSpeech.from_pretrained(HF_MODEL_ID).to(DEVICE)
        _vocoder = SpeechT5HifiGan.from_pretrained(HF_VOCODER_ID).to(DEVICE)
        embeddings_dataset = load_dataset(SPEAKER_EMBEDDINGS_DATASET, split="validation")
        _speaker_embeddings = (
            torch.tensor(embeddings_dataset[SPEAKER_IDX]["xvector"])
            .unsqueeze(0)
            .to(DEVICE)
        )
        logger.info("SpeechT5 components loaded on %s", DEVICE)
    return _processor, _model, _vocoder, _speaker_embeddings


def synthesize_tts_audio(text: str) -> AudioSegment | None:
    text = re.sub(r"\d", "", text).strip()
    if not text:
        return None
    try:
        processor, model, vocoder, speaker_embeddings = _get_tts_components()
        inputs = processor(text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            speech = model.generate_speech(
                inputs["input_ids"], speaker_embeddings, vocoder=vocoder
            )
        # SpeechT5 outputs float32 in [-1, 1] at 16 kHz mono
        pcm16 = (speech.cpu().numpy() * 32767).clip(-32768, 32767).astype(np.int16)
        return AudioSegment(
            pcm16.tobytes(),
            frame_rate=16_000,
            sample_width=2,
            channels=1,
        )
    except Exception as e:
        logger.error("SpeechT5 synthesis failed for '%.60s': %s", text, e)
        return None


@dataclass
class TalkSource:
    txt_path: Path
    year: str
    month: str
    stem: str


def find_talk_sources(txt_root: Path = TXT_ROOT) -> list[TalkSource]:
    sources = []
    for txt in txt_root.rglob("*.txt"):
        parts = txt.relative_to(txt_root).parts
        if len(parts) == 3:
            year, month, _ = parts
            sources.append(TalkSource(txt_path=txt, year=year, month=month, stem=txt.stem))
    sources.sort(key=lambda s: (s.year, s.month, s.stem))
    logger.info("Found %d text sources", len(sources))
    return sources


def parse_paragraphs(txt_path: Path) -> list[str]:
    text = re.sub(r"\d", "", txt_path.read_text(encoding="utf-8"))
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def _split_to_chunks(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split long paragraphs on sentence boundaries to stay within the model limit."""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence[:max_chars]
    if current:
        chunks.append(current)
    return chunks


def slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", errors="ignore").decode()
    slug = re.sub(r"[^\w]", "_", normalized)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:60]


def _synthetic_segment_prefix(source: TalkSource) -> str:
    year_num = source.year.replace("y_", "")
    month_num = source.month.replace("m_", "")
    return f"syn_{year_num}_{month_num}_{slugify(source.stem)}_"


def _talk_already_processed(source: TalkSource, data_dir: Path) -> bool:
    return any(data_dir.glob(f"{_synthetic_segment_prefix(source)}*.wav"))


def export_segment(
    audio: AudioSegment,
    text: str,
    source: TalkSource,
    seg_index: int,
    metadata_handle,
    data_dir: Path = DATA_DIR,
) -> None:
    resampled = audio.set_frame_rate(16_000).set_channels(1).set_sample_width(2)
    filename = f"{_synthetic_segment_prefix(source)}{seg_index:05d}.wav"
    wav_path = data_dir / filename
    resampled.export(wav_path, format="wav")

    record = {"file_name": f"data/{filename}", "transcription": text, "synthetic": True}
    metadata_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata_handle.flush()


def process_talk(
    source: TalkSource, metadata_handle, data_dir: Path = DATA_DIR
) -> int:
    """Returns the number of synthetic segments exported."""
    logger.info("Processing %s/%s/%s", source.year, source.month, source.stem)

    paragraphs = parse_paragraphs(source.txt_path)
    if not paragraphs:
        logger.warning("No paragraphs for %s", source.stem)
        return 0

    seg_counter = 0
    for para_idx, para in enumerate(paragraphs):
        for chunk in _split_to_chunks(para):
            audio = synthesize_tts_audio(chunk)
            if audio is not None:
                export_segment(audio, chunk, source, seg_counter, metadata_handle, data_dir)
                seg_counter += 1
            else:
                logger.warning(
                    "Skipping paragraph %d chunk (synthesis failed): %.60s", para_idx, chunk
                )

    logger.info("Talk %s: exported %d synthetic segments", source.stem, seg_counter)
    return seg_counter


def run_pipeline(
    txt_root: Path = TXT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    limit: Optional[int] = None,
) -> None:
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_root / "synthetic_metadata.jsonl"

    sources = find_talk_sources(txt_root)
    if limit is not None:
        sources = sources[:limit]

    total = 0
    with open(metadata_file, "a", encoding="utf-8") as meta_f:
        for source in sources:
            if _talk_already_processed(source, data_dir):
                logger.info("Skipping already-processed talk: %s", source.stem)
                continue
            try:
                total += process_talk(source, meta_f, data_dir)
            except Exception as e:
                logger.error("Failed to process %s: %s", source.stem, e, exc_info=True)

    logger.info("Synthetic pipeline complete: %d segments total", total)


if __name__ == "__main__":
    run_pipeline()
