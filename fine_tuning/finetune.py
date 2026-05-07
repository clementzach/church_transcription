import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import torch
from datasets import Audio, Dataset, concatenate_datasets
from dotenv import load_dotenv
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = REPO_ROOT / "fine_tuning" / "output"
MODEL_OUTPUT_DIR = REPO_ROOT / "fine_tuning" / "whisper-small-ht"

BASE_MODEL = "openai/whisper-small"
LANGUAGE = "Haitian Creole"
LANGUAGE_CODE = "ht"
TASK = "transcribe"
SAMPLING_RATE = 16_000

EVAL_TALK_IDS = frozenset({
    "2012_10_Nourise_a",
    "2021_10_Diyite_Pa_Vledi_Nou_Pafe",
    "2021_4_Kisa_Sove_nou_an_te_fe_pou_nou",
    "2021_10_Lanmou_Bondye_Sa_ki_bay_nanm_lan_plis_lajwa",
    "2012_10_Bon_Dezi_pou_Angaje",
    "2012_10_Eprev_lafwa_nou",
    "2015_4_Prezidan_Dieter_F_Uchtdorf",
    "2019_10_Bay_Espri_nou_Kontwol_Sou_Ko_nou",
    "2021_4_Chemen_alyans_lan",
    "2019_4_Jan_l_te_fe_a",
    "2014_4_Fotifye_nou_e_pran_kouraj",
    "2021_4_Bondye_Nan_Mitan_Nou",
    "2018_10_Vin_tounen_Sen_Denye_Jou_Egzanple",
    "2015_10_Elde_Quentin_L_Cook",
    "2012_4_Rapo_Depatman_Odit_Legliz_la_2011",
})

# Tuned for L4 (24 GB VRAM), 16 GB RAM, 4 vCPUs
# Effective batch size = 16 * 2 = 32
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
CANDIDATE_MAX_STEPS = 3000
LOGGING_STEPS = 25
DATALOADER_NUM_WORKERS = 2

CANDIDATES = ("synthetic", "phonetic", "combined")

DRY_RUN_TRAIN_SAMPLES = 8
DRY_RUN_EVAL_SAMPLES = 4
DRY_RUN_MAX_STEPS = 2


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip leading BOS that the decoder prepends during generation
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _extract_talk_id(file_name: str) -> str:
    """Stable talk identifier that is consistent across seg_ and syn_ prefixes.

    data/seg_2012_10_Aprann_avek_ke_nou_00003.wav
    data/syn_2012_10_Aprann_avek_ke_nou_00003.wav
    both → "2012_10_Aprann_avek_ke_nou"
    """
    stem = re.sub(r"_\d{5}$", "", Path(file_name).stem)
    return re.sub(r"^(seg|syn)_", "", stem)


def _load_audio_dataset(metadata_file: Path, output_root: Path) -> Dataset:
    """Load a metadata JSONL as a HuggingFace Dataset with audio paths resolved."""
    if not metadata_file.exists():
        logger.warning("Metadata file not found, returning empty dataset: %s", metadata_file)
        return Dataset.from_list([])

    records = []
    with open(metadata_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["audio"] = str(output_root / r["file_name"])
                records.append(r)

    ds = Dataset.from_list(records)
    if len(ds) == 0:
        return ds
    return ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))


def load_candidate_datasets(output_root: Path) -> tuple[Dataset, Dataset, Dataset, Dataset]:
    """Return (eval_ds, synth_train, phonetic_train, combined_train).

    Eval is drawn from exactly N_EVAL_TALKS real-speech talks (first N alphabetically).
    Those same talk IDs are excluded from every training set, including synthetic data,
    so no evaluation talk bleeds into any candidate's training run.
    """
    phonetic_full = _load_audio_dataset(output_root / "metadata.jsonl", output_root)
    phonetic_full = phonetic_full.filter(lambda x: not x["synthetic"])

    if len(phonetic_full) == 0:
        raise RuntimeError("No phonetic-alignment data found in metadata.jsonl")

    eval_talk_ids = EVAL_TALK_IDS
    logger.info("Eval talks (%d): %s", len(eval_talk_ids), sorted(eval_talk_ids))

    eval_ds = phonetic_full.filter(
        lambda x: _extract_talk_id(x["file_name"]) in eval_talk_ids
    )
    phonetic_train = phonetic_full.filter(
        lambda x: _extract_talk_id(x["file_name"]) not in eval_talk_ids
    )

    synth_full = _load_audio_dataset(output_root / "synthetic_metadata.jsonl", output_root)
    synth_train = synth_full.filter(
        lambda x: _extract_talk_id(x["file_name"]) not in eval_talk_ids
    ) if len(synth_full) > 0 else synth_full

    if len(synth_train) > 0 and len(phonetic_train) > 0:
        combined_train = concatenate_datasets([phonetic_train, synth_train]).shuffle(seed=42)
    elif len(synth_train) > 0:
        combined_train = synth_train
    else:
        combined_train = phonetic_train

    logger.info(
        "Datasets — eval: %d | phonetic train: %d | synth train: %d | combined train: %d",
        len(eval_ds), len(phonetic_train), len(synth_train), len(combined_train),
    )
    return eval_ds, synth_train, phonetic_train, combined_train


def make_prepare_fn(processor: WhisperProcessor):
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch
    return prepare_dataset


def make_compute_metrics(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": 100 * cer_metric.compute(predictions=pred_str, references=label_str),
        }

    return compute_metrics


def _preprocess(processor, ds: Dataset, desc: str) -> Dataset:
    prepare_fn = make_prepare_fn(processor)
    return ds.map(
        prepare_fn,
        remove_columns=ds.column_names,
        num_proc=1,
        desc=desc,
    )


def _build_trainer(
    model,
    processor,
    train_ds,
    eval_ds,
    output_dir: Path,
    max_steps: int,
    warmup_steps: int,
    dry_run: bool = False,
) -> Seq2SeqTrainer:
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    batch_size = 2 if dry_run else PER_DEVICE_TRAIN_BATCH_SIZE
    eval_batch_size = 2 if dry_run else PER_DEVICE_EVAL_BATCH_SIZE
    grad_accum = 1 if dry_run else GRADIENT_ACCUMULATION_STEPS
    eval_save_steps = 1 if dry_run else max(1, max_steps // 6)
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=3,
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=True,
        push_to_hub=False,
    )
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )


def _fresh_model() -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.use_cache = False
    model.generation_config.language = LANGUAGE_CODE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    return model


def run_finetuning(
    output_root: Path = OUTPUT_ROOT,
    model_output_dir: Path = MODEL_OUTPUT_DIR,
    candidate: str = "all",
    resume_from_checkpoint: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    if candidate != "all" and candidate not in CANDIDATES:
        raise ValueError(f"--candidate must be one of {('all',) + CANDIDATES}, got '{candidate}'")

    logger.info("Loading processor from %s", BASE_MODEL)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)

    if dry_run:
        logger.info("DRY RUN: using %d train / %d eval samples, %d steps",
                    DRY_RUN_TRAIN_SAMPLES, DRY_RUN_EVAL_SAMPLES, DRY_RUN_MAX_STEPS)

    logger.info("Loading datasets from %s", output_root)
    eval_raw, synth_train_raw, phonetic_train_raw, combined_train_raw = load_candidate_datasets(output_root)

    if dry_run:
        eval_raw = eval_raw.select(range(min(DRY_RUN_EVAL_SAMPLES, len(eval_raw))))

    # Preprocess eval once; all three candidates share it
    eval_ds = _preprocess(processor, eval_raw, desc="Preprocessing eval")

    candidate_data = {
        "synthetic": synth_train_raw,
        "phonetic":  phonetic_train_raw,
        "combined":  combined_train_raw,
    }

    to_run = CANDIDATES if candidate == "all" else (candidate,)
    results: dict[str, float | None] = {}

    for name in to_run:
        train_raw = candidate_data[name]
        if dry_run:
            train_raw = train_raw.select(range(min(DRY_RUN_TRAIN_SAMPLES, len(train_raw))))
        if len(train_raw) == 0:
            logger.warning("Candidate '%s' has no training data — skipping", name)
            results[name] = None
            continue

        out_dir = model_output_dir / f"candidate_{name}"
        logger.info(
            "Training candidate '%s' on %d samples → %s",
            name, len(train_raw), out_dir,
        )

        max_steps = DRY_RUN_MAX_STEPS if dry_run else CANDIDATE_MAX_STEPS
        warmup = 0 if dry_run else WARMUP_STEPS
        train_ds = _preprocess(processor, train_raw, desc=f"Preprocessing {name} train")
        model = _fresh_model()
        trainer = _build_trainer(
            model, processor, train_ds, eval_ds,
            output_dir=out_dir,
            max_steps=max_steps,
            warmup_steps=warmup,
            dry_run=dry_run,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model()
        processor.save_pretrained(str(out_dir))
        results[name] = trainer.state.best_metric
        logger.info("Candidate '%s' best WER: %.2f%%", name, results[name] or 0.0)

        if dry_run:
            shutil.rmtree(out_dir, ignore_errors=True)
            logger.info("Dry run: deleted %s", out_dir)

    logger.info(
        "Results — synthetic: %s | phonetic: %s | combined: %s",
        f"{results.get('synthetic'):.2f}%" if results.get("synthetic") is not None else "skipped",
        f"{results.get('phonetic'):.2f}%" if results.get("phonetic") is not None else "skipped",
        f"{results.get('combined'):.2f}%" if results.get("combined") is not None else "skipped",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model-output-dir", type=Path, default=MODEL_OUTPUT_DIR)
    parser.add_argument(
        "--candidate",
        choices=("all",) + CANDIDATES,
        default="all",
        help="Which candidate model to train (default: all three)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to resume from (only meaningful with --candidate <single>)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Run on {DRY_RUN_TRAIN_SAMPLES} train / {DRY_RUN_EVAL_SAMPLES} eval samples for {DRY_RUN_MAX_STEPS} steps, then delete output",
    )
    args = parser.parse_args()

    run_finetuning(
        output_root=args.output_root,
        model_output_dir=args.model_output_dir,
        candidate=args.candidate,
        resume_from_checkpoint=args.resume_from_checkpoint,
        dry_run=args.dry_run,
    )
