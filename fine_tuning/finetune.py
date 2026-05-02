import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import Audio, concatenate_datasets, load_dataset
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
EVAL_SPLIT = 0.05

# Tuned for L4 (24 GB VRAM), 16 GB RAM, 4 vCPUs
# Effective batch size = 16 * 2 = 32
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
LOGGING_STEPS = 25
DATALOADER_NUM_WORKERS = 2

# Phase 1: real speech only — establishes the baseline
# Phase 2: adds synthetic data — measures the incremental benefit
PHASE1_MAX_STEPS = 3000
PHASE2_MAX_STEPS = 1000


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


def load_split_datasets(output_root: Path):
    """Load the audiofolder dataset and split into real-train, all-train, and eval.

    Eval is drawn exclusively from real speech so the benchmark is consistent
    across both phases.
    """
    dataset = load_dataset("audiofolder", data_dir=str(output_root), split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    real_ds = dataset.filter(lambda x: not x["synthetic"])
    synth_ds = dataset.filter(lambda x: x["synthetic"])

    real_split = real_ds.train_test_split(test_size=EVAL_SPLIT, seed=42)
    real_train = real_split["train"]
    eval_ds = real_split["test"]

    # Combine real train + all synthetic, then shuffle
    if len(synth_ds) > 0:
        all_train = concatenate_datasets([real_train, synth_ds]).shuffle(seed=42)
    else:
        all_train = real_train

    logger.info(
        "Dataset split — real train: %d, synthetic: %d, eval: %d",
        len(real_train), len(synth_ds), len(eval_ds),
    )
    return real_train, all_train, eval_ds, len(synth_ds)


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
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics


def _preprocess_datasets(processor, train_ds, eval_ds):
    prepare_fn = make_prepare_fn(processor)
    col_names = train_ds.column_names
    train_ds = train_ds.map(
        prepare_fn,
        remove_columns=col_names,
        num_proc=DATALOADER_NUM_WORKERS,
        desc="Preprocessing train",
    )
    eval_ds = eval_ds.map(
        prepare_fn,
        remove_columns=eval_ds.column_names,
        num_proc=DATALOADER_NUM_WORKERS,
        desc="Preprocessing eval",
    )
    return train_ds, eval_ds


def _build_trainer(
    model,
    processor,
    train_ds,
    eval_ds,
    output_dir: Path,
    max_steps: int,
    warmup_steps: int,
) -> Seq2SeqTrainer:
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        fp16=True,
        eval_strategy="steps",
        eval_steps=max_steps // 6,
        save_strategy="steps",
        save_steps=max_steps // 6,
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
        tokenizer=processor.feature_extractor,
    )


def run_finetuning(
    output_root: Path = OUTPUT_ROOT,
    model_output_dir: Path = MODEL_OUTPUT_DIR,
    resume_from_checkpoint: str | None = None,
):
    logger.info("Loading processor from %s", BASE_MODEL)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)

    logger.info("Loading dataset from %s", output_root)
    real_train_raw, all_train_raw, eval_raw, n_synthetic = load_split_datasets(output_root)

    # Pre-process both training splits. HuggingFace datasets caches map outputs on
    # disk, so the eval set (shared across phases) is only processed once.
    real_train, eval_ds = _preprocess_datasets(processor, real_train_raw, eval_raw)
    all_train, _ = _preprocess_datasets(processor, all_train_raw, eval_raw)

    # ── Phase 1: real speech only ─────────────────────────────────────────────
    phase1_dir = model_output_dir / "phase1"
    logger.info("Phase 1: training on %d real samples → %s", len(real_train), phase1_dir)

    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.use_cache = False
    model.generation_config.language = LANGUAGE_CODE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    trainer1 = _build_trainer(
        model, processor, real_train, eval_ds,
        output_dir=phase1_dir,
        max_steps=PHASE1_MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
    )
    trainer1.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer1.save_model()
    processor.save_pretrained(str(phase1_dir))
    phase1_wer = trainer1.state.best_metric
    logger.info("Phase 1 complete — best WER: %.2f%%", phase1_wer or 0.0)

    # ── Phase 2: real + synthetic ─────────────────────────────────────────────
    if n_synthetic == 0:
        logger.warning("No synthetic samples found — skipping Phase 2")
        return

    phase2_dir = model_output_dir / "phase2"
    logger.info(
        "Phase 2: continuing from Phase 1 weights on %d samples (%d real + %d synthetic) → %s",
        len(all_train), len(real_train), n_synthetic, phase2_dir,
    )

    # Load Phase 1 weights fresh — don't restore optimizer state so Phase 2
    # is a clean fine-tune on top of the Phase 1 model.
    model2 = WhisperForConditionalGeneration.from_pretrained(str(phase1_dir))
    model2.config.use_cache = False
    model2.generation_config.language = LANGUAGE_CODE
    model2.generation_config.task = TASK
    model2.generation_config.forced_decoder_ids = None

    trainer2 = _build_trainer(
        model2, processor, all_train, eval_ds,
        output_dir=phase2_dir,
        max_steps=PHASE2_MAX_STEPS,
        warmup_steps=min(100, PHASE2_MAX_STEPS // 10),
    )
    trainer2.train()
    trainer2.save_model()
    processor.save_pretrained(str(phase2_dir))
    phase2_wer = trainer2.state.best_metric

    logger.info(
        "Training complete — Phase 1 WER (real only): %.2f%% | Phase 2 WER (real + synthetic): %.2f%%",
        phase1_wer or 0.0, phase2_wer or 0.0,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model-output-dir", type=Path, default=MODEL_OUTPUT_DIR)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Checkpoint path to resume Phase 1 from")
    args = parser.parse_args()

    run_finetuning(
        output_root=args.output_root,
        model_output_dir=args.model_output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
