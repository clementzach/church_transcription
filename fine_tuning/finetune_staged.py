"""
Multi-stage fine-tuning pipeline for Whisper (Haitian Creole).

Stages run sequentially; each starts from the previous stage's saved model.

  Stage 1 — Full fine-tuning on phonetic (real) data
             LR 1e-5 | 3 000 steps | all layers trainable

  Stage 2 — Decoder-only fine-tuning on augmented synthetic data
             LR 5e-6 | 1 500 steps | encoder frozen
             Audio augmentation (volume + noise) applied before feature extraction;
             SpecAugment (time + frequency masking) applied in the data collator.
             Rationale: freezing the proven encoder and letting the decoder absorb
             synthetic transcription patterns mirrors the frozen-encoder approach
             that out-performed full fine-tuning on synthetic data in prior runs.

  Stage 3 — Full fine-tuning on phonetic data at lower LR
             LR 3e-6 | 1 500 steps | all layers trainable
             Refines the model back to gold-standard data after Stage 2.

  Stage 4 — (Recommended) Full fine-tuning on phonetic + augmented synthetic
             LR 1e-6 | 1 000 steps | all layers trainable | SpecAugment
             Very low LR preserves stages 1-3 while SpecAugment adds a final
             regularisation pass on mixed acoustic conditions.
"""

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import torch
from audiomentations import AddGaussianNoise, Compose, Gain, PitchShift, TimeStretch
from datasets import Dataset, concatenate_datasets
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from fine_tuning.finetune import (
    DataCollatorSpeechSeq2SeqWithPadding,
    _evaluate_checkpoint,
    load_candidate_datasets,
    make_compute_metrics,
    DATALOADER_NUM_WORKERS,
    DRY_RUN_EVAL_SAMPLES,
    DRY_RUN_MAX_STEPS,
    DRY_RUN_TRAIN_SAMPLES,
    GRADIENT_ACCUMULATION_STEPS,
    LANGUAGE,
    LANGUAGE_CODE,
    LOGGING_STEPS,
    MAX_LABEL_LENGTH,
    MODEL_OUTPUT_DIRS,
    MODEL_SIZES,
    PER_DEVICE_EVAL_BATCH_SIZE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    TASK,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = REPO_ROOT / "fine_tuning" / "output"

MODEL_OUTPUT_DIRS_STAGED = {
    size: REPO_ROOT / "fine_tuning" / f"whisper-{size}-ht-staged"
    for size in MODEL_SIZES
}

# ── Stage hyperparameters ──────────────────────────────────────────────────────
STAGE1_LR = 1e-5
STAGE1_STEPS = 3_000
STAGE1_WARMUP = 500

STAGE2_LR = 5e-6
STAGE2_STEPS = 1_500
STAGE2_WARMUP = 200

STAGE3_LR = 3e-6
STAGE3_STEPS = 1_500
STAGE3_WARMUP = 100

STAGE4_LR = 1e-6
STAGE4_STEPS = 1_000
STAGE4_WARMUP = 50

# ── SpecAugment parameters ─────────────────────────────────────────────────────
TIME_MASK_PARAM = 100   # max consecutive time frames masked per policy
FREQ_MASK_PARAM = 27    # max consecutive mel bins masked per policy
NUM_TIME_MASKS = 2
NUM_FREQ_MASKS = 2


# ── Audio augmentation ─────────────────────────────────────────────────────────

def _make_augmenter() -> Compose:
    return Compose([
        Gain(min_gain_db=-6, max_gain_db=6, p=1.0),
        AddGaussianNoise(min_amplitude=1e-4, max_amplitude=5e-3, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    ])


def _make_prepare_fn(processor: WhisperProcessor, augment: bool = False):
    augmenter = _make_augmenter() if augment else None

    def prepare_dataset(batch):
        audio = batch["audio"]
        arr = audio["array"]
        if augmenter is not None:
            arr = augmenter(arr, sample_rate=audio["sampling_rate"])
        batch["input_features"] = processor.feature_extractor(
            arr, sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    return prepare_dataset


def _preprocess(processor: WhisperProcessor, ds: Dataset, desc: str, augment: bool = False) -> Dataset:
    ds = ds.map(
        _make_prepare_fn(processor, augment=augment),
        remove_columns=ds.column_names,
        writer_batch_size=100,
        desc=desc,
    )
    before = len(ds)
    ds = ds.filter(lambda x: len(x["labels"]) <= MAX_LABEL_LENGTH)
    dropped = before - len(ds)
    if dropped:
        logger.warning("Dropped %d / %d samples exceeding %d-token label limit", dropped, before, MAX_LABEL_LENGTH)
    return ds


# ── SpecAugment data collator ──────────────────────────────────────────────────

@dataclass
class DataCollatorWithSpecAugment(DataCollatorSpeechSeq2SeqWithPadding):
    """Extends the base collator with optional SpecAugment on mel features."""
    apply_spec_augment: bool = False
    time_mask_param: int = TIME_MASK_PARAM
    freq_mask_param: int = FREQ_MASK_PARAM
    num_time_masks: int = NUM_TIME_MASKS
    num_freq_masks: int = NUM_FREQ_MASKS

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features)
        if self.apply_spec_augment:
            batch["input_features"] = self._spec_augment(batch["input_features"])
        return batch

    def _spec_augment(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, mel_bins, time_steps]"""
        B, F, T = features.shape
        features = features.clone()
        for _ in range(self.num_time_masks):
            t = min(int(torch.randint(1, self.time_mask_param + 1, (1,))), T)
            starts = torch.randint(0, T - t + 1, (B,))
            for i in range(B):
                features[i, :, starts[i]: starts[i] + t] = 0.0
        for _ in range(self.num_freq_masks):
            f = min(int(torch.randint(1, self.freq_mask_param + 1, (1,))), F)
            starts = torch.randint(0, F - f + 1, (B,))
            for i in range(B):
                features[i, starts[i]: starts[i] + f, :] = 0.0
        return features


# ── Model helpers ──────────────────────────────────────────────────────────────

def _load_model(
    model_path: str,
    freeze_encoder: bool = False,
) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
    model.config.use_cache = False
    model.generation_config.language = LANGUAGE_CODE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    if freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen — training decoder only")
        # Required for gradient checkpointing with partially-frozen models
        model.enable_input_require_grads()
    return model


def _build_trainer(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    train_ds: Dataset,
    eval_ds: Dataset,
    output_dir: Path,
    lr: float,
    max_steps: int,
    warmup_steps: int,
    apply_spec_augment: bool = False,
    dry_run: bool = False,
) -> Seq2SeqTrainer:
    batch_size = 2 if dry_run else PER_DEVICE_TRAIN_BATCH_SIZE
    eval_batch_size = 2 if dry_run else PER_DEVICE_EVAL_BATCH_SIZE
    grad_accum = 1 if dry_run else GRADIENT_ACCUMULATION_STEPS
    eval_save_steps = 1 if dry_run else max(1, max_steps // 6)

    data_collator = DataCollatorWithSpecAugment(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        apply_spec_augment=apply_spec_augment,
    )
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=lr,
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


def _run_stage(
    stage_name: str,
    model_path: str,
    processor: WhisperProcessor,
    train_ds: Dataset,
    eval_ds: Dataset,
    output_dir: Path,
    lr: float,
    max_steps: int,
    warmup_steps: int,
    freeze_encoder: bool = False,
    apply_spec_augment: bool = False,
    dry_run: bool = False,
) -> float:
    logger.info("=== %s ===", stage_name)
    model = _load_model(model_path, freeze_encoder=freeze_encoder)
    trainer = _build_trainer(
        model, processor, train_ds, eval_ds,
        output_dir=output_dir,
        lr=lr,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        apply_spec_augment=apply_spec_augment,
        dry_run=dry_run,
    )
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(str(output_dir))
    best_wer = trainer.state.best_metric or 0.0
    logger.info("%s best WER: %.2f%%", stage_name, best_wer)
    return best_wer


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_staged_finetuning(
    output_root: Path = OUTPUT_ROOT,
    model_size: str = "medium",
    skip_stage4: bool = False,
    dry_run: bool = False,
) -> None:
    if model_size not in MODEL_SIZES:
        raise ValueError(f"--model-size must be one of {tuple(MODEL_SIZES)}, got '{model_size}'")

    base_model = MODEL_SIZES[model_size]
    staged_dir = MODEL_OUTPUT_DIRS_STAGED[model_size]
    stage1_dir = staged_dir / "stage1_phonetic_full"
    stage2_dir = staged_dir / "stage2_synthetic_decoder"
    stage3_dir = staged_dir / "stage3_phonetic_low_lr"
    stage4_dir = staged_dir / "stage4_combined_specaugment"

    logger.info("Loading processor from %s", base_model)
    processor = WhisperProcessor.from_pretrained(base_model, language=LANGUAGE, task=TASK)

    logger.info("Loading datasets from %s", output_root)
    eval_raw, synth_train_raw, phonetic_train_raw, _ = load_candidate_datasets(output_root)

    if dry_run:
        logger.info("DRY RUN: %d train / %d eval samples, %d steps per stage",
                    DRY_RUN_TRAIN_SAMPLES, DRY_RUN_EVAL_SAMPLES, DRY_RUN_MAX_STEPS)
        eval_raw           = eval_raw.select(range(min(DRY_RUN_EVAL_SAMPLES, len(eval_raw))))
        phonetic_train_raw = phonetic_train_raw.select(range(min(DRY_RUN_TRAIN_SAMPLES, len(phonetic_train_raw))))
        synth_train_raw    = synth_train_raw.select(range(min(DRY_RUN_TRAIN_SAMPLES, len(synth_train_raw))))

    eval_ds      = _preprocess(processor, eval_raw,           desc="Preprocessing eval")
    phonetic_ds  = _preprocess(processor, phonetic_train_raw, desc="Preprocessing phonetic train")

    steps  = lambda s: DRY_RUN_MAX_STEPS if dry_run else s
    warmup = lambda w: 0 if dry_run else w

    # Stage 1 ── full fine-tuning on phonetic data
    _run_stage(
        "Stage 1: Full fine-tuning on phonetic data",
        model_path=base_model,
        processor=processor,
        train_ds=phonetic_ds,
        eval_ds=eval_ds,
        output_dir=stage1_dir,
        lr=STAGE1_LR,
        max_steps=steps(STAGE1_STEPS),
        warmup_steps=warmup(STAGE1_WARMUP),
        dry_run=dry_run,
    )

    # Stage 2 ── encoder-only on augmented synthetic data
    if len(synth_train_raw) == 0:
        logger.warning("No synthetic data — skipping Stage 2, Stage 3 starts from Stage 1")
        stage2_source = str(stage1_dir)
    else:
        synth_aug_ds = _preprocess(processor, synth_train_raw, desc="Preprocessing synthetic (augmented)", augment=True)
        _run_stage(
            "Stage 2: Decoder-only fine-tuning on augmented synthetic data",
            model_path=str(stage1_dir),
            processor=processor,
            train_ds=synth_aug_ds,
            eval_ds=eval_ds,
            output_dir=stage2_dir,
            lr=STAGE2_LR,
            max_steps=steps(STAGE2_STEPS),
            warmup_steps=warmup(STAGE2_WARMUP),
            freeze_encoder=True,
            apply_spec_augment=True,
            dry_run=dry_run,
        )
        stage2_source = str(stage2_dir)

    # Stage 3 ── full fine-tuning on phonetic data, lower LR
    _run_stage(
        "Stage 3: Full fine-tuning on phonetic data (lower LR)",
        model_path=stage2_source,
        processor=processor,
        train_ds=phonetic_ds,
        eval_ds=eval_ds,
        output_dir=stage3_dir,
        lr=STAGE3_LR,
        max_steps=steps(STAGE3_STEPS),
        warmup_steps=warmup(STAGE3_WARMUP),
        dry_run=dry_run,
    )

    # Stage 4 ── combined data + SpecAugment, very low LR (recommended)
    if not skip_stage4:
        if len(synth_train_raw) > 0:
            synth_ds4    = _preprocess(processor, synth_train_raw, desc="Preprocessing synthetic for Stage 4", augment=True)
            combined_ds4 = concatenate_datasets([phonetic_ds, synth_ds4]).shuffle(seed=42)
        else:
            combined_ds4 = phonetic_ds
        _run_stage(
            "Stage 4: Full fine-tuning on combined data with SpecAugment (recommended)",
            model_path=str(stage3_dir),
            processor=processor,
            train_ds=combined_ds4,
            eval_ds=eval_ds,
            output_dir=stage4_dir,
            lr=STAGE4_LR,
            max_steps=steps(STAGE4_STEPS),
            warmup_steps=warmup(STAGE4_WARMUP),
            apply_spec_augment=True,
            dry_run=dry_run,
        )

    if dry_run:
        shutil.rmtree(staged_dir, ignore_errors=True)
        logger.info("Dry run: deleted %s", staged_dir)
        return

    # ── Final comparison eval ──────────────────────────────────────────────────
    to_compare: Dict[str, str] = {}

    # All base model sizes
    for size, model_id in MODEL_SIZES.items():
        to_compare[f"base whisper-{size}"] = model_id

    # All saved staged checkpoints
    for stage_label, stage_path in [
        (f"{model_size}/staged/stage1", stage1_dir),
        (f"{model_size}/staged/stage2", stage2_dir),
        (f"{model_size}/staged/stage3", stage3_dir),
        (f"{model_size}/staged/stage4", stage4_dir),
    ]:
        if (stage_path / "config.json").exists():
            to_compare[stage_label] = str(stage_path)

    # All existing single-stage candidate checkpoints from finetune.py runs
    for size, size_dir in MODEL_OUTPUT_DIRS.items():
        for candidate_dir in sorted(size_dir.glob("*/candidate_*")):
            if (candidate_dir / "config.json").exists():
                label = f"{size}/{candidate_dir.parent.name}/{candidate_dir.name}"
                to_compare[label] = str(candidate_dir)

    logger.info("Running comparison eval on %d models...", len(to_compare))
    tmp_dir = staged_dir / "_eval_tmp"
    comparison: Dict[str, Dict] = {}
    for label, path in to_compare.items():
        logger.info("  Evaluating: %s", label)
        comparison[label] = _evaluate_checkpoint(path, eval_raw, tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    sorted_results = sorted(
        comparison.items(),
        key=lambda x: x[1]["wer"] if x[1]["wer"] is not None else float("inf"),
    )
    logger.info("=" * 65)
    logger.info("%-42s  %8s  %8s", "Model", "WER", "CER")
    logger.info("-" * 65)
    for label, metrics in sorted_results:
        wer = f"{metrics['wer']:.2f}%" if metrics["wer"] is not None else "failed"
        cer = f"{metrics['cer']:.2f}%" if metrics["cer"] is not None else "failed"
        logger.info("%-42s  %8s  %8s", label, wer, cer)
    logger.info("=" * 65)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-stage Whisper fine-tuning pipeline")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_SIZES),
        default="medium",
        help="Which Whisper model to fine-tune (default: medium)",
    )
    parser.add_argument(
        "--skip-stage4",
        action="store_true",
        help="Skip Stage 4 (combined data + SpecAugment)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Run on {DRY_RUN_TRAIN_SAMPLES}/{DRY_RUN_EVAL_SAMPLES} samples for {DRY_RUN_MAX_STEPS} steps per stage, then delete output",
    )
    args = parser.parse_args()

    run_staged_finetuning(
        output_root=args.output_root,
        model_size=args.model_size,
        skip_stage4=args.skip_stage4,
        dry_run=args.dry_run,
    )
