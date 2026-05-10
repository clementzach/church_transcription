"""
registry.py — defines the benchmark strategies to run.

Each strategy is a dict with:
    label — filesystem-safe name (no spaces); used as the results subdirectory
    fn    — transcribe callable with signature (audio_path, lang) -> str

All strategies pass an explicit language hint to the provider API.
Gladia uses language_config.languages; AssemblyAI uses language_code; Whisper
uses the language parameter natively.
Use run_all_benchmarks(STRATEGIES) to run them in parallel.
"""

from functools import partial

from benchmark_utils import (
    CHURCH_VOCABULARY,
    transcribe_assemblyai,
    transcribe_gladia,
    transcribe_whisper,
)

STRATEGIES: list[dict] = [
    # ── Gladia (solaria-1) ─────────────────────────────────────────────────────
    {
        "label": "gladia_lang",
        "fn":    partial(transcribe_gladia, vocabulary=[], language_hint=True),
    },
    # ── AssemblyAI Universal Streaming (explicit language hint) ───────────────
    {
        "label": "assemblyai_multilingual_lang",
        "fn":    partial(
            transcribe_assemblyai,
            speech_model="universal-streaming-multilingual",
            language_hint=True,
        ),
    },
    {
        "label": "assemblyai_u3_pro_lang",
        "fn":    partial(
            transcribe_assemblyai,
            speech_model="u3-rt-pro",
            language_hint=True,
        ),
    },
    # ── OpenAI Whisper (already passes language by default) ───────────────────
    {
        "label": "whisper_api",
        "fn":    transcribe_whisper,
    },
]
