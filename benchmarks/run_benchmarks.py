"""
run_benchmarks.py — run all strategies from the registry in parallel.

Usage:
    cd benchmarks
    python run_benchmarks.py [--langs eng spa] [--workers 5]

Results are cached in benchmarks/results/<label>/.  Re-running reuses cached
transcripts and only recomputes metrics (WER, recall, accuracy, LLM score).
"""

import argparse
import sys

import pandas as pd

from benchmark_utils import print_scores, run_all_benchmarks
from registry import STRATEGIES


def main():
    parser = argparse.ArgumentParser(description="Run transcription benchmarks")
    parser.add_argument(
        "--langs",
        nargs="+",
        metavar="LANG",
        default=None,
        help="Language codes to benchmark (e.g. eng spa). Default: all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        metavar="N",
        help="Max concurrent strategies (default: 5).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        default=None,
        help="Run only these strategy labels. Default: all.",
    )
    args = parser.parse_args()

    strategies = STRATEGIES
    if args.labels:
        strategies = [s for s in strategies if s["label"] in args.labels]
        if not strategies:
            print(f"No strategies matched labels: {args.labels}", file=sys.stderr)
            print(f"Available: {[s['label'] for s in STRATEGIES]}", file=sys.stderr)
            sys.exit(1)

    print(f"Running {len(strategies)} strategy/strategies with up to {args.workers} in parallel:")
    for s in strategies:
        print(f"  • {s['label']}")
    if args.langs:
        print(f"Languages: {args.langs}")
    print()

    results = run_all_benchmarks(
        strategies,
        langs=args.langs,
        max_workers=args.workers,
    )

    # Print scores for each strategy
    for label, df in sorted(results.items()):
        print_scores(df, label=label)

    # Print a combined summary table: WER per language per strategy
    print(f"\n\n{'=' * 66}")
    print("  SUMMARY — Mean WER by language")
    print(f"{'=' * 66}")
    frames = []
    for label, df in sorted(results.items()):
        valid = df.dropna(subset=["wer"])
        agg = valid.groupby("lang")["wer"].mean().rename(label)
        frames.append(agg)
    if frames:
        summary = pd.concat(frames, axis=1)
        summary.loc["overall"] = summary.mean()
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
