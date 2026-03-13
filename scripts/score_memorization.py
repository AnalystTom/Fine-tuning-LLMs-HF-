"""Flag generated outputs that overlap too closely with the training corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .experiment_config import RUN1_MEMORIZATION_REPORT_PATH, RUN1_TRAIN_PATH
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import RUN1_MEMORIZATION_REPORT_PATH, RUN1_TRAIN_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=RUN1_TRAIN_PATH,
        help="Train JSONL containing assistant reference texts.",
    )
    parser.add_argument(
        "--generated",
        type=Path,
        action="append",
        required=True,
        help="Generated output JSONL from run_llamacpp_suite.py. Can be provided multiple times.",
    )
    parser.add_argument(
        "--threshold-words",
        type=int,
        default=14,
        help="Flag outputs with a contiguous overlap at or above this many words.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN1_MEMORIZATION_REPORT_PATH,
        help="JSON report output path.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def longest_contiguous_overlap(a_tokens: list[str], b_tokens: list[str]) -> int:
    if not a_tokens or not b_tokens:
        return 0
    previous = [0] * (len(b_tokens) + 1)
    best = 0
    for a_token in a_tokens:
        current = [0]
        for index, b_token in enumerate(b_tokens, start=1):
            if a_token == b_token:
                value = previous[index - 1] + 1
                current.append(value)
                if value > best:
                    best = value
            else:
                current.append(0)
        previous = current
    return best


def train_texts(path: Path) -> list[str]:
    rows = read_jsonl(path)
    return [str(row["messages"][2]["content"]) for row in rows]


def score_outputs(
    train_rows: list[str], generated_rows: list[dict[str, Any]], threshold_words: int
) -> list[dict[str, Any]]:
    train_tokens = [(text, tokenize(text)) for text in train_rows]
    scored: list[dict[str, Any]] = []
    for row in generated_rows:
        response_text = str(row["response_text"])
        response_tokens = tokenize(response_text)
        best_overlap = 0
        best_match = ""
        for train_text, candidate_tokens in train_tokens:
            overlap = longest_contiguous_overlap(response_tokens, candidate_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = train_text
        scored.append(
            {
                "model_label": row["model_label"],
                "bucket": row.get("bucket"),
                "prompt_id": row["prompt_id"],
                "seed": row.get("seed"),
                "overlap_words": best_overlap,
                "flagged": best_overlap >= threshold_words,
                "best_match_excerpt": best_match[:280],
            }
        )
    return scored


def main() -> None:
    args = parse_args()
    corpus = train_texts(args.train)
    report: dict[str, Any] = {
        "threshold_words": args.threshold_words,
        "generated_files": [],
    }
    for path in args.generated:
        rows = read_jsonl(path)
        scored = score_outputs(corpus, rows, args.threshold_words)
        report["generated_files"].append(
            {
                "path": str(path),
                "flagged_count": sum(1 for row in scored if row["flagged"]),
                "rows": scored,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote memorization report to {args.output}")


if __name__ == "__main__":
    main()
