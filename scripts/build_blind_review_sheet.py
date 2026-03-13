"""Build a blind A/B review sheet and scoring template from evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

try:
    from .experiment_config import (
        ALL_REVIEW_METRICS,
        RUN1_BASELINE_OUTPUTS_PATH,
        RUN1_BLIND_REVIEW_KEY_PATH,
        RUN1_BLIND_REVIEW_PATH,
        RUN1_FINETUNED_OUTPUTS_PATH,
        RUN1_HUMAN_SCORES_PATH,
    )
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import (
        ALL_REVIEW_METRICS,
        RUN1_BASELINE_OUTPUTS_PATH,
        RUN1_BLIND_REVIEW_KEY_PATH,
        RUN1_BLIND_REVIEW_PATH,
        RUN1_FINETUNED_OUTPUTS_PATH,
        RUN1_HUMAN_SCORES_PATH,
    )


REVIEW_COLUMNS = (
    "review_id",
    "bucket",
    "prompt_id",
    "seed",
    "platform",
    "prompt_text",
    "output_a",
    "output_b",
)

KEY_COLUMNS = (
    "review_id",
    "model_a",
    "model_b",
)

SCORES_COLUMNS = (
    "review_id",
    "bucket",
    *[f"a_{metric}" for metric in ALL_REVIEW_METRICS],
    *[f"b_{metric}" for metric in ALL_REVIEW_METRICS],
    "winner",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=RUN1_BASELINE_OUTPUTS_PATH,
        help="Baseline outputs JSONL.",
    )
    parser.add_argument(
        "--finetuned",
        type=Path,
        default=RUN1_FINETUNED_OUTPUTS_PATH,
        help="Fine-tuned outputs JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN1_BLIND_REVIEW_PATH,
        help="Blind review CSV output path.",
    )
    parser.add_argument(
        "--key-output",
        type=Path,
        default=RUN1_BLIND_REVIEW_KEY_PATH,
        help="Blind review key CSV output path.",
    )
    parser.add_argument(
        "--scores-template-output",
        type=Path,
        default=RUN1_HUMAN_SCORES_PATH,
        help="Human scores CSV template output path.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=3407,
        help="Deterministic seed for A/B randomization.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["prompt_id"]), int(row["seed"])


def build_blind_rows(
    baseline_rows: list[dict[str, Any]],
    finetuned_rows: list[dict[str, Any]],
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_map = {row_key(row): row for row in baseline_rows}
    finetuned_map = {row_key(row): row for row in finetuned_rows}
    keys = sorted(set(baseline_map) | set(finetuned_map))
    if set(baseline_map) != set(finetuned_map):
        raise ValueError("Baseline and fine-tuned outputs must contain identical prompt_id+seed keys")

    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    rng = random.Random(random_seed)

    for prompt_id, seed in keys:
        baseline = baseline_map[(prompt_id, seed)]
        finetuned = finetuned_map[(prompt_id, seed)]
        review_id = f"{prompt_id}__{seed}"
        swap = bool(rng.randint(0, 1))
        if swap:
            output_a, output_b = finetuned, baseline
            model_a, model_b = finetuned["model_label"], baseline["model_label"]
        else:
            output_a, output_b = baseline, finetuned
            model_a, model_b = baseline["model_label"], finetuned["model_label"]
        review_rows.append(
            {
                "review_id": review_id,
                "bucket": baseline.get("bucket") or finetuned.get("bucket"),
                "prompt_id": prompt_id,
                "seed": seed,
                "platform": baseline.get("platform") or finetuned.get("platform"),
                "prompt_text": baseline.get("prompt_text") or finetuned.get("prompt_text"),
                "output_a": output_a["response_text"],
                "output_b": output_b["response_text"],
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "model_a": model_a,
                "model_b": model_b,
            }
        )
        template_rows.append(
            {
                "review_id": review_id,
                "bucket": baseline.get("bucket") or finetuned.get("bucket"),
                **{f"a_{metric}": "" for metric in ALL_REVIEW_METRICS},
                **{f"b_{metric}": "" for metric in ALL_REVIEW_METRICS},
                "winner": "",
                "notes": "",
            }
        )
    return review_rows, key_rows, template_rows


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    review_rows, key_rows, template_rows = build_blind_rows(
        read_jsonl(args.baseline),
        read_jsonl(args.finetuned),
        args.random_seed,
    )
    write_csv(args.output, REVIEW_COLUMNS, review_rows)
    write_csv(args.key_output, KEY_COLUMNS, key_rows)
    write_csv(args.scores_template_output, SCORES_COLUMNS, template_rows)
    print(f"Blind review rows: {len(review_rows)}")
    print(f"Blind review CSV: {args.output}")
    print(f"Blind review key CSV: {args.key_output}")
    print(f"Human scores template CSV: {args.scores_template_output}")


if __name__ == "__main__":
    main()
