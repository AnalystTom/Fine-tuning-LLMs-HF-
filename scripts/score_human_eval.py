"""Score blind-review human ratings and produce a weighted summary."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .experiment_config import (
        BUCKET_METRICS,
        BUCKET_PRIMARY_METRIC,
        BUCKET_WEIGHTS,
        RUN1_BLIND_EVAL_SUMMARY_PATH,
        RUN1_BLIND_REVIEW_KEY_PATH,
        RUN1_HUMAN_SCORES_PATH,
    )
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import (
        BUCKET_METRICS,
        BUCKET_PRIMARY_METRIC,
        BUCKET_WEIGHTS,
        RUN1_BLIND_EVAL_SUMMARY_PATH,
        RUN1_BLIND_REVIEW_KEY_PATH,
        RUN1_HUMAN_SCORES_PATH,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=RUN1_HUMAN_SCORES_PATH,
        help="Completed human scores CSV.",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=RUN1_BLIND_REVIEW_KEY_PATH,
        help="Blind review key CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN1_BLIND_EVAL_SUMMARY_PATH,
        help="Markdown summary output path.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_metric(row: dict[str, str], prefix: str, metric: str) -> float:
    key = f"{prefix}_{metric}"
    raw = row.get(key, "").strip()
    if not raw:
        raise ValueError(f"Missing score for {key} in review_id={row.get('review_id')}")
    return float(raw)


def score_review_row(row: dict[str, str]) -> dict[str, Any]:
    bucket = row["bucket"]
    metrics = BUCKET_METRICS[bucket]
    a_scores = [parse_metric(row, "a", metric) for metric in metrics]
    b_scores = [parse_metric(row, "b", metric) for metric in metrics]
    a_mean = sum(a_scores) / len(a_scores)
    b_mean = sum(b_scores) / len(b_scores)
    if a_mean > b_mean:
        winner = "A"
    elif b_mean > a_mean:
        winner = "B"
    else:
        primary = BUCKET_PRIMARY_METRIC[bucket]
        a_primary = parse_metric(row, "a", primary)
        b_primary = parse_metric(row, "b", primary)
        if a_primary > b_primary:
            winner = "A"
        elif b_primary > a_primary:
            winner = "B"
        else:
            winner = "tie"
    return {
        "review_id": row["review_id"],
        "bucket": bucket,
        "winner": winner,
        "a_mean": a_mean,
        "b_mean": b_mean,
        "a_authenticity": parse_metric(row, "a", "authenticity"),
        "b_authenticity": parse_metric(row, "b", "authenticity"),
        "notes": row.get("notes", ""),
    }


def score_reviews(rows: list[dict[str, str]], key_rows: list[dict[str, str]]) -> dict[str, Any]:
    key_map = {row["review_id"]: row for row in key_rows}
    scored_rows = [score_review_row(row) for row in rows]
    per_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_wins = defaultdict(int)
    authenticity_wins = defaultdict(int)

    for row in scored_rows:
        key = key_map[row["review_id"]]
        per_bucket[row["bucket"]].append(row)
        if row["winner"] == "A":
            model_wins[key["model_a"]] += 1
        elif row["winner"] == "B":
            model_wins[key["model_b"]] += 1
        if row["a_authenticity"] > row["b_authenticity"]:
            authenticity_wins[key["model_a"]] += 1
        elif row["b_authenticity"] > row["a_authenticity"]:
            authenticity_wins[key["model_b"]] += 1

    bucket_results: dict[str, Any] = {}
    weighted_score = 0.0
    total_weight = 0.0
    for bucket, items in per_bucket.items():
        wins = defaultdict(int)
        for row in items:
            key = key_map[row["review_id"]]
            if row["winner"] == "A":
                wins[key["model_a"]] += 1
            elif row["winner"] == "B":
                wins[key["model_b"]] += 1
        bucket_results[bucket] = {
            "count": len(items),
            "wins": dict(wins),
        }
        total_bucket = max(1, len(items))
        finetuned_wins = sum(
            count for model, count in wins.items() if "finetuned" in model.lower()
        )
        bucket_rate = finetuned_wins / total_bucket
        weighted_score += bucket_rate * BUCKET_WEIGHTS[bucket]
        total_weight += BUCKET_WEIGHTS[bucket]

    return {
        "total_rows": len(scored_rows),
        "model_wins": dict(model_wins),
        "authenticity_wins": dict(authenticity_wins),
        "bucket_results": bucket_results,
        "weighted_finetuned_win_rate": weighted_score / total_weight if total_weight else 0.0,
        "scored_rows": scored_rows,
    }


def render_summary(summary: dict[str, Any]) -> str:
    lines = ["# Blind Human Evaluation Summary", ""]
    lines.append(f"- Total review rows: `{summary['total_rows']}`")
    lines.append(
        f"- Weighted fine-tuned win rate: `{summary['weighted_finetuned_win_rate']:.2%}`"
    )
    lines.append("")
    lines.append("## Overall Wins")
    lines.append("")
    for model, wins in sorted(summary["model_wins"].items()):
        lines.append(f"- `{model}`: `{wins}` row wins")
    lines.append("")
    lines.append("## Authenticity Wins")
    lines.append("")
    for model, wins in sorted(summary["authenticity_wins"].items()):
        lines.append(f"- `{model}`: `{wins}` authenticity wins")
    lines.append("")
    lines.append("## Bucket Results")
    lines.append("")
    for bucket, result in sorted(summary["bucket_results"].items()):
        lines.append(f"### {bucket}")
        lines.append("")
        lines.append(f"- Rows: `{result['count']}`")
        for model, wins in sorted(result["wins"].items()):
            lines.append(f"- `{model}` wins: `{wins}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    summary = score_reviews(read_csv(args.scores), read_csv(args.key))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_summary(summary), encoding="utf-8")
    print(f"Wrote blind human evaluation summary to {args.output}")


if __name__ == "__main__":
    main()
