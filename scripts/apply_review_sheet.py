"""Apply an edited review sheet to produce the curated run-1 corpus."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .experiment_config import (
        CURATED_PUBLIC_POSTS_PATH,
        CURATION_MIN_LINKEDIN_ROWS,
        CURATION_MIN_ROWS,
        PUBLIC_POSTS_PATH,
        PUBLIC_POSTS_REVIEW_PATH,
    )
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import (
        CURATED_PUBLIC_POSTS_PATH,
        CURATION_MIN_LINKEDIN_ROWS,
        CURATION_MIN_ROWS,
        PUBLIC_POSTS_PATH,
        PUBLIC_POSTS_REVIEW_PATH,
    )


TRUTHY = {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PUBLIC_POSTS_PATH,
        help="Source public post JSONL.",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=PUBLIC_POSTS_REVIEW_PATH,
        help="Edited review CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CURATED_PUBLIC_POSTS_PATH,
        help="Curated JSONL output.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=CURATION_MIN_ROWS,
        help="Minimum total curated rows required.",
    )
    parser.add_argument(
        "--min-linkedin-rows",
        type=int,
        default=CURATION_MIN_LINKEDIN_ROWS,
        help="Minimum LinkedIn rows required.",
    )
    parser.add_argument(
        "--allow-low-counts",
        action="store_true",
        help="Write output even if the curated set is below the configured floors.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_review_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def apply_review(records: list[dict[str, Any]], review_map: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    curated: list[dict[str, Any]] = []
    missing = [record["id"] for record in records if str(record["id"]) not in review_map]
    if missing:
        raise ValueError(f"Review sheet is missing {len(missing)} ids, first missing: {missing[0]}")

    for record in records:
        review = review_map[str(record["id"])]
        keep = review.get("keep", "").strip().lower() in TRUTHY
        drop_reason = review.get("drop_reason", "").strip()
        if keep and drop_reason:
            raise ValueError(f"Review row {record['id']} is marked keep=true but has drop_reason={drop_reason!r}")
        if not keep:
            continue
        curated.append({**record, "voice_phase": review.get("voice_phase", "")})
    curated.sort(key=lambda row: (str(row["created_at"]), str(row["id"])))
    return curated


def validate_counts(rows: list[dict[str, Any]], min_rows: int, min_linkedin_rows: int) -> None:
    platform_counts = Counter(str(row["platform"]) for row in rows)
    if len(rows) < min_rows:
        raise ValueError(f"Curated dataset has {len(rows)} rows, below required minimum {min_rows}")
    if platform_counts.get("linkedin", 0) < min_linkedin_rows:
        raise ValueError(
            "Curated dataset has "
            f"{platform_counts.get('linkedin', 0)} LinkedIn rows, below required minimum {min_linkedin_rows}"
        )


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    review_map = read_review_csv(args.review)
    curated = apply_review(records, review_map)
    if not args.allow_low_counts:
        validate_counts(curated, args.min_rows, args.min_linkedin_rows)
    write_jsonl(args.output, curated)
    platform_counts = Counter(str(row["platform"]) for row in curated)
    print(f"Curated rows: {len(curated)}")
    print(f"Platform counts: {dict(platform_counts)}")
    print(f"Curated JSONL: {args.output}")


if __name__ == "__main__":
    main()
