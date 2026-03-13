"""Create an editable review sheet for curating run-1 public posts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .experiment_config import (
        CURATION_RECENCY_CUTOFF,
        CURATION_TRANSITION_CUTOFF,
        PUBLIC_POSTS_PATH,
        PUBLIC_POSTS_REVIEW_PATH,
    )
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import (
        CURATION_RECENCY_CUTOFF,
        CURATION_TRANSITION_CUTOFF,
        PUBLIC_POSTS_PATH,
        PUBLIC_POSTS_REVIEW_PATH,
    )


AI_BUILDER_KEYWORDS = {
    "agent",
    "agents",
    "agentic",
    "ai",
    "automation",
    "benchmark",
    "build",
    "builder",
    "builders",
    "building",
    "claude",
    "clawbench",
    "clawdbot",
    "context",
    "copilot",
    "data",
    "deployment",
    "eval",
    "evaluation",
    "gpt",
    "inference",
    "llama",
    "llm",
    "llms",
    "mcp",
    "model",
    "models",
    "nl2sql",
    "notebooklm",
    "hackathon",
    "synthetic",
    "openai",
    "product",
    "products",
    "prompt",
    "rag",
    "reasoning",
    "ship",
    "shipping",
    "sql",
    "system",
    "systems",
    "workflow",
    "workflows",
}

BUILD_SIGNAL_PHRASES = (
    "i built",
    "we built",
    "building",
    "shipped",
    "shipping",
    "working on",
    "daily updates",
    "productivity boost",
    "what changed",
    "my take",
    "i've been doing",
    "i've been building",
    "today i've",
    "trying myself out",
    "here is a huge productivity boost",
)

JOB_SEARCH_PHRASES = (
    "looking for a data analyst role",
    "looking for opportunities",
    "would appreciate your support",
    "my previous places of employment",
)

COURSE_PHRASES = (
    "course completion",
    "certificate",
    "just completed",
    "google tag fundamentals",
    "taking my exam",
)

EMPLOYMENT_PHRASES = (
    "starting a new position",
    "happy to share that i'm starting",
    "happy to share that i’m starting",
)

EVENT_PHRASES = (
    "attended",
    "attending",
    "event",
    "summit",
    "conference",
    "festival",
    "meetup",
    "podcast",
    "world fair",
    "great evening",
    "great day",
    "thanks",
    "give me a shout",
)

GENERIC_RESOURCE_PHRASES = (
    "check out",
    "sharing for wider reach",
    "reading list",
    "best book out there",
    "this webinar looks great",
    "great team and highly driven",
    "must read",
    "kaggle profile",
)

OFF_TOPIC_PERSONAL_PHRASES = (
    "#mma",
    "ufc",
    "boxing",
    "fight",
    "followers",
    "x famous",
    "honeycomb",
    "deep state operative",
    "market will survive this",
    "molty",
)

HIGH_SIGNAL_EXCEPTION_PHRASES = (
    "hackathon",
    "rag",
    "llm",
    "agent",
    "agents",
    "reasoning",
    "notebooklm",
    "hackathon",
    "synthetic",
    "open claw",
    "clawbench",
    "clawdbot",
    "claude code",
    "prompt engineering",
    "synthetic data",
)

REVIEW_COLUMNS = (
    "id",
    "platform",
    "source_type",
    "created_at",
    "char_len",
    "length_bucket",
    "text",
    "keep",
    "drop_reason",
    "voice_phase",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PUBLIC_POSTS_PATH,
        help="Source public post JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PUBLIC_POSTS_REVIEW_PATH,
        help="Editable CSV output path.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def keyword_score(text: str) -> int:
    lowered = text.lower()
    score = 0
    for keyword in AI_BUILDER_KEYWORDS:
        if keyword in lowered:
            score += 1
    return score


def has_build_signal(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in BUILD_SIGNAL_PHRASES)


def voice_phase(record: dict[str, Any]) -> str:
    created_at = str(record["created_at"])
    text = str(record["text"])
    score = keyword_score(text)
    if created_at >= CURATION_RECENCY_CUTOFF and (score >= 1 or has_build_signal(text)):
        return "current_ai_builder"
    if created_at >= CURATION_TRANSITION_CUTOFF and (score >= 2 or has_build_signal(text)):
        return "transition_ai_builder"
    return "legacy"


def is_event_only(text: str, char_len: int, score: int) -> bool:
    lowered = text.lower()
    if not any(phrase in lowered for phrase in EVENT_PHRASES):
        return False
    if any(phrase in lowered for phrase in HIGH_SIGNAL_EXCEPTION_PHRASES) and char_len >= 220:
        return False
    if score >= 4 and char_len >= 320:
        return False
    if any(token in lowered for token in ("here's", "here is", "what changed", "what stood out", "my take", "i liked", "my favourite", "deep dive", "looking forward to building", "planning to explore", "visibility in the ai age", "highly recommend anyone interested in ai")):
        return False
    if char_len >= 180 and score >= 2:
        return False
    return True


def is_generic_resource_share(text: str, char_len: int, score: int) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in GENERIC_RESOURCE_PHRASES) and char_len < 260 and score < 4 and not has_build_signal(text):
        return True
    if "planning to explore" in lowered and score >= 1:
        return False
    if "<url>" in lowered and char_len < 220 and score < 3 and not has_build_signal(text):
        return True
    return False


def classify_post(record: dict[str, Any]) -> tuple[bool, str, str]:
    text = str(record["text"])
    lowered = text.lower()
    char_len = int(record["char_len"])
    phase = voice_phase(record)
    score = keyword_score(text)

    if any(phrase in lowered for phrase in JOB_SEARCH_PHRASES):
        return False, "job_search", phase
    if any(phrase in lowered for phrase in COURSE_PHRASES):
        return False, "course_or_certificate", phase
    if any(phrase in lowered for phrase in EMPLOYMENT_PHRASES):
        return False, "employment_announcement", phase
    if any(phrase in lowered for phrase in OFF_TOPIC_PERSONAL_PHRASES):
        return False, "off_topic_personal", phase
    if char_len < 50 and score < 2 and not has_build_signal(text):
        return False, "low_signal_short", phase
    if is_event_only(text, char_len, score):
        return False, "event_only", phase
    if is_generic_resource_share(text, char_len, score):
        return False, "generic_resource_share", phase
    if phase == "legacy" and char_len >= 180 and any(token in lowered for token in ("highly recommend anyone interested in ai", "practice and think about it daily")):
        return True, "", phase
    if phase == "legacy" and score < 3 and not has_build_signal(text):
        return False, "non_current_voice", phase
    return True, "", phase


def build_review_rows(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in sorted(records, key=lambda row: (str(row["created_at"]), str(row["id"]))):
        keep, drop_reason, phase = classify_post(record)
        rows.append(
            {
                "id": str(record["id"]),
                "platform": str(record["platform"]),
                "source_type": str(record["source_type"]),
                "created_at": str(record["created_at"]),
                "char_len": str(record["char_len"]),
                "length_bucket": str(record["length_bucket"]),
                "text": str(record["text"]),
                "keep": "true" if keep else "false",
                "drop_reason": drop_reason,
                "voice_phase": phase,
            }
        )
    return rows


def write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_review_rows(read_jsonl(args.input))
    write_review_csv(args.output, rows)
    kept = sum(1 for row in rows if row["keep"] == "true")
    print(f"Review rows: {len(rows)}")
    print(f"Suggested keep rows: {kept}")
    print(f"Review CSV: {args.output}")


if __name__ == "__main__":
    main()
