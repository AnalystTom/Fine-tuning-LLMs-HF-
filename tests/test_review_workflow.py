from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.apply_review_sheet import apply_review
from scripts.export_review_sheet import build_review_rows, classify_post


def test_classify_post_drops_job_search_content() -> None:
    keep, drop_reason, phase = classify_post(
        {
            "created_at": "2024-03-12T12:46:18Z",
            "char_len": 250,
            "text": "Hi everyone - I am looking for a data analyst role and would appreciate your support.",
        }
    )
    assert not keep
    assert drop_reason == "job_search"
    assert phase == "legacy"


def test_classify_post_keeps_current_ai_builder_content() -> None:
    keep, drop_reason, phase = classify_post(
        {
            "created_at": "2026-02-08T12:15:59Z",
            "char_len": 800,
            "text": "Agents are infrastructure now. I have been building agent workflows and evaluation systems all month.",
        }
    )
    assert keep
    assert drop_reason == ""
    assert phase == "current_ai_builder"


def test_build_review_rows_preserves_required_columns() -> None:
    rows = build_review_rows(
        [
            {
                "id": "x1",
                "platform": "x",
                "source_type": "tweet_original",
                "created_at": "2026-01-01T00:00:00Z",
                "char_len": 120,
                "length_bucket": "short",
                "text": "A useful AI systems post.",
            }
        ]
    )
    assert rows[0]["id"] == "x1"
    assert rows[0]["platform"] == "x"
    assert rows[0]["keep"] in {"true", "false"}
    assert rows[0]["voice_phase"]


def test_apply_review_keeps_only_true_without_drop_reason(tmp_path: Path) -> None:
    records = [
        {
            "id": "keep_me",
            "platform": "linkedin",
            "source_type": "linkedin_share",
            "created_at": "2026-01-01T00:00:00Z",
            "text": "Keep this.",
            "char_len": 10,
            "length_bucket": "short",
        },
        {
            "id": "drop_me",
            "platform": "x",
            "source_type": "tweet_original",
            "created_at": "2026-01-02T00:00:00Z",
            "text": "Drop this.",
            "char_len": 10,
            "length_bucket": "short",
        },
    ]
    review_map = {
        "keep_me": {
            "id": "keep_me",
            "keep": "true",
            "drop_reason": "",
            "voice_phase": "current_ai_builder",
        },
        "drop_me": {
            "id": "drop_me",
            "keep": "false",
            "drop_reason": "other",
            "voice_phase": "legacy",
        },
    }
    curated = apply_review(records, review_map)
    assert [row["id"] for row in curated] == ["keep_me"]
    assert curated[0]["voice_phase"] == "current_ai_builder"
