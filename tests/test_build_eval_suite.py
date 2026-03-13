from __future__ import annotations

from collections import Counter

from scripts.build_eval_suite import build_eval_rows


def test_build_eval_rows_creates_20x3_suite() -> None:
    rows = build_eval_rows()
    assert len(rows) == 60
    bucket_counts = Counter(row["bucket"] for row in rows)
    assert bucket_counts == {
        "x_topic": 15,
        "blog": 15,
        "various_topic": 15,
        "response": 15,
    }


def test_build_eval_rows_repeat_each_prompt_for_three_seeds() -> None:
    rows = build_eval_rows()
    prompt_counts = Counter(row["prompt_id"] for row in rows)
    assert set(prompt_counts.values()) == {3}
    seeds = {row["seed"] for row in rows}
    assert seeds == {3407, 3408, 3409}
