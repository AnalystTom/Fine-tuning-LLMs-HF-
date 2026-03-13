from __future__ import annotations

from scripts.build_sft_dataset import (
    build_reconstruction_prompt_rows,
    infer_post_type,
    infer_topic,
    split_records_chronologically,
)


def test_infer_topic_prefers_abstract_categories() -> None:
    topic = infer_topic("We shipped a new agent workflow and the evaluation loop finally feels reliable.")
    assert topic == "multi agent workflows"


def test_infer_post_type_uses_question_before_other_categories() -> None:
    assert infer_post_type("How are you validating agentic products before shipping?") == "question"


def test_split_records_chronologically_keeps_latest_per_source_in_eval() -> None:
    rows = [
        {"id": "a", "created_at": "2026-01-01T00:00:00Z", "source_type": "tweet_original"},
        {"id": "b", "created_at": "2026-01-02T00:00:00Z", "source_type": "tweet_original"},
        {"id": "c", "created_at": "2026-01-03T00:00:00Z", "source_type": "tweet_original"},
        {"id": "d", "created_at": "2026-01-01T00:00:00Z", "source_type": "linkedin_share"},
        {"id": "e", "created_at": "2026-01-02T00:00:00Z", "source_type": "linkedin_share"},
    ]
    train, eval_rows = split_records_chronologically(rows)
    assert [row["id"] for row in train] == ["a", "d", "b"]
    assert [row["id"] for row in eval_rows] == ["e", "c"]


def test_reconstruction_prompts_include_reference_text() -> None:
    rows = [
        {
            "id": "x1",
            "platform": "x",
            "topic": "ai coding tradeoffs",
            "post_type": "observation",
            "length_bucket": "medium",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "reference output"},
            ],
        }
    ]
    prompts = build_reconstruction_prompt_rows(rows)
    assert prompts[0]["prompt_type"] == "held_out_reconstruction"
    assert prompts[0]["reference_text"] == "reference output"
