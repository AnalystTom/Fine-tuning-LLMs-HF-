from __future__ import annotations

from scripts.build_blind_review_sheet import build_blind_rows
from scripts.score_human_eval import score_review_row, score_reviews


def test_build_blind_rows_randomizes_but_preserves_pairing() -> None:
    baseline = [
        {
            "model_label": "baseline_qwen35_9b",
            "bucket": "x_topic",
            "prompt_id": "x_parallel_agents",
            "seed": 3407,
            "platform": "x",
            "prompt_text": "Prompt",
            "response_text": "Baseline text",
        }
    ]
    finetuned = [
        {
            "model_label": "finetuned_qwen35_9b",
            "bucket": "x_topic",
            "prompt_id": "x_parallel_agents",
            "seed": 3407,
            "platform": "x",
            "prompt_text": "Prompt",
            "response_text": "Fine-tuned text",
        }
    ]
    review_rows, key_rows, template_rows = build_blind_rows(baseline, finetuned, 3407)
    assert review_rows[0]["review_id"] == "x_parallel_agents__3407"
    assert {review_rows[0]["output_a"], review_rows[0]["output_b"]} == {
        "Baseline text",
        "Fine-tuned text",
    }
    assert key_rows[0]["model_a"] != key_rows[0]["model_b"]
    assert template_rows[0]["winner"] == ""


def test_score_review_row_uses_primary_metric_tiebreak() -> None:
    row = {
        "review_id": "resp_debugging_work__3407",
        "bucket": "response",
        "a_authenticity": "4",
        "a_conversational_value": "5",
        "a_judgment": "4",
        "a_specificity": "4",
        "a_platform_fit": "3",
        "a_readability": "4",
        "b_authenticity": "5",
        "b_conversational_value": "4",
        "b_judgment": "4",
        "b_specificity": "4",
        "b_platform_fit": "4",
        "b_readability": "3",
        "notes": "",
    }
    scored = score_review_row(row)
    assert scored["winner"] == "A"


def test_score_reviews_computes_weighted_finetuned_rate() -> None:
    rows = [
        {
            "review_id": "x_parallel_agents__3407",
            "bucket": "x_topic",
            "a_authenticity": "5",
            "a_engagement_likelihood": "5",
            "a_specificity": "5",
            "a_platform_fit": "5",
            "a_readability": "5",
            "a_originality": "5",
            "b_authenticity": "3",
            "b_engagement_likelihood": "3",
            "b_specificity": "3",
            "b_platform_fit": "3",
            "b_readability": "3",
            "b_originality": "3",
            "notes": "",
        }
    ]
    key_rows = [
        {
            "review_id": "x_parallel_agents__3407",
            "model_a": "finetuned_qwen35_9b",
            "model_b": "baseline_qwen35_9b",
        }
    ]
    summary = score_reviews(rows, key_rows)
    assert summary["model_wins"]["finetuned_qwen35_9b"] == 1
    assert summary["weighted_finetuned_win_rate"] == 1.0
