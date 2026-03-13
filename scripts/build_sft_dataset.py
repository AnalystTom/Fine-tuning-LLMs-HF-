"""Label curated public posts and build train/eval SFT datasets for run 1."""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

try:
    from .experiment_config import (
        ASSISTANT_SYSTEM_PROMPT,
        CURATED_PUBLIC_POSTS_PATH,
        FIXED_PROMPTS,
        POST_TYPES,
        RUN1_EVAL_PATH,
        RUN1_EVAL_TEXT_PATH,
        RUN1_LABELED_POSTS_PATH,
        RUN1_RECONSTRUCTION_PROMPTS_PATH,
        RUN1_TRAIN_PATH,
    )
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import (
        ASSISTANT_SYSTEM_PROMPT,
        CURATED_PUBLIC_POSTS_PATH,
        FIXED_PROMPTS,
        POST_TYPES,
        RUN1_EVAL_PATH,
        RUN1_EVAL_TEXT_PATH,
        RUN1_LABELED_POSTS_PATH,
        RUN1_RECONSTRUCTION_PROMPTS_PATH,
        RUN1_TRAIN_PATH,
    )


STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "up",
    "us",
    "was",
    "we",
    "what",
    "when",
    "with",
    "you",
    "your",
}
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=CURATED_PUBLIC_POSTS_PATH,
        help="Curated public-post JSONL from apply_review_sheet.py.",
    )
    parser.add_argument(
        "--labeled-output",
        type=Path,
        default=RUN1_LABELED_POSTS_PATH,
        help="JSONL output for labeled public posts.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=RUN1_TRAIN_PATH,
        help="JSONL output for train SFT examples.",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=RUN1_EVAL_PATH,
        help="JSONL output for eval SFT examples.",
    )
    parser.add_argument(
        "--reconstruction-prompts-output",
        type=Path,
        default=RUN1_RECONSTRUCTION_PROMPTS_PATH,
        help="JSONL output for held-out reconstruction prompts.",
    )
    parser.add_argument(
        "--eval-text-output",
        type=Path,
        default=RUN1_EVAL_TEXT_PATH,
        help="Plain-text output for llama-perplexity.",
    )
    parser.add_argument(
        "--labeler-backend",
        choices=("heuristic", "openai"),
        default="heuristic",
        help="Use heuristics or an OpenAI-compatible endpoint for dataset labels.",
    )
    parser.add_argument(
        "--labeler-endpoint",
        default="http://127.0.0.1:8080/v1/chat/completions",
        help="OpenAI-compatible chat completion endpoint for label generation.",
    )
    parser.add_argument(
        "--labeler-model",
        default="labeler",
        help="Model identifier sent to the OpenAI-compatible endpoint.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def extract_keywords(text: str) -> list[str]:
    tokens = [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]
    unique: list[str] = []
    for token in tokens:
        if token in {"url", "handle"} or token.isdigit() or len(token) < 3:
            continue
        if token not in unique:
            unique.append(token)
        if len(unique) == 6:
            break
    return unique


def infer_topic(text: str) -> str:
    lowered = text.lower()
    keywords = extract_keywords(text)
    keyword_set = set(keywords)
    if {"agent", "agents", "multi", "delegation"} & keyword_set:
        return "multi agent workflows"
    if {"model", "models", "llm", "llms", "eval", "evaluation"} & keyword_set:
        return "llm evaluation tradeoffs"
    if {"debugging", "debug", "coding", "code", "claude"} & keyword_set:
        return "ai coding tradeoffs"
    if {"job", "jobs", "layoff", "career"} & keyword_set:
        return "ai and changing work"
    if {"startup", "saas", "product", "ship", "shipping", "build", "building"} & keyword_set:
        return "shipping ai products"
    if {"telegram", "workflow", "automation", "productivity"} & keyword_set:
        return "ai workflow automation"
    if {"data", "inference", "infrastructure", "cloud", "context"} & keyword_set:
        return "ai infrastructure lessons"
    if "?" in lowered:
        return "builder feedback request"
    if keywords:
        return " ".join(keywords[: min(5, max(3, len(keywords)))])
    return "ai product observations"


def infer_post_type(text: str) -> str:
    lowered = text.lower()
    if "?" in text or lowered.startswith(("what ", "how ", "would ", "should ", "can ")):
        return "question"
    if "<URL>" in text or any(
        phrase in lowered
        for phrase in ("guide", "paper", "resource", "read this", "check out", "article", "video", "book")
    ):
        return "resource_share"
    if any(
        phrase in lowered
        for phrase in (
            "i built",
            "i've been building",
            "we built",
            "we're building",
            "shipped",
            "shipping",
            "working on",
            "prototype",
            "side project",
            "hackathon",
            "today i've",
            "daily updates",
        )
    ):
        return "build_update"
    if any(
        phrase in lowered
        for phrase in ("i think", "i believe", "should", "feels", "opinion", "overrated", "underrated")
    ):
        return "opinion"
    return "observation"


def label_prompt(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You label social posts for a fine-tuning dataset.",
        },
        {
            "role": "user",
            "content": (
                "Return strict JSON with keys topic, post_type, length_bucket.\n"
                "Topic must be 3-8 words and abstract, not a copy of the post.\n"
                "post_type must be one of: build_update, opinion, observation, question, resource_share.\n"
                "length_bucket must be short, medium, or long.\n"
                f'Post:\n"""{text}"""'
            ),
        },
    ]


def call_openai_compatible_labeler(
    endpoint: str, model: str, text: str, length_bucket: str
) -> dict[str, str]:
    payload = {
        "model": model,
        "messages": label_prompt(text),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:  # pragma: no cover - depends on local server
        raise RuntimeError(f"Failed to call labeler endpoint {endpoint}: {exc}") from exc

    content = data["choices"][0]["message"]["content"]
    label = json.loads(content)
    topic = str(label.get("topic", "")).strip() or infer_topic(text)
    post_type = str(label.get("post_type", "")).strip()
    if post_type not in POST_TYPES:
        post_type = infer_post_type(text)
    api_length_bucket = str(label.get("length_bucket", "")).strip()
    return {
        "topic": topic,
        "post_type": post_type,
        "length_bucket": api_length_bucket if api_length_bucket in {"short", "medium", "long"} else length_bucket,
    }


def label_post(record: dict[str, Any], backend: str, endpoint: str, model: str) -> dict[str, str]:
    text = str(record["text"])
    existing_length_bucket = str(record["length_bucket"])
    if backend == "openai":
        return call_openai_compatible_labeler(endpoint, model, text, existing_length_bucket)
    return {
        "topic": infer_topic(text),
        "post_type": infer_post_type(text),
        "length_bucket": existing_length_bucket,
    }


def build_user_prompt(platform: str, topic: str, post_type: str, length_bucket: str) -> str:
    return (
        f"Write a {platform} post.\n"
        f"Topic: {topic}\n"
        f"Post type: {post_type}\n"
        f"Length: {length_bucket}\n"
        "Constraints: public-facing, original wording, no copied handles or links."
    )


def build_sft_row(record: dict[str, Any], label: dict[str, str]) -> dict[str, Any]:
    platform = str(record["platform"])
    topic = label["topic"]
    post_type = label["post_type"]
    length_bucket = label["length_bucket"]
    user_prompt = build_user_prompt(platform, topic, post_type, length_bucket)
    return {
        "id": record["id"],
        "platform": platform,
        "source_type": record["source_type"],
        "created_at": record["created_at"],
        "topic": topic,
        "post_type": post_type,
        "length_bucket": length_bucket,
        "messages": [
            {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": record["text"]},
        ],
    }


def split_records_chronologically(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_type"])] += [row]

    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for items in grouped.values():
        items.sort(key=lambda row: (str(row["created_at"]), str(row["id"])))
        if len(items) <= 1:
            train.extend(items)
            continue
        split_index = math.floor(len(items) * 0.85)
        split_index = max(1, min(split_index, len(items) - 1))
        train.extend(items[:split_index])
        eval_rows.extend(items[split_index:])
    train.sort(key=lambda row: (str(row["created_at"]), str(row["id"])))
    eval_rows.sort(key=lambda row: (str(row["created_at"]), str(row["id"])))
    return train, eval_rows


def build_reconstruction_prompt_rows(eval_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for row in eval_rows:
        user_prompt = build_user_prompt(
            platform=str(row["platform"]),
            topic=str(row["topic"]),
            post_type=str(row["post_type"]),
            length_bucket=str(row["length_bucket"]),
        )
        prompts.append(
            {
                "suite_name": "run1_reconstruction",
                "prompt_type": "held_out_reconstruction",
                "bucket": "reconstruction",
                "prompt_id": f"reconstruct_{row['id']}",
                "seed": 3407,
                "platform": row["platform"],
                "prompt_text": user_prompt,
                "reference_text": row["messages"][2]["content"],
                "source_id": row["id"],
                "max_tokens": 180 if row["platform"] == "x" else 420,
                "messages": [
                    {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
        )
    return prompts


def build_fixed_prompt_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt in FIXED_PROMPTS:
        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "prompt_type": "fixed_suite",
                "platform": prompt.platform,
                "prompt_text": prompt.prompt_text,
                "max_tokens": prompt.max_tokens,
                "messages": [
                    {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt.prompt_text},
                ],
            }
        )
    return rows


def build_eval_text(eval_rows: Sequence[dict[str, Any]]) -> str:
    return "\n\n<|end_of_sample|>\n\n".join(
        str(row["messages"][2]["content"]).strip() for row in eval_rows
    )


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    labeled_rows: list[dict[str, Any]] = []
    for record in records:
        label = label_post(
            record,
            args.labeler_backend,
            args.labeler_endpoint,
            args.labeler_model,
        )
        labeled_rows.append({**record, **label})

    sft_rows = [build_sft_row(record, record) for record in labeled_rows]
    train_rows, eval_rows = split_records_chronologically(sft_rows)
    reconstruction_rows = build_reconstruction_prompt_rows(eval_rows)
    eval_text = build_eval_text(eval_rows)

    write_jsonl(args.labeled_output, labeled_rows)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    write_jsonl(args.reconstruction_prompts_output, reconstruction_rows)
    args.eval_text_output.parent.mkdir(parents=True, exist_ok=True)
    args.eval_text_output.write_text(
        eval_text + ("\n" if eval_text else ""),
        encoding="utf-8",
    )

    print(f"Labeled rows: {len(labeled_rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Eval rows: {len(eval_rows)}")
    print(f"Reconstruction prompts: {len(reconstruction_rows)}")


if __name__ == "__main__":
    main()
