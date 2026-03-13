"""Render a run-1 experiment summary and blog-oriented notes from available artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .experiment_config import (
        CURATED_PUBLIC_POSTS_PATH,
        RUN1_BLIND_EVAL_SUMMARY_PATH,
        RUN1_MEMORIZATION_REPORT_PATH,
        RUN1_PERPLEXITY_BASELINE_PATH,
        RUN1_PERPLEXITY_FINETUNED_PATH,
        RUN1_SUMMARY_PATH,
        RUN1_TRAIN_PATH,
        RUN1_EVAL_PATH,
        RUN1_EVAL_DIR,
    )
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import (
        CURATED_PUBLIC_POSTS_PATH,
        RUN1_BLIND_EVAL_SUMMARY_PATH,
        RUN1_MEMORIZATION_REPORT_PATH,
        RUN1_PERPLEXITY_BASELINE_PATH,
        RUN1_PERPLEXITY_FINETUNED_PATH,
        RUN1_SUMMARY_PATH,
        RUN1_TRAIN_PATH,
        RUN1_EVAL_PATH,
        RUN1_EVAL_DIR,
    )


BLOG_NOTES_PATH = RUN1_EVAL_DIR / "blog_post_notes.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", type=Path, default=CURATED_PUBLIC_POSTS_PATH)
    parser.add_argument("--train", type=Path, default=RUN1_TRAIN_PATH)
    parser.add_argument("--eval", type=Path, default=RUN1_EVAL_PATH)
    parser.add_argument("--memorization", type=Path, default=RUN1_MEMORIZATION_REPORT_PATH)
    parser.add_argument("--perplexity-baseline", type=Path, default=RUN1_PERPLEXITY_BASELINE_PATH)
    parser.add_argument("--perplexity-finetuned", type=Path, default=RUN1_PERPLEXITY_FINETUNED_PATH)
    parser.add_argument("--human-summary", type=Path, default=RUN1_BLIND_EVAL_SUMMARY_PATH)
    parser.add_argument("--summary-output", type=Path, default=RUN1_SUMMARY_PATH)
    parser.add_argument("--blog-notes-output", type=Path, default=BLOG_NOTES_PATH)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_first_float(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", text)
    return float(match.group(1)) if match else None


def read_memo_flags(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, int] = {}
    for item in payload.get("generated_files", []):
        result[Path(item["path"]).name] = int(item.get("flagged_count", 0))
    return result


def dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "platform_counts": Counter(str(row.get("platform", "unknown")) for row in rows),
        "source_counts": Counter(str(row.get("source_type", "unknown")) for row in rows),
    }


def render_summary(args: argparse.Namespace) -> tuple[str, str]:
    curated = read_jsonl(args.curated)
    train = read_jsonl(args.train)
    eval_rows = read_jsonl(args.eval)
    curated_stats = dataset_stats(curated)
    train_stats = dataset_stats(train)
    eval_stats = dataset_stats(eval_rows)
    memo_flags = read_memo_flags(args.memorization)
    ppl_base = parse_first_float(args.perplexity_baseline)
    ppl_ft = parse_first_float(args.perplexity_finetuned)
    human_summary_text = args.human_summary.read_text(encoding="utf-8") if args.human_summary.exists() else None

    lines = ["# Run 1 Summary", ""]
    lines.extend([
        "## Dataset",
        "",
        f"- Curated rows: `{curated_stats['rows']}`",
        f"- Train rows: `{train_stats['rows']}`",
        f"- Eval rows: `{eval_stats['rows']}`",
        f"- Curated platform mix: `{dict(curated_stats['platform_counts'])}`",
        f"- Curated source mix: `{dict(curated_stats['source_counts'])}`",
        "",
        "## Automated Checks",
        "",
    ])
    if ppl_base is not None and ppl_ft is not None:
        lines.append(f"- Baseline perplexity: `{ppl_base}`")
        lines.append(f"- Fine-tuned perplexity: `{ppl_ft}`")
        lines.append(f"- Perplexity delta (fine-tuned - baseline): `{ppl_ft - ppl_base:.4f}`")
    else:
        lines.append("- Perplexity: pending")
    if memo_flags:
        for name, count in sorted(memo_flags.items()):
            lines.append(f"- Memorization flags `{name}`: `{count}`")
    else:
        lines.append("- Memorization: pending")
    lines.append("")
    lines.append("## Human Review")
    lines.append("")
    if human_summary_text:
        lines.append("See `blind_eval_summary.md` for the scored review packet.")
    else:
        lines.append("- Human review: pending")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- Promote only if social buckets improve, memorization does not regress, and placeholder leakage stays zero.")
    lines.append("")

    blog_lines = ["# Blog Post Notes", ""]
    blog_lines.extend([
        "## Proposed Headline",
        "",
        "What happened when I fine-tuned Qwen3.5-9B on my public X and LinkedIn posts",
        "",
        "## Core Narrative",
        "",
        "- Goal: see whether a small, controlled QLoRA fine-tune can shift a strong base model toward my public AI-builder voice.",
        f"- Dataset: {curated_stats['rows']} curated public posts, with {curated_stats['platform_counts'].get('x', 0)} X posts and {curated_stats['platform_counts'].get('linkedin', 0)} LinkedIn posts.",
        f"- Training split: {train_stats['rows']} train / {eval_stats['rows']} eval.",
        "- Evaluation: fixed 20-prompt suite x 3 seeds, held-out reconstruction, perplexity, memorization, and blind human review.",
        "",
        "## Sections To Write",
        "",
        "1. Why I chose posts-only first instead of mixing in replies.",
        "2. How I curated the dataset to reflect my current AI-builder voice instead of my whole historical feed.",
        "3. Why I used the same llama.cpp runtime for baseline and fine-tuned models.",
        "4. What changed in X posts, responses, and blog-style writing.",
        "5. What the evals showed and what they missed.",
        "6. What I would change for run 2.",
        "",
        "## Numbers To Drop In",
        "",
        f"- Curated dataset rows: {curated_stats['rows']}",
        f"- X rows: {curated_stats['platform_counts'].get('x', 0)}",
        f"- LinkedIn rows: {curated_stats['platform_counts'].get('linkedin', 0)}",
        f"- Train rows: {train_stats['rows']}",
        f"- Eval rows: {eval_stats['rows']}",
        f"- Perplexity baseline: {ppl_base if ppl_base is not None else 'pending'}",
        f"- Perplexity fine-tuned: {ppl_ft if ppl_ft is not None else 'pending'}",
        f"- Memorization flags: {memo_flags if memo_flags else 'pending'}",
        "",
        "## Honest Caveats",
        "",
        "- The dataset is still X-heavy, so the adapter is biased toward short-form voice.",
        "- Blogs are evaluation probes, not the training target.",
        "- Offline engagement scoring is a proxy, not real audience feedback.",
        "- If replies underperform, run 2 should add only curated high-signal replies/comments, not the full reply pool.",
        "",
    ])
    return "\n".join(lines).rstrip() + "\n", "\n".join(blog_lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    summary_text, blog_notes = render_summary(args)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(summary_text, encoding="utf-8")
    args.blog_notes_output.parent.mkdir(parents=True, exist_ok=True)
    args.blog_notes_output.write_text(blog_notes, encoding="utf-8")
    print(f"Wrote run summary to {args.summary_output}")
    print(f"Wrote blog notes to {args.blog_notes_output}")


if __name__ == "__main__":
    main()
