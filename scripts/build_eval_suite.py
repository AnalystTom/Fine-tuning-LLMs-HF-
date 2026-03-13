"""Build the fixed run-1 evaluation suite for baseline vs fine-tuned comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .experiment_config import EVAL_SYSTEM_PROMPT, RUN1_EVAL_SUITE_PATH, RUN1_PROMPT_SPECS
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import EVAL_SYSTEM_PROMPT, RUN1_EVAL_SUITE_PATH, RUN1_PROMPT_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN1_EVAL_SUITE_PATH,
        help="JSONL output path for the full prompt suite.",
    )
    return parser.parse_args()


def render_user_prompt(prompt_text: str, constraints: tuple[str, ...]) -> str:
    constraint_lines = "\n".join(f"- {constraint}" for constraint in constraints)
    return f"{prompt_text}\n\nConstraints:\n{constraint_lines}"


def build_eval_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in RUN1_PROMPT_SPECS:
        for seed in spec.seeds:
            rows.append(
                {
                    "suite_name": spec.suite_name,
                    "prompt_type": "run1_eval",
                    "bucket": spec.bucket,
                    "prompt_id": spec.prompt_id,
                    "seed": seed,
                    "platform": spec.platform,
                    "system_prompt": EVAL_SYSTEM_PROMPT,
                    "prompt_text": spec.prompt_text,
                    "constraints": list(spec.constraints),
                    "max_tokens": spec.max_tokens,
                    "messages": [
                        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": render_user_prompt(spec.prompt_text, spec.constraints),
                        },
                    ],
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    rows = build_eval_rows()
    write_jsonl(args.output, rows)
    print(f"Eval suite rows: {len(rows)}")
    print(f"Eval suite path: {args.output}")


if __name__ == "__main__":
    main()
