"""Run the full post-train evaluation workflow for run 1."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

try:
    from .experiment_config import (
        RUN1_BASELINE_OUTPUTS_PATH,
        RUN1_BLIND_REVIEW_PATH,
        RUN1_EVAL_SUITE_PATH,
        RUN1_FINETUNED_OUTPUTS_PATH,
        RUN1_MEMORIZATION_REPORT_PATH,
        RUN1_PERPLEXITY_BASELINE_PATH,
        RUN1_PERPLEXITY_FINETUNED_PATH,
        RUN1_RECONSTRUCTION_BASELINE_PATH,
        RUN1_RECONSTRUCTION_FINETUNED_PATH,
        RUN1_RECONSTRUCTION_PROMPTS_PATH,
        RUN1_SUMMARY_PATH,
    )
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import (
        RUN1_BASELINE_OUTPUTS_PATH,
        RUN1_BLIND_REVIEW_PATH,
        RUN1_EVAL_SUITE_PATH,
        RUN1_FINETUNED_OUTPUTS_PATH,
        RUN1_MEMORIZATION_REPORT_PATH,
        RUN1_PERPLEXITY_BASELINE_PATH,
        RUN1_PERPLEXITY_FINETUNED_PATH,
        RUN1_RECONSTRUCTION_BASELINE_PATH,
        RUN1_RECONSTRUCTION_FINETUNED_PATH,
        RUN1_RECONSTRUCTION_PROMPTS_PATH,
        RUN1_SUMMARY_PATH,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-gguf", type=Path, required=True)
    parser.add_argument("--finetuned-gguf", type=Path, required=True)
    parser.add_argument("--llama-server-bin", default="llama-server")
    parser.add_argument("--llama-perplexity-bin", default="llama-perplexity")
    parser.add_argument("--context-size", type=int, default=8192)
    return parser.parse_args()


def run(cmd: list[str], *, stdout_path: Path | None = None) -> None:
    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    if stdout_path is None:
        subprocess.run(cmd, check=True)
        return
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle)


def main() -> None:
    args = parse_args()

    run(
        [
            "python3",
            "scripts/run_llamacpp_suite.py",
            "--prompts",
            str(RUN1_EVAL_SUITE_PATH),
            "--baseline-gguf",
            str(args.baseline_gguf),
            "--finetuned-gguf",
            str(args.finetuned_gguf),
            "--baseline-output",
            str(RUN1_BASELINE_OUTPUTS_PATH),
            "--finetuned-output",
            str(RUN1_FINETUNED_OUTPUTS_PATH),
            "--summary-output",
            str(RUN1_SUMMARY_PATH),
            "--llama-server-bin",
            args.llama_server_bin,
            "--context-size",
            str(args.context_size),
        ]
    )

    run(
        [
            "python3",
            "scripts/run_llamacpp_suite.py",
            "--prompts",
            str(RUN1_RECONSTRUCTION_PROMPTS_PATH),
            "--baseline-gguf",
            str(args.baseline_gguf),
            "--finetuned-gguf",
            str(args.finetuned_gguf),
            "--baseline-output",
            str(RUN1_RECONSTRUCTION_BASELINE_PATH),
            "--finetuned-output",
            str(RUN1_RECONSTRUCTION_FINETUNED_PATH),
            "--summary-output",
            str(RUN1_SUMMARY_PATH.parent / 'reconstruction_summary.md'),
            "--llama-server-bin",
            args.llama_server_bin,
            "--context-size",
            str(args.context_size),
        ]
    )

    run(
        [
            args.llama_perplexity_bin,
            "-m",
            str(args.baseline_gguf),
            "-f",
            str(RUN1_SUMMARY_PATH.parent / 'eval_text.txt'),
        ],
        stdout_path=RUN1_PERPLEXITY_BASELINE_PATH,
    )
    run(
        [
            args.llama_perplexity_bin,
            "-m",
            str(args.finetuned_gguf),
            "-f",
            str(RUN1_SUMMARY_PATH.parent / 'eval_text.txt'),
        ],
        stdout_path=RUN1_PERPLEXITY_FINETUNED_PATH,
    )

    run(
        [
            "python3",
            "scripts/score_memorization.py",
            "--generated",
            str(RUN1_BASELINE_OUTPUTS_PATH),
            "--generated",
            str(RUN1_FINETUNED_OUTPUTS_PATH),
            "--output",
            str(RUN1_MEMORIZATION_REPORT_PATH),
        ]
    )

    run(["python3", "scripts/build_blind_review_sheet.py"])
    run(["python3", "scripts/render_run1_report.py"])

    print(f"Blind review sheet ready at {RUN1_BLIND_REVIEW_PATH}")


if __name__ == "__main__":
    main()
