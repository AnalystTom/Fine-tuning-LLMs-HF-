"""Run the same llama.cpp prompt suite across multiple model presets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from contextlib import ExitStack, contextmanager
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from .experiment_config import ARTIFACTS_DIR, EVAL_PROMPTS_PATH
    from .run_llamacpp_suite import (
        capture_outputs,
        read_jsonl,
        wait_for_port_release,
        wait_for_server,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import ARTIFACTS_DIR, EVAL_PROMPTS_PATH
    from run_llamacpp_suite import (
        capture_outputs,
        read_jsonl,
        wait_for_port_release,
        wait_for_server,
        write_jsonl,
    )


DEFAULT_MODELS = ("qwen35-4b", "qwen35-9b", "glm47-flash")
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "model_compare"
DEFAULT_LAUNCHER = Path(__file__).resolve().parent / "start_llama_server.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model presets understood by start_llama_server.sh.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=EVAL_PROMPTS_PATH,
        help="Prompt JSONL to run against each model.",
    )
    parser.add_argument(
        "--prompt-type",
        default="fixed_suite",
        help="Optional prompt_type filter. Use 'all' to disable filtering.",
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=DEFAULT_LAUNCHER,
        help="Path to start_llama_server.sh.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-model JSONL and markdown summary output.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to reuse sequentially for llama-server runs.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=8192,
        help="Context size for llama-server runs.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="CPU threads for llama-server runs.",
    )
    parser.add_argument(
        "--threads-batch",
        type=int,
        default=8,
        help="Batch CPU threads for llama-server runs.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the local llama-server instances.",
    )
    return parser.parse_args()


def filter_prompts(
    prompts: list[dict[str, Any]], prompt_type: str
) -> list[dict[str, Any]]:
    if prompt_type == "all":
        return prompts
    return [prompt for prompt in prompts if prompt.get("prompt_type") == prompt_type]


@contextmanager
def managed_preset_server(
    launcher: Path,
    preset: str,
    host: str,
    port: int,
    context_size: int,
    threads: int,
    threads_batch: int,
    log_path: Path,
):
    if not launcher.is_file():
        raise FileNotFoundError(f"Launcher not found: {launcher}")
    env = {
        **os.environ,
        "LLAMA_HOST": host,
        "LLAMA_PORT": str(port),
        "LLAMA_CTX_SIZE": str(context_size),
        "LLAMA_THREADS": str(threads),
        "LLAMA_THREADS_BATCH": str(threads_batch),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [str(launcher), preset],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(launcher.parent.parent),
        )
        base_url = f"http://{host}:{port}"
        try:
            wait_for_server(base_url, timeout_seconds=600)
            yield base_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
            wait_for_port_release(port)


def build_summary(all_rows: dict[str, list[dict[str, Any]]]) -> str:
    model_labels = list(all_rows)
    prompt_keys = sorted({(row["prompt_id"], int(row.get("seed", 3407))) for rows in all_rows.values() for row in rows})
    lines = ["# llama.cpp Model Comparison", ""]
    lines.extend(["## Overview", ""])
    for model_label in model_labels:
        rows = all_rows[model_label]
        avg_latency = int(mean(row["latency_ms"] for row in rows))
        avg_tokens = int(mean((row["tokens_generated"] or 0) for row in rows))
        lines.append(
            f"- `{model_label}`: `{len(rows)}` prompts, avg latency `{avg_latency} ms`, avg tokens `{avg_tokens}`"
        )
    lines.append("")

    by_model_prompt = {
        model_label: {(row["prompt_id"], int(row.get("seed", 3407))): row for row in rows}
        for model_label, rows in all_rows.items()
    }
    for prompt_id, seed in prompt_keys:
        exemplar = next(
            by_model_prompt[model_label][(prompt_id, seed)]
            for model_label in model_labels
            if (prompt_id, seed) in by_model_prompt[model_label]
        )
        lines.extend(
            [
                f"## {prompt_id} (seed {seed})",
                "",
                f"- Prompt type: `{exemplar['prompt_type']}`",
                f"- Platform: `{exemplar['platform']}`",
                "",
                "### Prompt",
                "",
                exemplar["prompt_text"],
                "",
            ]
        )
        for model_label in model_labels:
            row = by_model_prompt[model_label].get((prompt_id, seed))
            if not row:
                continue
            lines.extend(
                [
                    f"### {model_label}",
                    "",
                    row["response_text"],
                    "",
                    f"- Latency: `{row['latency_ms']} ms`",
                    f"- Tokens: `{row['tokens_generated']}`",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    prompts = filter_prompts(read_jsonl(args.prompts), args.prompt_type)
    if not prompts:
        raise ValueError("No prompts matched the requested filter")

    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for model_label in args.models:
        log_path = logs_dir / f"{model_label}.log"
        with managed_preset_server(
            launcher=args.launcher,
            preset=model_label,
            host=args.host,
            port=args.port,
            context_size=args.context_size,
            threads=args.threads,
            threads_batch=args.threads_batch,
            log_path=log_path,
        ) as endpoint:
            rows = capture_outputs(prompts, endpoint, model_label)
        output_path = output_dir / f"{model_label}.jsonl"
        write_jsonl(output_path, rows)
        all_rows[model_label] = rows
        print(f"Wrote {len(rows)} rows for {model_label} to {output_path}")

    summary = build_summary(all_rows)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Wrote comparison summary to {summary_path}")


if __name__ == "__main__":
    main()
