"""Run prompt JSONL files against llama-server and compare baseline vs fine-tuned outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    from .experiment_config import (
        RUN1_BASELINE_OUTPUTS_PATH,
        RUN1_EVAL_SUITE_PATH,
        RUN1_FINETUNED_OUTPUTS_PATH,
        RUN1_SUMMARY_PATH,
    )
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import (
        RUN1_BASELINE_OUTPUTS_PATH,
        RUN1_EVAL_SUITE_PATH,
        RUN1_FINETUNED_OUTPUTS_PATH,
        RUN1_SUMMARY_PATH,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=RUN1_EVAL_SUITE_PATH,
        help="Prompt JSONL to run against llama-server.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8080",
        help="Base URL for a running llama-server instance.",
    )
    parser.add_argument(
        "--model-label",
        help="Label for a single capture run against an already-running server.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL path for a single capture run.",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="Existing JSONL output to compare against when writing the summary.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=RUN1_SUMMARY_PATH,
        help="Markdown summary output path.",
    )
    parser.add_argument(
        "--baseline-gguf",
        type=Path,
        help="Baseline GGUF path for managed llama-server capture.",
    )
    parser.add_argument(
        "--finetuned-gguf",
        type=Path,
        help="Fine-tuned GGUF path for managed llama-server capture.",
    )
    parser.add_argument(
        "--baseline-label",
        help="Model label to use when capturing --baseline-gguf.",
    )
    parser.add_argument(
        "--finetuned-label",
        help="Model label to use when capturing --finetuned-gguf.",
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=RUN1_BASELINE_OUTPUTS_PATH,
        help="Baseline JSONL output path in managed mode.",
    )
    parser.add_argument(
        "--finetuned-output",
        type=Path,
        default=RUN1_FINETUNED_OUTPUTS_PATH,
        help="Fine-tuned JSONL output path in managed mode.",
    )
    parser.add_argument(
        "--llama-server-bin",
        default="llama-server",
        help="llama-server executable to use in managed mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for managed llama-server runs.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=8192,
        help="Context size passed to llama-server in managed mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling value.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def make_request(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.loads(response.read().decode("utf-8"))


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["prompt_id"]), int(row.get("seed", 3407))


def sanitize_label(value: str) -> str:
    cleaned: list[str] = []
    previous_separator = False
    for char in value.strip().lower():
        if char.isalnum():
            cleaned.append(char)
            previous_separator = False
            continue
        if not previous_separator:
            cleaned.append("_")
            previous_separator = True
    label = "".join(cleaned).strip("_")
    return label or "model"


def default_managed_label(role: str, model_path: Path) -> str:
    return f"{role}_{sanitize_label(model_path.stem)}"


def capture_outputs(
    prompts: list[dict[str, Any]],
    endpoint: str,
    model_label: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    chat_endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
    for prompt in prompts:
        payload = {
            "model": model_label,
            "messages": prompt["messages"],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": int(prompt.get("seed", 3407)),
            "max_tokens": int(prompt["max_tokens"]),
        }
        started = time.monotonic()
        response = make_request(chat_endpoint, payload)
        latency_ms = int((time.monotonic() - started) * 1000)
        choice = response["choices"][0]
        response_text = choice["message"]["content"]
        usage = response.get("usage", {})
        outputs.append(
            {
                "suite_name": prompt.get("suite_name"),
                "prompt_type": prompt.get("prompt_type"),
                "bucket": prompt.get("bucket"),
                "model_label": model_label,
                "prompt_id": prompt["prompt_id"],
                "seed": int(prompt.get("seed", 3407)),
                "platform": prompt.get("platform"),
                "prompt_text": prompt.get("prompt_text"),
                "reference_text": prompt.get("reference_text"),
                "response_text": response_text,
                "latency_ms": latency_ms,
                "tokens_generated": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "char_count": len(response_text),
                "word_count": len(response_text.split()),
            }
        )
    return outputs


def wait_for_server(base_url: str, timeout_seconds: int = 180) -> None:
    model_endpoint = base_url.rstrip("/") + "/v1/models"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(model_endpoint, timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for llama-server at {base_url}")


def wait_for_port_release(port: int, timeout_seconds: int = 30) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"Port {port} did not become free in time")


@contextmanager
def managed_server(
    llama_server_bin: str,
    model_path: Path,
    port: int,
    context_size: int,
    alias: str,
):
    if shutil.which(llama_server_bin) is None:
        raise FileNotFoundError(f"Unable to find llama-server executable: {llama_server_bin}")
    command = [
        llama_server_bin,
        "-m",
        str(model_path),
        "--alias",
        alias,
        "--port",
        str(port),
        "-c",
        str(context_size),
        "--reasoning-budget",
        "0",
        "--reasoning-format",
        "none",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
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


def load_outputs_by_key(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    return {row_key(row): row for row in read_jsonl(path)}


def build_summary(
    baseline_rows: dict[tuple[str, int], dict[str, Any]],
    finetuned_rows: dict[tuple[str, int], dict[str, Any]],
) -> str:
    keys = sorted(set(baseline_rows) | set(finetuned_rows))
    lines = ["# llama.cpp Evaluation Summary", ""]
    for prompt_id, seed in keys:
        baseline = baseline_rows.get((prompt_id, seed))
        finetuned = finetuned_rows.get((prompt_id, seed))
        exemplar = baseline or finetuned
        assert exemplar is not None
        lines.extend(
            [
                f"## {prompt_id} (seed {seed})",
                "",
                f"- Bucket: `{exemplar.get('bucket')}`",
                f"- Prompt type: `{exemplar.get('prompt_type')}`",
                f"- Platform: `{exemplar.get('platform')}`",
                "",
                "### Prompt",
                "",
                str(exemplar.get("prompt_text", "")),
                "",
            ]
        )
        reference_text = exemplar.get("reference_text")
        if reference_text:
            lines.extend(["### Held-out Reference", "", str(reference_text), ""])
        if baseline:
            lines.extend(
                [
                    "### Baseline",
                    "",
                    baseline["response_text"],
                    "",
                    f"- Latency: `{baseline['latency_ms']} ms`",
                    f"- Tokens: `{baseline['tokens_generated']}`",
                    f"- Finish reason: `{baseline['finish_reason']}`",
                    "",
                ]
            )
        if finetuned:
            lines.extend(
                [
                    "### Fine-tuned",
                    "",
                    finetuned["response_text"],
                    "",
                    f"- Latency: `{finetuned['latency_ms']} ms`",
                    f"- Tokens: `{finetuned['tokens_generated']}`",
                    f"- Finish reason: `{finetuned['finish_reason']}`",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def run_managed_capture(
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    *,
    model_path: Path,
    model_label: str,
    output_path: Path,
) -> list[dict[str, Any]]:
    with managed_server(
        args.llama_server_bin,
        model_path,
        args.port,
        args.context_size,
        model_label,
    ) as endpoint:
        rows = capture_outputs(
            prompts,
            endpoint,
            model_label,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return rows


def run_managed_mode(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> None:
    baseline_rows: dict[tuple[str, int], dict[str, Any]] | None = None
    finetuned_rows: dict[tuple[str, int], dict[str, Any]] | None = None

    if args.baseline_gguf:
        baseline_label = args.baseline_label or default_managed_label("baseline", args.baseline_gguf)
        captured = run_managed_capture(
            args,
            prompts,
            model_path=args.baseline_gguf,
            model_label=baseline_label,
            output_path=args.baseline_output,
        )
        baseline_rows = {row_key(row): row for row in captured}

    if args.finetuned_gguf:
        finetuned_label = args.finetuned_label or default_managed_label("finetuned", args.finetuned_gguf)
        captured = run_managed_capture(
            args,
            prompts,
            model_path=args.finetuned_gguf,
            model_label=finetuned_label,
            output_path=args.finetuned_output,
        )
        finetuned_rows = {row_key(row): row for row in captured}

    if baseline_rows is not None and finetuned_rows is not None:
        summary = build_summary(baseline_rows=baseline_rows, finetuned_rows=finetuned_rows)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(summary, encoding="utf-8")
        print(f"Wrote managed comparison summary to {args.summary_output}")


def run_single_capture_mode(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> None:
    if not args.model_label or not args.output:
        raise ValueError("Single-server mode requires --model-label and --output")
    rows = capture_outputs(
        prompts,
        args.endpoint,
        args.model_label,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")

    if args.compare_to:
        current = {row_key(row): row for row in rows}
        peer = load_outputs_by_key(args.compare_to)
        if args.model_label.startswith("baseline"):
            baseline_rows, finetuned_rows = current, peer
        else:
            baseline_rows, finetuned_rows = peer, current
        summary = build_summary(baseline_rows, finetuned_rows)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(summary, encoding="utf-8")
        print(f"Wrote comparison summary to {args.summary_output}")


def main() -> None:
    args = parse_args()
    prompts = read_jsonl(args.prompts)
    if args.baseline_gguf or args.finetuned_gguf:
        run_managed_mode(args, prompts)
        return
    run_single_capture_mode(args, prompts)


if __name__ == "__main__":
    main()
