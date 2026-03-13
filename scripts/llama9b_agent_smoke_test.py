#!/usr/bin/env python3
"""Smoke test for a local llama-server chat-completions endpoint.

Defaults to the Codex-oriented local endpoint at 127.0.0.1:8001, but any endpoint can be
passed via `--api-url` or LLAMA_API_URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib import request

DEFAULT_API_URL = "http://127.0.0.1:8001/v1/chat/completions"


def build_request(url: str) -> request.Request:
    payload = {
        "model": "qwen35-9b",
        "messages": [
            {"role": "user", "content": "Reply only with: hello world"}
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": 1,
    }
    return request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="llama-server chat-completions endpoint",
    )
    args = parser.parse_args()

    try:
        with request.urlopen(build_request(args.api_url), timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"request_failed: {exc}")
        sys.exit(1)

    message = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    print(message)
    print("ok=", "hello world" in message.lower())


if __name__ == "__main__":
    main()
