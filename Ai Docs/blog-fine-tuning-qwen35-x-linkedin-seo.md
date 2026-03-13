---
title: "Fine-Tuning Qwen3.5 on X and LinkedIn Data: End-to-End Guide with Unsloth and llama.cpp"
slug: "fine-tuning-qwen35-x-linkedin-unsloth-llamacpp"
meta_description: "A practical guide to fine-tuning Qwen3.5-4B on X and LinkedIn writing data using Unsloth QLoRA, then evaluating baseline vs fine-tuned outputs with llama.cpp."
meta_keywords:
  - fine-tuning qwen3.5
  - unsloth qlora tutorial
  - llama.cpp evaluation
  - train llm on personal writing
  - linkedin and x dataset
canonical_url: "https://your-site.com/blog/fine-tuning-qwen35-x-linkedin-unsloth-llamacpp"
author: "Thomas Mann"
published_at: "2026-03-13"
---

# Fine-Tuning Qwen3.5 on X and LinkedIn Data: End-to-End Guide

If you want an LLM to sound more like your writing voice, the fastest path is a narrow, text-only fine-tune with clean chat-format data and a strict evaluation loop.

This post covers exactly how I fine-tuned `Qwen3.5-4B` with `Unsloth` on X and LinkedIn data, plus how I evaluated baseline vs fine-tuned behavior in `llama.cpp`.

## TL;DR Results

- Model: `Qwen3.5-4B` (Unsloth, 4-bit QLoRA)
- Dataset: `171` train rows, `33` eval rows
- Training: `3` epochs, `33` steps
- Train loss: `2.970`
- Eval loss: `3.014`
- Quick evaluation slice (`12` prompts):
  - Baseline avg tokens: `52.33`
  - Fine-tuned avg tokens: `76.08`
  - Fine-tuned outputs were longer on `10/12` prompt-seed pairs

## Why I Used Unsloth for Qwen3.5 Fine-Tuning

`Unsloth` makes practical QLoRA runs easier to execute:

- efficient 4-bit loading for lower VRAM pressure
- straightforward trainer integration for chat-style SFT
- clean save/export flow for LoRA artifacts

For small-to-medium adaptation runs, this gives a fast iteration loop without overcomplicating infrastructure.

## Data Format That Worked

I used OpenAI-style chat messages in JSONL:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Then each row was rendered into a single training field with:

- `tokenizer.apply_chat_template(..., add_generation_prompt=False)`

This keeps training aligned with how the model will be prompted later.

## Training Setup

- Base model: `unsloth/Qwen3.5-4B`
- Mode: text-only SFT (no image/video transforms)
- Runtime: Runpod RTX 4090 24GB
- Data:
  - `data/processed/train_run1_qwen35_9b.jsonl`
  - `data/processed/eval_run1_qwen35_9b.jsonl`

Even though file names contain `9b`, the successful training run in this cycle was `4b`.

## How I Evaluated Baseline vs Fine-Tuned in llama.cpp

I used the same `llama.cpp` runtime path for both variants:

1. Baseline GGUF run
2. Fine-tuned run with LoRA adapter applied
3. Same prompt set and decode settings

This reduces runtime confounds and makes output comparisons more credible.

## What Changed in Outputs

On the quick evaluation slice:

- Fine-tuned outputs were generally longer (`76.08` avg tokens vs `52.33`)
- Latency increased as expected with longer generations
- Style coherence improved in several prompts, but a full-suite run is still needed before making strong claims

## Practical Lessons

- Validate dataset schema before training.
- Keep the training path text-only if your source is text-only.
- Use the same inference runtime for baseline and fine-tuned comparisons.
- Start with a small eval slice for iteration speed, then run the full suite.

## Reproducible Workflow Commands

Install `llama.cpp`:

```bash
brew install llama.cpp
```

Download baseline model:

```bash
./scripts/download_llama_models.sh qwen35-4b
```

Run baseline-only eval:

```bash
python3 scripts/run_llamacpp_suite.py \
  --prompts data/processed/eval_suite_run1_social_20x3.jsonl \
  --baseline-gguf artifacts/gguf/qwen35_4b_baseline_q4_k_m.gguf \
  --baseline-output artifacts/eval/run1/baseline_qwen35_4b_outputs.jsonl
```

Run baseline vs fine-tuned comparison:

```bash
python3 scripts/run_llamacpp_suite.py \
  --prompts data/processed/eval_suite_run1_social_20x3.jsonl \
  --baseline-gguf artifacts/gguf/qwen35_4b_baseline_q4_k_m.gguf \
  --finetuned-gguf artifacts/gguf/qwen35_4b_brand_voice_q4_k_m.gguf \
  --baseline-output artifacts/eval/run1/baseline_qwen35_4b_outputs.jsonl \
  --finetuned-output artifacts/eval/run1/finetuned_qwen35_4b_outputs.jsonl
```

## Final Takeaway

If your goal is voice adaptation, a focused QLoRA run on clean chat-format examples can move quality quickly. The important part is not just training; it is disciplined evaluation under identical runtime conditions.

