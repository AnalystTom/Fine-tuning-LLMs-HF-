# Qwen3.5 Brand Voice Experiments

This repo runs a local experiment loop for fine-tuning `Qwen3.5` models on curated public X and LinkedIn posts, exporting baseline and fine-tuned models to GGUF, and comparing both models with `llama.cpp`.

Run 1 is intentionally narrow:
- optimize for social voice first
- use public posts only
- compare baseline vs fine-tuned under identical runtime conditions
- treat blogs as evaluation probes, not the primary training target

## Repository Layout

- `scripts/prep_social_exports.py`: ingest LinkedIn and X export zips and write cleaned public-post JSONL.
- `scripts/export_review_sheet.py`: generate an editable CSV with keep/drop recommendations for run-1 curation.
- `scripts/apply_review_sheet.py`: apply the edited review sheet and write the curated run-1 corpus.
- `scripts/build_sft_dataset.py`: label the curated corpus and build run-1 train/eval JSONL plus held-out reconstruction prompts.
- `scripts/build_eval_suite.py`: build the fixed `20 x 3` evaluation suite.
- `scripts/run_llamacpp_suite.py`: run prompt JSONL through `llama-server` and write baseline/fine-tuned outputs plus a markdown summary.
- `scripts/score_memorization.py`: flag generated outputs with long contiguous overlap against the train corpus.
- `scripts/build_blind_review_sheet.py`: build the blind A/B review sheet and human scoring template.
- `scripts/score_human_eval.py`: score human ratings and produce a weighted evaluation summary.
- `scripts/run_run1_posttrain.py`: orchestrate run-1 baseline/fine-tuned evals, perplexity, memorization, blind review, and summary generation after the notebook export.
- `scripts/render_run1_report.py`: render the run-1 experiment summary and blog-post notes from available artifacts.
- `notebooks/qwen35_brand_voice_llamacpp_experiment.ipynb`: Colab notebook for `Qwen3.5-9B` QLoRA and GGUF export.

## Run 1 Artifacts

Curated dataset artifacts:
- `data/processed/public_posts_review.csv`
- `data/processed/public_posts_run1_curated.jsonl`
- `data/processed/public_posts_labeled_run1.jsonl`
- `data/processed/train_run1_qwen35_9b.jsonl`
- `data/processed/eval_run1_qwen35_9b.jsonl`
- `data/processed/reconstruction_prompts_run1_qwen35_9b.jsonl`
- `data/processed/eval_suite_run1_social_20x3.jsonl`
- `artifacts/eval/run1/eval_text.txt`

Expected post-training artifacts:
- `artifacts/gguf/qwen35_9b_baseline_q4_k_m.gguf`
- `artifacts/gguf/qwen35_9b_brand_voice_q4_k_m.gguf`
- `artifacts/eval/run1/baseline_qwen35_9b_outputs.jsonl`
- `artifacts/eval/run1/finetuned_qwen35_9b_outputs.jsonl`
- `artifacts/eval/run1/blind_review.csv`
- `artifacts/eval/run1/human_scores.csv`
- `artifacts/eval/run1/blind_eval_summary.md`

## Local Data Prep And Curation

Prepare the raw public-post corpus:

```bash
python3 scripts/prep_social_exports.py \
  --linkedin-zip /absolute/path/to/LinkedIn-export.zip \
  --x-zip /absolute/path/to/X-export.zip
```

Create the editable review sheet and curated run-1 corpus:

```bash
python3 scripts/export_review_sheet.py
python3 scripts/apply_review_sheet.py
```

Build the run-1 SFT dataset and evaluation suite:

```bash
python3 scripts/build_sft_dataset.py --labeler-backend heuristic
python3 scripts/build_eval_suite.py
```

Notes:
- `build_sft_dataset.py` supports `--labeler-backend openai` for an OpenAI-compatible local labeler at `temperature=0`.
- The checked-in workflow currently uses the heuristic labeler by default because a local labeler endpoint is not guaranteed to be running.

## Fine-Tuning In Colab

Open `notebooks/qwen35_brand_voice_llamacpp_experiment.ipynb` in Colab and mount or upload this repo.

The notebook:
- loads `unsloth/Qwen3.5-9B`
- uses 4-bit QLoRA, not 16-bit LoRA
- trains on `train_run1_qwen35_9b.jsonl`
- evaluates on `eval_run1_qwen35_9b.jsonl`
- saves the LoRA adapter
- saves the merged model
- exports baseline and fine-tuned GGUFs with `q4_k_m`

## llama.cpp Runtime

Install locally:

```bash
brew install llama.cpp
```

Download smoke-test models or comparison models:

```bash
./scripts/download_llama_models.sh qwen35-4b
./scripts/download_llama_models.sh qwen35-9b
```

Start a local server with preset models:

```bash
./scripts/start_llama_server.sh qwen35-4b
./scripts/start_llama_server.sh qwen35-9b
./scripts/start_llama_server.sh glm47-flash
```

Install `heretic` for local safety red-team research:

```bash
./scripts/install_heretic.sh
```

### Codex Bridge (Unsloth guide)

Use this flow to make `Qwen3.5-9B` available in `codex` via `llama-server` + OpenAI-compatible `/v1` API.

1. Start `llama-server` on a stable port (`8001` matches Codex examples):

```bash
./scripts/start_qwen35_9b_codex_server.sh
```

2. Install a provider entry in `~/.codex/config.toml`:

```bash
./scripts/setup_codex_qwen35_9b_provider.sh
```

3. Run Codex using that provider/model:

```bash
codex -c model_provider=qwen35_local -c model=qwen35-9b --search
```

Or use the wrapper script:

```bash
./scripts/run_codex_qwen35_9b.sh --search
```

If `start_qwen35_9b_codex_server.sh` is already running, you can smoke-test the endpoint directly:

```bash
python3 scripts/llama9b_agent_smoke_test.py
```

Run a local file-generation task from the model:

```bash
python3 scripts/llama9b_file_agent.py --output hello.txt
```

The file-task script prints `hello.txt` containing model-generated content and is useful as a practical end-to-end local-agent check.

## Baseline vs Fine-Tuned Evaluation

To run only the unfine-tuned baseline in managed `llama.cpp` mode (for first-pass baseline capture):

```bash
python3 scripts/run_llamacpp_suite.py \
  --prompts data/processed/eval_suite_run1_social_20x3.jsonl \
  --baseline-gguf artifacts/gguf/qwen35_4b_baseline_q4_k_m.gguf \
  --baseline-output artifacts/eval/run1/baseline_qwen35_4b_outputs.jsonl
```

After the Colab notebook exports the two run-1 GGUF files, run the full evaluation suite locally:

```bash
python3 scripts/run_llamacpp_suite.py \
  --prompts data/processed/eval_suite_run1_social_20x3.jsonl \
  --baseline-gguf artifacts/gguf/qwen35_4b_baseline_q4_k_m.gguf \
  --finetuned-gguf artifacts/gguf/qwen35_4b_brand_voice_q4_k_m.gguf \
  --baseline-output artifacts/eval/run1/baseline_qwen35_4b_outputs.jsonl \
  --finetuned-output artifacts/eval/run1/finetuned_qwen35_4b_outputs.jsonl
```

Run held-out reconstruction with the same harness:

```bash
python3 scripts/run_llamacpp_suite.py \
  --prompts data/processed/reconstruction_prompts_run1_qwen35_9b.jsonl \
  --baseline-gguf artifacts/gguf/qwen35_4b_baseline_q4_k_m.gguf \
  --finetuned-gguf artifacts/gguf/qwen35_4b_brand_voice_q4_k_m.gguf \
  --baseline-output artifacts/eval/run1/reconstruction_baseline.jsonl \
  --finetuned-output artifacts/eval/run1/reconstruction_finetuned.jsonl \
  --summary-output artifacts/eval/run1/reconstruction_summary.md
```

## Perplexity, Memorization, And Human Review

Perplexity:

```bash
llama-perplexity -m artifacts/gguf/qwen35_4b_baseline_q4_k_m.gguf -f artifacts/eval/run1/eval_text.txt > artifacts/eval/run1/perplexity_baseline.txt
llama-perplexity -m artifacts/gguf/qwen35_4b_brand_voice_q4_k_m.gguf -f artifacts/eval/run1/eval_text.txt > artifacts/eval/run1/perplexity_finetuned.txt
```

Memorization:

```bash
python3 scripts/score_memorization.py \
  --train data/processed/train_run1_qwen35_9b.jsonl \
  --generated artifacts/eval/run1/baseline_qwen35_4b_outputs.jsonl \
  --generated artifacts/eval/run1/finetuned_qwen35_4b_outputs.jsonl
```

Blind review packet:

```bash
python3 scripts/build_blind_review_sheet.py
```

After filling in `artifacts/eval/run1/human_scores.csv`, score the human review:

```bash
python3 scripts/score_human_eval.py
```
To run the full post-train workflow in one command after the two GGUFs exist:

```bash
python3 scripts/run_run1_posttrain.py \
  --baseline-gguf artifacts/gguf/qwen35_4b_baseline_q4_k_m.gguf \
  --finetuned-gguf artifacts/gguf/qwen35_4b_brand_voice_q4_k_m.gguf
```

## Tests

Run:

```bash
python3 -m pytest
```
