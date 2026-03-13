# Qwen3.5 Brand Voice Fine-Tuning (Local + Runpod)

This repository contains a practical workflow to:

1. curate public X + LinkedIn posts
2. build a text-only chat-format SFT dataset
3. fine-tune `Qwen3.5` with Unsloth
4. compare baseline vs fine-tuned outputs in `llama.cpp`

The workflow is designed so you can train on Runpod and run evaluation locally.

## Unsloth Documentation (Official)

- Docs home: [https://unsloth.ai/docs](https://unsloth.ai/docs)
- Installation: [https://unsloth.ai/docs/get-started/install](https://unsloth.ai/docs/get-started/install)
- Fine-tuning guide: [https://unsloth.ai/docs/get-started/fine-tuning-llms-guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- Notebook catalog: [https://unsloth.ai/docs/get-started/unsloth-notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)
- Qwen guide: [https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
- Saving to GGUF: [https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)

## Repository Layout

- `scripts/prep_social_exports.py`: parse LinkedIn/X export zips into cleaned public posts
- `scripts/export_review_sheet.py`: build CSV for keep/drop curation
- `scripts/apply_review_sheet.py`: apply curation and produce run-1 corpus
- `scripts/build_sft_dataset.py`: create train/eval chat-format JSONL
- `scripts/build_eval_suite.py`: create fixed evaluation prompt suite
- `scripts/run_llamacpp_suite.py`: run prompts against `llama-server`
- `scripts/run_run1_posttrain.py`: orchestrate post-train eval flow
- `scripts/install_heretic.sh`: install/update heretic into `tools/heretic`
- `notebooks/qwen35_brand_voice_llamacpp_experiment.ipynb`: notebook training/export path

## Quick Start (Local)

Use this if you want data prep + evaluation on your own machine.

1. Install Python + llama.cpp:

```bash
brew install llama.cpp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Build dataset from export zips:

```bash
python3 scripts/prep_social_exports.py \
  --linkedin-zip /absolute/path/to/LinkedIn-export.zip \
  --x-zip /absolute/path/to/X-export.zip

python3 scripts/export_review_sheet.py
python3 scripts/apply_review_sheet.py
python3 scripts/build_sft_dataset.py --labeler-backend heuristic
python3 scripts/build_eval_suite.py
```

3. Download baseline GGUF:

```bash
./scripts/download_llama_models.sh qwen35-4b
```

4. Run baseline-only evaluation:

```bash
python3 scripts/run_llamacpp_suite.py \
  --prompts data/processed/eval_suite_run1_social_20x3.jsonl \
  --baseline-gguf artifacts/gguf/qwen35_4b_baseline_q4_k_m.gguf \
  --baseline-output artifacts/eval/run1/baseline_qwen35_4b_outputs.jsonl
```

## Quick Start (Runpod)

Use this for training runs.

1. Create a GPU pod in Runpod (for example `RTX 4090 24GB`), then SSH into it.
2. Clone repo and set up environment:

```bash
git clone https://github.com/AnalystTom/Fine-tuning-LLMs-HF-.git
cd Fine-tuning-LLMs-HF-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Install Unsloth following the official installation docs for your CUDA/runtime:

- [https://unsloth.ai/docs/get-started/install](https://unsloth.ai/docs/get-started/install)

4. Open and run notebook:

- `notebooks/qwen35_brand_voice_llamacpp_experiment.ipynb`

5. Ensure dataset files exist in `data/processed/`:

- `train_run1_qwen35_9b.jsonl`
- `eval_run1_qwen35_9b.jsonl`

6. After training, export LoRA / GGUF artifacts and copy needed files back locally for evaluation.

## Data Format

Training data is OpenAI-style chat messages:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

The training notebook/scripts render rows into a single `text` field via chat template before SFT.

## Baseline vs Fine-Tuned Evaluation

Run full suite after you have both baseline and fine-tuned GGUF files:

```bash
python3 scripts/run_llamacpp_suite.py \
  --prompts data/processed/eval_suite_run1_social_20x3.jsonl \
  --baseline-gguf artifacts/gguf/qwen35_4b_baseline_q4_k_m.gguf \
  --finetuned-gguf artifacts/gguf/qwen35_4b_brand_voice_q4_k_m.gguf \
  --baseline-output artifacts/eval/run1/baseline_qwen35_4b_outputs.jsonl \
  --finetuned-output artifacts/eval/run1/finetuned_qwen35_4b_outputs.jsonl
```

Run held-out reconstruction:

```bash
python3 scripts/run_llamacpp_suite.py \
  --prompts data/processed/reconstruction_prompts_run1_qwen35_9b.jsonl \
  --baseline-gguf artifacts/gguf/qwen35_4b_baseline_q4_k_m.gguf \
  --finetuned-gguf artifacts/gguf/qwen35_4b_brand_voice_q4_k_m.gguf \
  --baseline-output artifacts/eval/run1/reconstruction_baseline.jsonl \
  --finetuned-output artifacts/eval/run1/reconstruction_finetuned.jsonl \
  --summary-output artifacts/eval/run1/reconstruction_summary.md
```

## Optional: Convert LoRA Adapter to GGUF Adapter

This is only needed if you want to apply LoRA adapters directly in `llama.cpp`.

1. Install optional conversion dependencies:

```bash
pip install -r requirements-gguf.txt
```

2. Run converter:

```bash
PYTHONPATH=/opt/homebrew/opt/llama.cpp/libexec \
python3 scripts/convert_lora_to_gguf.py \
  --base-model-id unsloth/Qwen3.5-4B \
  --outfile artifacts/lora/qwen35_4b_brand_voice_lora.gguf \
  /path/to/lora_adapter_dir
```

## heretic Install

```bash
./scripts/install_heretic.sh
```

The repo ignores `tools/heretic/` so vendor code is not committed.

## Tests

```bash
python3 -m pytest
```
