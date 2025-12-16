# LLM — Finetune / Train / Validate / Run

This document explains how to set up the environment and run the included scripts to finetune, train, validate, and run the model found in this repository's `LLM` folder.

**Prerequisites**
- Python 3.8+ installed
- Git
- (Optional) CUDA drivers and a suitable GPU for faster training

**Setup**
1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

2. Install required packages. If a `requirements.txt` is present, use it; otherwise install typical packages used for finetuning:

```bash
pip install -r requirements.txt  # if available
# or common packages:
pip install torch transformers accelerate datasets peft safetensors sentencepiece
```

**Files of interest**
- `finetune.py` — script used to finetune/train the model on `train_data.jsonl`.
- `run_adapter.py` — script to run inference against a finetuned adapter/model.
- `train_data.jsonl` — training data in JSONL format.
- `validate_prompts.py` — script to run validation prompts and produce results.
- `validation_prompts.jsonl` — validation prompts.
- `validation_results.jsonl` — sample or output validation results.

Note: Inspect each script to confirm CLI flags and expected arguments; the examples below are generic and may need to be adjusted to the script's actual flags.

**Finetune / Train (example)**
Run `finetune.py` pointing at your training data and desired output directory. Replace flags with those implemented by the script when needed.

```bash
python LLM/finetune.py \
  --train-file LLM/train_data.jsonl \
  --output-dir fine_tuned_adapter \
  --epochs 3 \
  --batch-size 8
```

If your script uses `accelerate` or other libraries, configure them beforehand (e.g., `accelerate config`).

**Validate**
Run `validate_prompts.py` (or `validation_prompts.jsonl`) to generate evaluation results. Example:

```bash
python LLM/validate_prompts.py \
  --prompts LLM/validation_prompts.jsonl \
  --model-dir fine_tuned_adapter \
  --output LLM/validation_results.jsonl
```

**Run / Inference**
Use `run_adapter.py` to load the finetuned model and run interactive prompts or scripted inference. Example:

```bash
python LLM/run_adapter.py \
  --model-dir fine_tuned_adapter \
  --prompt "Hello, please summarize the following text..."
```

You can also run the script without flags if it reads defaults from a config file — inspect `run_adapter.py` for details.

**Tips & Notes**
- Adjust batch sizes and learning rates to fit your GPU memory.
- Exclude large model files from git (see repository `.gitignore`).
- If using Hugging Face tooling, set `HF_HOME` or use `transformers` caching to a suitable location.
- Back up important checkpoints outside the repository before cleaning or reinitializing directories.

If you'd like, I can:
- Add a `requirements.txt` capturing the exact packages used here.
- Add example CLI wrappers or a small shell script to run common workflows.

**Cross-platform notes**
- On Windows use PowerShell or WSL for the bash commands shown above; use a Conda environment or `py -m venv .venv` to create virtual environments.
- For CPU-only environments, install CPU builds of PyTorch (see https://pytorch.org for platform-specific install commands).
- Set these environment variables to control caching locations if you want reproducible behavior across machines:
  - `HF_HOME` — Hugging Face home directory
  - `TRANSFORMERS_CACHE` — Transformers model cache directory

**Cleanup (safe, cross-machine)**
This repository includes `cleanup_generated.sh` (at the repo root) which performs a dry-run by default and removes ephemeral caches and IDE folders when run with `--yes`.

Dry-run (lists targets, does not delete):

```bash
bash cleanup_generated.sh
```

To actually delete the listed ephemeral files/folders:

```bash
bash cleanup_generated.sh --yes
```

Windows (PowerShell) equivalent (dry-run):

```powershell
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Recurse -File -Include *.pyc,*.log -ErrorAction SilentlyContinue
```

Notes:
- The cleanup script intentionally skips directories that look like model checkpoints or `data` directories (it will print a skip message). It only removes ephemeral caches and logs by default.
- If you'd like me to be more or less aggressive (e.g. remove `validation_results.jsonl` or old checkpoints), tell me which patterns you want removed and I will adjust the script and re-run it.
