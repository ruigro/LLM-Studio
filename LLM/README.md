# LLM — Finetune / Train / Validate / Run

This document explains how to set up the environment and run the included scripts to finetune, train, validate, and run the model found in this repository's `LLM` folder.

**Prerequisites**
- Python 3.8+ installed
- Git
- (Optional) CUDA drivers and a suitable GPU for faster training
# LLM — Quick Usage

Minimal instructions to set up and use the LLM management scripts in this folder.

Setup
-----

Run these inside the `LLM` folder. Change into `LLM`, create and activate a virtual environment, then install the requirements:

```bash
cd LLM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Example training data (JSONL)
----------------------------

Save one example per line to `train_data.jsonl`:

```jsonl
{"instruction": "Translate the following sentence into French:", "output": "Bonjour le monde"}
```

Using the recommended helper: `workflow.py` (run inside `LLM`)
---------------------------------------

```bash
# Train only
cd LLM
python workflow.py train --data-path train_data.jsonl --output-dir ./fine_tuned_adapter --epochs 3 --batch-size 1

# Validate only
python workflow.py validate --adapter-dir ./fine_tuned_adapter/M1Checkpoint1

# Run inference
python workflow.py run --adapter-dir ./fine_tuned_adapter/M1Checkpoint1 --prompt "### Instruction:\nSummarize this paragraph.\n\n### Response:\n"

# Full pipeline (train -> validate -> run)
python workflow.py all --epochs 3 --prompt "Test prompt"
```

Manual commands (run inside `LLM`)
---------------

```bash
# Train (manual)
cd LLM
python finetune.py --data-path train_data.jsonl --output-dir ./fine_tuned_adapter --epochs 3 --batch-size 1

# Validate (manual)
python validate_prompts.py --adapter-dir ./fine_tuned_adapter/M1Checkpoint1 --base-model unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit --prompts validation_prompts.jsonl --out validation_results.jsonl

# Run (manual)
python run_adapter.py --adapter-dir ./fine_tuned_adapter/M1Checkpoint1 --prompt "### Instruction:\nSummarize this paragraph.\n\n### Response:\n"
```

Notes
-----

- Checkpoints are saved as `M{n}Checkpoint{m}` under your `--output-dir` (e.g. `fine_tuned_adapter/M1Checkpoint1`).
- W&B is offline by default. To enable online W&B, install `wandb` and run `workflow.py` with `--enable-wandb`.
- Preview cleanup with `bash cleanup_generated.sh` and delete with `bash cleanup_generated.sh --yes`.

Example instruction (short)
--------------------------

```jsonl
{"instruction": "Summarize the following paragraph in one sentence:", "output": "This paragraph explains how to..."}
```

That's it — use `workflow.py` for most tasks; manual commands remain available for custom workflows.
