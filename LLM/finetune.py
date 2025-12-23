import argparse
import os
import sys
import json
import re
import torch

# Default to offline W&B unless explicitly enabled via --enable-wandb or env var
if "--enable-wandb" not in sys.argv and "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "offline"

# If Weave is available, import it so W&B can enable Weave features locally
try:
    import weave  # type: ignore
    print("Weave imported: enhanced LLM call tracing enabled (local).")
except Exception:
    # weave is optional; continue without it
    pass
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--data-path", default="train_data.jsonl")
    p.add_argument("--output-dir", default="./fine_tuned_adapter")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate for training")
    p.add_argument("--max-examples", type=int, default=None, help="Limit dataset for quick runs")
    return p.parse_args()


def main():
    args = parse_args()

    MODEL_NAME = args.model_name
    MAX_SEQ_LENGTH = args.max_seq_length
    DATASET_PATH = args.data_path
    OUTPUT_DIR = args.output_dir

    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION = args.grad_accum
    LEARNING_RATE = args.learning_rate

    print("Loading model and tokenizer in 4-bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        device_map="auto",
        quantization_config=bnb_config,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print("Preparing dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    def formatting_func(example):
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return {"text": text}

    dataset = dataset.map(formatting_func, remove_columns=dataset.column_names)

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=LEARNING_RATE,  # Use the configurable learning rate
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",
            seed=3407,
            # Disable Hugging Face Trainer intermediate checkpoints (creates `checkpoint-<step>` dirs)
            save_strategy="no",
            # Keep a small number if you enable saving later
            save_total_limit=2,
        ),
    )

    trainer.train()

    print("Saving LoRA adapters and tokenizer...")

    # Prepare a mapping file to keep consistent per-base-model IDs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    map_path = os.path.join(OUTPUT_DIR, "model_map.json")
    try:
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as mf:
                model_map = json.load(mf)
        else:
            model_map = {}
    except Exception:
        model_map = {}

    # Assign a short model id like M1, M2 for the base model name
    if MODEL_NAME in model_map:
        mid = model_map[MODEL_NAME]
    else:
        # next index
        next_idx = len(model_map) + 1
        mid = f"M{next_idx}"
        model_map[MODEL_NAME] = mid
        try:
            with open(map_path, "w", encoding="utf-8") as mf:
                json.dump(model_map, mf, indent=2)
        except Exception:
            pass

    # Find existing checkpoints for this model id and pick next numeric suffix
    existing = []
    try:
        for name in os.listdir(OUTPUT_DIR):
            m = re.match(rf"^{re.escape(mid)}Checkpoint(\d+)$", name)
            if m:
                existing.append(int(m.group(1)))
    except Exception:
        existing = []

    next_num = max(existing) + 1 if existing else 1
    peft_out = os.path.join(OUTPUT_DIR, f"{mid}Checkpoint{next_num}")
    os.makedirs(peft_out, exist_ok=True)

    model.save_pretrained(peft_out)
    tokenizer.save_pretrained(peft_out)
    print(f"Finetuning complete! Adapter saved to: {peft_out}")


if __name__ == "__main__":
    main()