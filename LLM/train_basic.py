#!/usr/bin/env python3
"""
Simple training WITHOUT unsloth - uses standard PEFT/LoRA
This will work.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import sys
import argparse
import json
import os
import time

# Parse arguments or use defaults
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="./models/unsloth_llama-3.1-8b-instruct-unsloth-bnb-4bit")
parser.add_argument("--data-path", type=str, default="train_data.jsonl")
parser.add_argument("--output-dir", type=str, default="./fine_tuned_adapter")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--lora-r", type=int, default=8)
parser.add_argument("--lora-alpha", type=int, default=16)
parser.add_argument("--lora-dropout", type=float, default=0.05)
parser.add_argument("--grad-accum", type=int, default=8)
parser.add_argument("--max-seq-length", type=int, default=512)
parser.add_argument("--max-examples", type=int, default=None, help="Limit dataset size for testing")
args = parser.parse_args()

# Or read from config file if it exists
config_file = os.path.join(os.path.dirname(__file__), "train_config.json")
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        args.data_path = config.get("dataset_path", args.data_path)
        args.output_dir = config.get("output_dir", args.output_dir)
        args.epochs = config.get("epochs", args.epochs)

MODEL_NAME = args.model_name
DATA_PATH = args.data_path
OUTPUT_DIR = args.output_dir
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LORA_R = args.lora_r
LORA_ALPHA = args.lora_alpha
LORA_DROPOUT = args.lora_dropout
GRAD_ACCUM = args.grad_accum
MAX_SEQ_LENGTH = args.max_seq_length
MAX_EXAMPLES = args.max_examples

print(f"Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Dataset: {DATA_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  LoRA R: {LORA_R}")
print(f"  LoRA Alpha: {LORA_ALPHA}")
print(f"  LoRA Dropout: {LORA_DROPOUT}")
print(f"  Gradient Accumulation: {GRAD_ACCUM}")
print(f"  Max Seq Length: {MAX_SEQ_LENGTH}")
if MAX_EXAMPLES:
    print(f"  Max Examples: {MAX_EXAMPLES}")
print()

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print("Preparing model for training...")
model = prepare_model_for_kbit_training(model)

print("Adding LoRA adapters...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# Limit dataset size if max_examples is specified
if MAX_EXAMPLES and MAX_EXAMPLES > 0:
    dataset = dataset.select(range(min(MAX_EXAMPLES, len(dataset))))
    print(f"Limited dataset to {len(dataset)} examples")

def format_prompts(examples):
    texts = []
    for inst, out in zip(examples["instruction"], examples["output"]):
        text = f"### Instruction: {inst}\n\n### Response: {out}{tokenizer.eos_token}"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

def tokenize(examples):
    result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH)
    result["labels"] = result["input_ids"].copy()  # Add labels for training
    return result

dataset = dataset.map(tokenize, batched=True)

# Progress tracking callback
class ProgressCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, "training_progress.json")
        self.start_time = time.time()
        self.loss_history = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        elapsed = time.time() - self.start_time
        current_step = state.global_step
        total_steps = state.max_steps
        
        # Calculate ETA
        if current_step > 0:
            steps_per_sec = current_step / elapsed
            remaining_steps = total_steps - current_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
        else:
            eta_seconds = 0
        
        # Track loss
        if "loss" in logs:
            self.loss_history.append({"step": current_step, "loss": logs["loss"]})
        
        # Get GPU memory if available
        gpu_memory_used = 0
        gpu_memory_total = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        progress_data = {
            "epoch": state.epoch,
            "total_epochs": args.num_train_epochs,
            "step": current_step,
            "total_steps": total_steps,
            "loss": logs.get("loss", 0),
            "learning_rate": logs.get("learning_rate", 0),
            "elapsed_time": int(elapsed),
            "eta_seconds": int(eta_seconds),
            "samples_per_second": logs.get("samples_per_second", 0),
            "gpu_memory_used_gb": round(gpu_memory_used, 2),
            "gpu_memory_total_gb": round(gpu_memory_total, 2),
            "loss_history": self.loss_history[-50:]  # Keep last 50 points
        }
        
        # Write to file
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f)
        except Exception as e:
            print(f"Warning: Could not write progress: {e}")

print("Training...")
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=1,
        save_steps=100,
        learning_rate=5e-5,  # Much lower for instruct models! Was 2e-4
        fp16=True,
        report_to="none",  # Disable wandb
        warmup_steps=10,  # Add warmup
        lr_scheduler_type="cosine",  # Better learning rate schedule
    ),
    callbacks=[ProgressCallback(OUTPUT_DIR)]
)

trainer.train()

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Done! Model saved to {OUTPUT_DIR}")

