#!/usr/bin/env python3
"""
Simple training WITHOUT unsloth - uses standard PEFT/LoRA
This will work.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import sys
import argparse
import json
import os

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
        learning_rate=2e-4,
        fp16=True,
        report_to="none",  # Disable wandb
    ),
)

trainer.train()

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Done! Model saved to {OUTPUT_DIR}")

