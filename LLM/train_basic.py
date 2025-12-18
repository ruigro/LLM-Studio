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

# Config
MODEL_NAME = "./models/unsloth_llama-3.1-8b-instruct-unsloth-bnb-4bit"  # Use local model
DATA_PATH = "train_data.jsonl"
OUTPUT_DIR = "./fine_tuned_adapter"
EPOCHS = 1

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
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_prompts(examples):
    texts = []
    for inst, out in zip(examples["instruction"], examples["output"]):
        text = f"### Instruction: {inst}\n\n### Response: {out}{tokenizer.eos_token}"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

def tokenize(examples):
    result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
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
        per_device_train_batch_size=1,
        logging_steps=1,
        save_steps=100,
        learning_rate=2e-4,
        fp16=True,
    ),
)

trainer.train()

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Done! Model saved to {OUTPUT_DIR}")

