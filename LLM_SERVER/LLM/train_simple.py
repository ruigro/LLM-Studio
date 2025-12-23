#!/usr/bin/env python3
"""
Simple training script - just run it directly.
No GUI, no threads, no complications.
"""
import sys
sys.path.insert(0, '.')

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Configuration
MODEL_NAME = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
DATA_PATH = "train_data.jsonl"
OUTPUT_DIR = "./fine_tuned_adapter"
EPOCHS = 1

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print("Configuring LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_prompts(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        text = f"""### Instruction:
{instruction}

### Response:
{output}"""
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

print("Starting training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        fp16=not FastLanguageModel.is_bfloat16_supported(),
        bf16=FastLanguageModel.is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
    ),
)

trainer.train()

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete!")
print(f"Model saved to: {OUTPUT_DIR}")

