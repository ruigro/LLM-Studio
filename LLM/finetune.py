import argparse
import os
import sys
import json
import re
import torch
import io
import shutil
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emojis and ensure COMPLETELY UNBUFFERED output for real-time GUI updates
# CRITICAL: Force immediate flushing for every write
if sys.platform == "win32":
    # Create unbuffered TextIOWrapper - flush on every write
    class UnbufferedTextIOWrapper(io.TextIOWrapper):
        def write(self, s):
            result = super().write(s)
            self.flush()  # Force flush after every write
            return result
        def writelines(self, lines):
            result = super().writelines(lines)
            self.flush()  # Force flush after every write
            return result
    
    sys.stdout = UnbufferedTextIOWrapper(
        sys.stdout.buffer, 
        encoding="utf-8", 
        errors="replace",
        line_buffering=False
    )
    sys.stderr = UnbufferedTextIOWrapper(
        sys.stderr.buffer, 
        encoding="utf-8", 
        errors="replace",
        line_buffering=False
    )
else:
    # On Unix, ensure unbuffered output
    try:
        sys.stdout.reconfigure(line_buffering=False)
        sys.stderr.reconfigure(line_buffering=False)
    except AttributeError:
        # Python < 3.7 doesn't have reconfigure
        pass

# Create a print function that always flushes for real-time GUI updates
_original_print = print
def print(*args, **kwargs):
    """Print function that always flushes for real-time GUI updates"""
    _original_print(*args, **kwargs)
    sys.stdout.flush()
    if 'file' in kwargs and kwargs['file'] is sys.stderr:
        sys.stderr.flush()

# Check if bitsandbytes can be used
# Note: bitsandbytes 0.45.5+ fixed the triton.ops compatibility issue
# This check is kept for graceful fallback if bitsandbytes is unavailable
def check_bitsandbytes_available():
    """Check if bitsandbytes can be imported and used"""
    try:
        import bitsandbytes
        # Try a simple import to verify it works
        from bitsandbytes.nn import Linear8bitLt
        return True
    except (ImportError, ModuleNotFoundError, Exception):
        return False

BITSANDBYTES_AVAILABLE = check_bitsandbytes_available()

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

# Import unsloth at the top (as requested)
try:
    import unsloth
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    FastLanguageModel = None

# Always import transformers classes - we may need them even if unsloth is available (fallback)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# Import compatibility module for runtime capability detection
from core.model_compatibility import (
    detect_model_type,
    check_peft_capabilities,
    check_unsloth_capabilities,
    check_bitsandbytes_capabilities,
    get_compatible_peft_params,
    get_compatible_unsloth_params,
    get_optimal_loading_strategy
)


def detect_file_format(file_path: str) -> str:
    """Detect if file is JSON or JSONL format.
    Returns: 'json', 'jsonl', or 'auto'
    """
    # Check extension first
    ext = Path(file_path).suffix.lower()
    if ext == '.jsonl':
        return 'jsonl'
    if ext == '.json':
        # Try to parse as JSON first
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return 'json'
        except (json.JSONDecodeError, ValueError):
            # If JSON fails, assume JSONL
            return 'jsonl'
    # Default: try JSON first, fallback to JSONL
    return 'auto'


def load_jsonl(file_path: str, skip_errors: bool = True) -> list:
    """Load JSONL file (one JSON object per line).
    
    Args:
        file_path: Path to JSONL file
        skip_errors: If True, skip malformed lines with warning. If False, raise on first error.
    
    Returns:
        List of parsed JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                if skip_errors:
                    print(f"⚠ Warning: Skipping malformed line {line_num}: {e}")
                    continue
                else:
                    raise ValueError(f"Malformed JSON on line {line_num}: {e}") from e
    return data


def cleanup_old_adapters(adapters_dir: Path, keep_latest: int = 10):
    """
    Clean up old adapters, keeping only the N latest successful ones.
    
    Args:
        adapters_dir: Directory containing adapters
        keep_latest: Number of latest adapters to keep (default: 10)
    """
    if not adapters_dir.exists():
        return
    
    # Get all valid adapters sorted by modification time
    adapters = []
    for adapter_dir in adapters_dir.iterdir():
        if not adapter_dir.is_dir():
            continue
        
        # Check if it's a valid adapter (has required files)
        adapter_config = adapter_dir / "adapter_config.json"
        adapter_model = adapter_dir / "adapter_model.safetensors"
        if not adapter_model.exists():
            adapter_model = adapter_dir / "adapter_model.bin"
        
        if adapter_config.exists() and adapter_model.exists():
            # Get modification time from the adapter model file
            mtime = adapter_model.stat().st_mtime
            adapters.append((mtime, adapter_dir))
    
    if len(adapters) <= keep_latest:
        return  # Nothing to clean up
    
    # Sort by modification time (newest first)
    adapters.sort(reverse=True)
    
    # Remove old adapters (keep only latest N)
    removed_count = 0
    for mtime, adapter_dir in adapters[keep_latest:]:
        try:
            print(f"[CLEANUP] Removing old adapter: {adapter_dir.name}")
            shutil.rmtree(adapter_dir)
            removed_count += 1
        except Exception as e:
            print(f"[CLEANUP] Warning: Failed to remove {adapter_dir.name}: {e}")
    
    if removed_count > 0:
        print(f"[CLEANUP] Cleaned up {removed_count} old adapter(s), kept {keep_latest} latest")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--data-path", default="train_data.jsonl")
    p.add_argument("--output-dir", default="./fine_tuned")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate for training")
    p.add_argument("--max-examples", type=int, default=None, help="Limit dataset for quick runs")
    p.add_argument("--use-unsloth", action="store_true", default=True, help="Use unsloth for faster training (if available)")
    p.add_argument("--no-unsloth", dest="use_unsloth", action="store_false", help="Disable unsloth even if available")
    p.add_argument("--strict-jsonl", action="store_true", default=False,
                   help="Fail on malformed JSONL lines instead of skipping")
    return p.parse_args()


def main():
    args = parse_args()

    # Determine if we should use unsloth
    use_unsloth = HAS_UNSLOTH and args.use_unsloth
    if not use_unsloth:
        if args.use_unsloth and not HAS_UNSLOTH:
            print("[INFO] Unsloth requested but not found. Falling back to standard PEFT.")
        else:
            print("[INFO] Using standard PEFT training.")
    else:
        print("[INFO] Using Unsloth for optimized training.")

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

    print(f"Loading model and tokenizer: {MODEL_NAME}")

    # Use compatibility module to detect model type and capabilities
    model_info = detect_model_type(MODEL_NAME)
    peft_caps = check_peft_capabilities()
    unsloth_caps = check_unsloth_capabilities()
    bnb_caps = check_bitsandbytes_capabilities()
    
    # Determine optimal loading strategy
    strategy, strategy_info = get_optimal_loading_strategy(MODEL_NAME)
    
    # Configure quantization based on model requirements and capabilities
    if bnb_caps["functional"] and model_info["requires_quantization"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        bnb_config = None  # No quantization - use FP16
        if model_info["requires_quantization"] and not bnb_caps["functional"]:
            print("[WARNING] Model requires quantization but bitsandbytes is not available/functional")
            print("[WARNING] Will attempt to load base model without quantization")
            MODEL_NAME = model_info["base_model_name"]
            if MODEL_NAME != model_info["original_name"]:
                print(f"[INFO] Using base model: {MODEL_NAME}")

    # Determine if we should try unsloth based on strategy and user preference
    should_try_unsloth = (
        args.use_unsloth and 
        strategy == "unsloth" and 
        unsloth_caps["functional"] and 
        peft_caps["available"]
    )
    
    if should_try_unsloth:
        try:
            print(f"[INFO] Using Unsloth for optimized training (peft {peft_caps['version']})")
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                device_map="auto",
                quantization_config=bnb_config,
            )

            # Get version-aware parameters for unsloth
            unsloth_params = get_compatible_unsloth_params(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing="unsloth",
                capabilities=peft_caps
            )
            
            model = FastLanguageModel.get_peft_model(model, **unsloth_params)
            
            # Ensure model is in training mode and parameters require gradients
            model.train()
            for param in model.parameters():
                if param.requires_grad is False:
                    param.requires_grad = True
            
        except (TypeError, AttributeError, ImportError) as e:
            # Unsloth failed due to version incompatibility
            error_msg = str(e).lower()
            if "ensure_weight_tying" in error_msg or "unexpected keyword argument" in error_msg or "loraconfig" in error_msg:
                print(f"[WARNING] Unsloth failed due to version incompatibility: {e}")
                print(f"[WARNING] peft version {peft_caps.get('version', 'unknown')} may not support all unsloth features")
                print("[INFO] Automatically falling back to standard PEFT training (slower but compatible)")
                MODEL_NAME = model_info["base_model_name"]
                should_try_unsloth = False
            else:
                # Re-raise if it's a different error
                raise
        except Exception as e:
            # Other unsloth errors - also fall back gracefully
            print(f"[WARNING] Unsloth failed: {e}")
            print("[INFO] Automatically falling back to standard PEFT training")
            MODEL_NAME = model_info["base_model_name"]
            should_try_unsloth = False
    
    if not should_try_unsloth:
        # Standard PEFT Loading
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Try loading with quantization first, fallback to FP16 if bitsandbytes fails
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        # Only add quantization_config if it's not None
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        else:
            # Use FP16 when quantization is not available
            load_kwargs["torch_dtype"] = torch.float16
            print("[WARNING] bitsandbytes not available - using FP16 (requires more VRAM)")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
            # Prepare for k-bit training (only if using quantization)
            if BITSANDBYTES_AVAILABLE and bnb_config is not None:
                model = prepare_model_for_kbit_training(model)
        except (RuntimeError, AttributeError, ImportError, ModuleNotFoundError) as e:
            error_str = str(e).lower()
            if "triton.ops" in error_str or "bitsandbytes" in error_str or "quantization" in error_str:
                print("[WARNING] bitsandbytes/quantization failed")
                print("[WARNING] Falling back to FP16 (requires more VRAM)")
                # Retry without quantization - try base model if quantization model was used
                load_kwargs.pop("quantization_config", None)
                load_kwargs["torch_dtype"] = torch.float16
                try_model_name = MODEL_NAME
                if "-bnb-" in MODEL_NAME or "-4bit" in MODEL_NAME or "-8bit" in MODEL_NAME:
                    base_name = MODEL_NAME.replace("-bnb-4bit", "").replace("-bnb-8bit", "").replace("-4bit", "").replace("-8bit", "")
                    if base_name != MODEL_NAME:
                        print(f"[INFO] Trying base model: {base_name}")
                        try_model_name = base_name
                model = AutoModelForCausalLM.from_pretrained(try_model_name, **load_kwargs)
            else:
                raise
        
        # Configure LoRA with version-aware parameters
        if not peft_caps["available"]:
            raise RuntimeError("PEFT is required but not available. Please install peft.")
        
        peft_params = get_compatible_peft_params(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
            capabilities=peft_caps
        )
        
        peft_config = LoraConfig(**peft_params)
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        
        # Ensure model is in training mode and parameters require gradients
        model.train()
        for param in model.parameters():
            if param.requires_grad is False:
                param.requires_grad = True
        
        # Enable gradient checkpointing if supported
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    print("Preparing dataset...")
    
    # Smart dataset loader - handles JSON and JSONL automatically
    import tempfile
    
    file_format = detect_file_format(DATASET_PATH)
    
    if file_format == 'jsonl' or (file_format == 'auto' and DATASET_PATH.endswith('.jsonl')):
        print(f"✓ Detected JSONL format")
        raw_data = load_jsonl(DATASET_PATH, skip_errors=not args.strict_jsonl)
    else:
        # Standard JSON loading
        print(f"✓ Detected JSON format")
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    
    # Step 1: Convert to list if needed
    if isinstance(raw_data, dict):
        # Try common keys that contain the actual data
        for key in ['data', 'examples', 'train', 'dataset', 'items', 'conversations', 'entries']:
            if key in raw_data and isinstance(raw_data[key], list):
                raw_data = raw_data[key]
                print(f"✓ Extracted data from '{key}' field")
                break
        else:
            # If still a dict, treat as single example
            raw_data = [raw_data]
            print("✓ Converted single dict to list")
    
    if not isinstance(raw_data, list):
        raise ValueError(f"Dataset must be a list or dict with data field. Got: {type(raw_data)}")
    
    print(f"✓ Found {len(raw_data)} examples")
    
    # Step 2: Normalize field names (handle various formats)
    normalized_data = []
    
    # Detect format from first item
    if len(raw_data) > 0:
        first = raw_data[0]
        
        # Determine instruction/output field names
        instruction_key = None
        output_key = None
        
        # Common field name mappings (including customer_message/assistant_response)
        instruction_fields = ['instruction', 'prompt', 'input', 'question', 'query', 'text', 'user', 'human', 
                              'customer_message', 'customer', 'message', 'query_text']
        output_fields = ['output', 'response', 'completion', 'answer', 'reply', 'assistant', 'gpt', 'bot',
                        'assistant_response', 'assistant', 'response_text', 'answer_text']
        
        # Find matching fields
        for key in instruction_fields:
            if key in first:
                instruction_key = key
                break
        
        for key in output_fields:
            if key in first:
                output_key = key
                break
        
        # Handle chat/messages format (like ShareGPT, OpenAI)
        if 'messages' in first or 'conversations' in first:
            print("✓ Detected chat/messages format")
            msg_key = 'messages' if 'messages' in first else 'conversations'
            for item in raw_data:
                messages = item[msg_key]
                # Extract user/assistant pairs
                instruction = ""
                output = ""
                for msg in messages:
                    role = msg.get('role', msg.get('from', '')).lower()
                    content = msg.get('content', msg.get('value', ''))
                    if role in ['user', 'human']:
                        instruction = content
                    elif role in ['assistant', 'gpt', 'bot']:
                        output = content
                if instruction and output:
                    normalized_data.append({'instruction': instruction, 'output': output})
        
        # Handle standard formats
        elif instruction_key and output_key:
            print(f"✓ Detected format: '{instruction_key}' -> '{output_key}'")
            for item in raw_data:
                normalized_data.append({
                    'instruction': str(item.get(instruction_key, '')),
                    'output': str(item.get(output_key, ''))
                })
        
        # Handle Alpaca format with optional input field
        elif 'instruction' in first:
            print("✓ Detected Alpaca format (instruction + optional input)")
            for item in raw_data:
                instruction = item.get('instruction', '')
                inp = item.get('input', '')
                output = item.get('output', item.get('response', ''))
                # Combine instruction and input if present
                if inp:
                    full_instruction = f"{instruction}\n\nInput: {inp}"
                else:
                    full_instruction = instruction
                normalized_data.append({'instruction': full_instruction, 'output': output})
        
        else:
            raise ValueError(f"Could not detect dataset format. First item keys: {list(first.keys())}\n"
                           f"Expected one of: {instruction_fields} -> {output_fields}, or 'messages' format")
    
    if not normalized_data:
        raise ValueError("No valid examples found in dataset")
    
    print(f"✓ Normalized {len(normalized_data)} examples")
    
    # Step 3: Write to JSONL format for reliable loading
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
    for item in normalized_data:
        temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
    temp_file.close()
    
    # Step 4: Load with HuggingFace datasets
    dataset = load_dataset("json", data_files=temp_file.name, split="train")
    print(f"✓ Loaded dataset with {len(dataset)} examples")
    
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    def formatting_func(example):
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return {"text": text}

    dataset = dataset.map(formatting_func, remove_columns=dataset.column_names)

    print("Starting training...")
    
    # Configure tqdm for real-time progress updates in GUI
    # Set environment variable to ensure tqdm flushes immediately
    os.environ["TQDM_DISABLE"] = "0"
    os.environ["TQDM_MININTERVAL"] = "0.1"  # Update at least every 0.1 seconds
    
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

    # Train and capture training state
    print("[INFO] Starting training loop...")
    train_result = trainer.train()
    
    # Verify training actually happened
    if train_result.metrics:
        print(f"[INFO] Training completed successfully!")
        print(f"[INFO] Final metrics: {train_result.metrics}")
        if 'train_loss' in train_result.metrics:
            print(f"[INFO] Final training loss: {train_result.metrics['train_loss']:.4f}")
        if 'train_runtime' in train_result.metrics:
            print(f"[INFO] Training runtime: {train_result.metrics['train_runtime']:.2f} seconds")
    else:
        print("[WARNING] Training completed but no metrics were recorded!")
        print("[WARNING] This might indicate training did not actually run.")

    print("Saving LoRA adapter...")

    # Ensure output directory exists (should be fine_tuned/)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Cleanup old adapters before saving new one (keep latest 10)
    cleanup_old_adapters(Path(OUTPUT_DIR), keep_latest=10)
    
    # Generate meaningful adapter name: base_model-task-version
    # e.g., "gemma-2-2b-it-beauty-v1"
    base_slug = MODEL_NAME.replace("/", "-").replace("__", "-").replace("_", "-")
    # Extract task name from dataset filename
    dataset_name = Path(DATASET_PATH).stem
    # Clean up dataset name
    task_name = dataset_name.replace("_dataset", "").replace("_finetune", "").replace("_", "-")
    if not task_name or task_name == dataset_name:
        # Fallback: use generic name
        task_name = "custom"
    
    # Find next version number for this base_model-task combination
    version = 1
    while True:
        adapter_name = f"{base_slug}-{task_name}-v{version}"
        adapter_path = Path(OUTPUT_DIR) / adapter_name
        if not adapter_path.exists():
            break
        version += 1
    
    adapter_path.mkdir(parents=True, exist_ok=True)
    
    # Save adapter (this saves adapter_config.json and adapter_model.safetensors/.bin)
    model.save_pretrained(str(adapter_path))
    
    # Remove tokenizer files - base model already has them, no need to duplicate
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json", 
        "merges.txt", "special_tokens_map.json", "tokenizer.model"
    ]
    for tokenizer_file in tokenizer_files:
        tokenizer_path = adapter_path / tokenizer_file
        if tokenizer_path.exists():
            tokenizer_path.unlink()
    
    # Save training metadata
    training_info = {
        "base_model": MODEL_NAME,
        "dataset": str(DATASET_PATH),
        "training_params": {
            "epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
        },
        "created_at": datetime.now().isoformat(),
        "train_result": {
            "train_loss": train_result.metrics.get("train_loss", None) if train_result.metrics else None,
            "train_runtime": train_result.metrics.get("train_runtime", None) if train_result.metrics else None,
        } if train_result.metrics else None
    }
    
    with open(adapter_path / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2)
    
    # Verify adapter files were saved
    adapter_config = adapter_path / "adapter_config.json"
    adapter_model = adapter_path / "adapter_model.safetensors"
    if not adapter_model.exists():
        adapter_model = adapter_path / "adapter_model.bin"
    
    if adapter_config.exists() and adapter_model.exists():
        adapter_size = adapter_model.stat().st_size
        print(f"[INFO] ✓ LoRA adapter saved successfully ({adapter_size / 1024 / 1024:.2f} MB)")
        print(f"[INFO] ✓ Adapter location: {adapter_path}")
    else:
        print(f"[WARNING] LoRA adapter files not found! Check {adapter_path}")
        if not adapter_config.exists():
            print(f"[ERROR] Missing: {adapter_config}")
        if not adapter_model.exists():
            print(f"[ERROR] Missing: {adapter_model}")
    
    print(f"Finetuning complete! Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()