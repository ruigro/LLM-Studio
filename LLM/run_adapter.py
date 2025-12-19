#!/usr/bin/env python3
"""Run a fine-tuned LoRA adapter with a base model (supports 4-bit + offload).

Usage examples:
    python run_adapter.py --adapter-dir ./fine_tuned_adapter/M1Checkpoint1 --prompt "Say hello"
  python run_adapter.py --base-model unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit --prompt "Hello" --max-new-tokens 128
"""
import argparse
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Suppress known warnings
warnings.filterwarnings("ignore", message=".*quantization_config.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")


# Optional: import weave to enable W&B Weave tracing if installed
try:
    import weave  # type: ignore
    print("Weave imported in run_adapter: LLM call tracing enabled (local).")
except Exception:
    pass


def load_model(base_model, adapter_dir, use_4bit=True, offload=True):
    import os
    
    tokenizer = None
    
    # Check if adapter_dir is a checkpoint subdirectory and use parent if so
    if "checkpoint-" in adapter_dir and os.path.basename(adapter_dir).startswith("checkpoint-"):
        print(f"[INFO] Detected checkpoint subdirectory, using parent: {os.path.dirname(adapter_dir)}")
        adapter_dir = os.path.dirname(adapter_dir)
    
    # Try loading tokenizer from adapter dir first, then base model
    try:
        print(f"[INFO] Loading tokenizer from adapter dir: {adapter_dir}")
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        print("[OK] Tokenizer loaded from adapter dir")
    except Exception as e:
        print(f"[WARN] Could not load tokenizer from adapter dir: {e}")
        print(f"[INFO] Loading tokenizer from base model: {base_model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            print("[OK] Tokenizer loaded from base model")
        except Exception as e2:
            raise RuntimeError(f"Failed to load tokenizer from both adapter dir and base model: {e2}")
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[INFO] Set pad_token to eos_token: {tokenizer.eos_token}")

    # Load base model without device_map to avoid accelerate compatibility issues
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        # Check if base model already has quantization - if so, just load it
        try:
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        except Exception:
            # Fallback: model already quantized, load without config
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
            )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
        # Move to GPU if available and not quantized
        if torch.cuda.is_available():
            base = base.to("cuda")

    # Attach adapter (PEFT) to base model if adapter files exist; otherwise
    # treat adapter_dir as a merged model and load directly.
    try:
        # Prefer local-only loading to avoid treating the path as an HF repo id
        model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
        return tokenizer, model
    except Exception:
        # Fall back: if adapter_dir is actually a merged model directory, try to load it.
        try:
            # If the merged folder lacks a config, try to fetch config from the provided base_model
            import os
            from transformers import AutoConfig

            config_path = os.path.join(adapter_dir, "config.json")
            if not os.path.exists(config_path):
                # Attempt to read a README front-matter to discover base_model
                base = base_model
                readme_path = os.path.join(adapter_dir, "README.md")
                try:
                    if os.path.exists(readme_path):
                        with open(readme_path, "r", encoding="utf-8") as f:
                            txt = f.read()
                        # crude parse for 'base_model: <name>' in the YAML frontmatter
                        for line in txt.splitlines():
                            line = line.strip()
                            if line.startswith("base_model:"):
                                base = line.split("base_model:", 1)[1].strip()
                                break
                except Exception:
                    pass

                # If we have a base model name, download its config and save locally
                try:
                    cfg = AutoConfig.from_pretrained(base, trust_remote_code=True)
                    cfg.save_pretrained(adapter_dir)
                except Exception:
                    # ignore and try loading directly; the later call will raise a clear error
                    pass

            # Try loading merged model (adapter dir contains full safetensors checkpoint)
            # Prefer CPU-only load to avoid meta-tensor and accelerate version issues
            try:
                merged = AutoModelForCausalLM.from_pretrained(
                    adapter_dir,
                    device_map=None,  # Load on CPU
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                )
                # Move to GPU if available
                if torch.cuda.is_available():
                    merged = merged.to("cuda")
                return tokenizer, merged
            except Exception as e:
                # Provide clearer guidance for common environment mismatches
                raise RuntimeError(
                    f"Failed to load merged model from '{adapter_dir}': {e}\n"
                    "Common fixes: Ensure the model checkpoint is complete and compatible with your transformers version.\n"
                )
        except Exception as e:
            # Re-raise original error with context
            raise RuntimeError(f"Failed to load PEFT adapter or merged model from '{adapter_dir}': {e}")


def generate_text(tokenizer, model, prompt, max_new_tokens=128, temperature=0.7):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Only add sampling parameters if temperature > 0
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False
    
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    
    # Return only newly generated tokens (exclude prompt)
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", default="./fine_tuned_adapter/M1Checkpoint1", help="Path to saved LoRA adapter checkpoint")
    p.add_argument("--base-model", default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit", help="Base model name or path")
    p.add_argument("--prompt", default="### Instruction:\nSay hello\n\n### Response:\n", help="Prompt to generate from")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--no-4bit", dest="use_4bit", action="store_false", help="Disable 4-bit quantization (requires more memory)")
    p.add_argument("--no-offload", dest="offload", action="store_false", help="Disable CPU offload for fp32 parts")
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    # convert literal "\n" sequences into real newlines (allows shell-friendly prompts)
    prompt = args.prompt.replace("\\n", "\n").strip()

    # Set UTF-8 encoding for Windows console
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    print("[INFO] Loading tokenizer and model (this may take a few minutes)...")
    tokenizer, model = load_model(args.base_model, args.adapter_dir, use_4bit=args.use_4bit, offload=args.offload)

    # Generate and print only the new text (no prompt echo)
    print("[INFO] Generating...")
    out = generate_text(tokenizer, model, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print("\n--- OUTPUT ---\n")
    print(out)


if __name__ == "__main__":
    main()
