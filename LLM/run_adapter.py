#!/usr/bin/env python3
"""Run a fine-tuned LoRA adapter with a base model (supports 4-bit + offload).

Usage examples:
    python run_adapter.py --adapter-dir ./fine_tuned_adapter/M1Checkpoint1 --prompt "Say hello"
  python run_adapter.py --base-model unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit --prompt "Hello" --max-new-tokens 128
"""
import argparse
import torch
import warnings
import sys
import platform

# Suppress known warnings
warnings.filterwarnings("ignore", message=".*quantization_config.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Check if running on Windows
IS_WINDOWS = platform.system() == "Windows"

# Check for optional dependencies early
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
except ImportError as e:
    print(f"❌ Missing required package: {e}", file=sys.stderr)
    print("Please run: pip install transformers peft", file=sys.stderr)
    sys.exit(1)

# Optional: import weave to enable W&B Weave tracing if installed
try:
    import weave  # type: ignore
    print("Weave imported in run_adapter: LLM call tracing enabled (local).")
except Exception:
    pass


def load_model(base_model, adapter_dir, use_4bit=True, offload=True):
    import os
    
    tokenizer = None
    
    # If adapter_dir is None, we're loading base model only
    if adapter_dir is None:
        print(f"[INFO] Loading base model only (no adapter): {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        print("[OK] Tokenizer loaded from base model")
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"[INFO] Set pad_token to eos_token: {tokenizer.eos_token}")
        
        # Load base model
        # On Windows, bitsandbytes is unreliable - use FP16 instead
        if use_4bit and not IS_WINDOWS:
            print("[INFO] Loading with 4-bit quantization (non-Windows)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                )
            except ImportError as e:
                error_str = str(e).lower()
                if "timm" in error_str:
                    print(f"❌ Error: Vision model requires 'timm' package", file=sys.stderr)
                    print("Please run: pip install timm>=0.9.0", file=sys.stderr)
                    raise RuntimeError("Missing dependency: timm. Install with: pip install timm>=0.9.0")
                elif "einops" in error_str:
                    print(f"❌ Error: Model requires 'einops' package", file=sys.stderr)
                    print("Please run: pip install einops>=0.6.0", file=sys.stderr)
                    raise RuntimeError("Missing dependency: einops. Install with: pip install einops>=0.6.0")
                elif "open_clip" in error_str or "open-clip" in error_str:
                    print(f"❌ Error: CLIP-based model requires 'open-clip-torch' package", file=sys.stderr)
                    print("Please run: pip install open-clip-torch>=2.20.0", file=sys.stderr)
                    raise RuntimeError("Missing dependency: open-clip-torch. Install with: pip install open-clip-torch>=2.20.0")
                raise
            except Exception:
                # Fallback to non-quantized
                print("[WARN] 4-bit loading failed, falling back to FP16")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
        else:
            # Windows or non-4bit: use FP16 on CUDA
            if IS_WINDOWS:
                print("[INFO] Windows detected - loading with FP16 (bitsandbytes disabled)")
            else:
                print("[INFO] Loading without quantization")
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            except ImportError as e:
                error_str = str(e).lower()
                if "timm" in error_str:
                    print(f"❌ Error: Vision model requires 'timm' package", file=sys.stderr)
                    print("Please run: pip install timm>=0.9.0", file=sys.stderr)
                    raise RuntimeError("Missing dependency: timm. Install with: pip install timm>=0.9.0")
                elif "einops" in error_str:
                    print(f"❌ Error: Model requires 'einops' package", file=sys.stderr)
                    print("Please run: pip install einops>=0.6.0", file=sys.stderr)
                    raise RuntimeError("Missing dependency: einops. Install with: pip install einops>=0.6.0")
                elif "open_clip" in error_str or "open-clip" in error_str:
                    print(f"❌ Error: CLIP-based model requires 'open-clip-torch' package", file=sys.stderr)
                    print("Please run: pip install open-clip-torch>=2.20.0", file=sys.stderr)
                    raise RuntimeError("Missing dependency: open-clip-torch. Install with: pip install open-clip-torch>=2.20.0")
                raise
        
        return tokenizer, model
    
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


def generate_text(tokenizer, model, prompt, max_new_tokens=128, temperature=0.7, model_type="base"):
    """Generate text with proper formatting based on model type
    
    Args:
        tokenizer: The tokenizer
        model: The model
        prompt: The user's input text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        model_type: "instruct" or "base" - determines prompt formatting
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Format prompt based on model type
    if model_type == "instruct":
        # Use chat template for instruct models
        messages = [
            {"role": "system", "content": ""},  # Empty but included for consistency
            {"role": "user", "content": prompt}
        ]
        try:
            # Use tokenizer's built-in chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback if tokenizer doesn't have chat template
            print(f"[WARN] Chat template not available: {e}. Using plain prompt.")
            formatted_prompt = prompt
    else:
        # Base model - use plain prompt
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
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
        try:
            out = model.generate(**inputs, **gen_kwargs)
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    # Return only newly generated tokens (exclude prompt)
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    # Debug: Check if output is empty
    if not text or not text.strip():
        print(f"[WARN] Empty generation. Input length: {input_len}, Output length: {len(out[0])}, New tokens: {len(gen_ids)}")
    
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
    p.add_argument("--no-adapter", action="store_true", help="Load only base model without adapter")
    p.add_argument("--model-type", default="base", choices=["base", "instruct"], help="Model type: base or instruct (affects prompt formatting)")
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
    # If no-adapter flag is set, pass None as adapter_dir
    adapter_dir = None if args.no_adapter else args.adapter_dir
    tokenizer, model = load_model(args.base_model, adapter_dir, use_4bit=args.use_4bit, offload=args.offload)

    # Generate and print only the new text (no prompt echo)
    print("[INFO] Generating...")
    out = generate_text(tokenizer, model, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, model_type=args.model_type)
    
    if out and out.strip():
        print("\n--- OUTPUT ---\n")
        print(out)
    else:
        print("[ERROR] No output generated. Model may have failed to produce text.")
        print("[DEBUG] Check CUDA availability and model loading above.")


if __name__ == "__main__":
    main()
