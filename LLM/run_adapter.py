#!/usr/bin/env python3
"""Run a fine-tuned LoRA adapter with a base model (supports 4-bit + offload).

Usage examples:
    python run_adapter.py --adapter-dir ./fine_tuned/gemma-2-2b-it-custom-v1 --prompt "Say hello"
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
    import sys
    
    # Validate inputs are strings
    if not isinstance(base_model, str) or not base_model:
        raise ValueError(f"base_model must be a non-empty string, got: {type(base_model)} = {base_model!r}")
    
    if adapter_dir is not None and (not isinstance(adapter_dir, str) or not adapter_dir):
        raise ValueError(f"adapter_dir must be None or a non-empty string, got: {type(adapter_dir)} = {adapter_dir!r}")
    
    tokenizer = None
    
    # Normalize base_model path early - ensure it's a proper string
    import os
    from pathlib import Path
    
    # Check if it looks like a local file path (contains path separators or starts with drive letter on Windows)
    is_local_path = os.sep in base_model or (os.name == 'nt' and len(base_model) > 1 and base_model[1] == ':')
    
    if is_local_path:
        # It's a local path - normalize it
        model_path = Path(base_model).resolve()
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {base_model}")
        if not model_path.is_dir():
            raise ValueError(f"Model path is not a directory: {base_model}")
        # Check for essential model files
        config_file = model_path / "config.json"
        if not config_file.exists():
            raise ValueError(f"Model directory missing config.json: {base_model}")
        base_model = str(model_path)
    
    # Final validation - ensure it's a clean string
    model_path_str = str(base_model).strip()
    if not model_path_str:
        raise ValueError("base_model path is empty after normalization")
    
    # If adapter_dir is None, we're loading base model only
    if adapter_dir is None:
        if not is_local_path:
            print(f"[INFO] Loading base model from HuggingFace: {model_path_str}", file=sys.stderr)
        else:
            print(f"[INFO] Loading local base model: {model_path_str}", file=sys.stderr)
        
        try:
            # Use the normalized string path
            tokenizer = AutoTokenizer.from_pretrained(model_path_str)
        except Exception as e:
            error_msg = str(e)
            error_lower = error_msg.lower()
            if "not a string" in error_lower:
                # This is a transformers library error - provide more context
                import traceback
                print(f"[DEBUG] Full traceback:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                raise ValueError(
                    f"Transformers library error: 'not a string' when loading model.\n"
                    f"Path: {model_path_str!r}\n"
                    f"Path type: {type(model_path_str)}\n"
                    f"Path length: {len(model_path_str)}\n"
                    f"Path exists: {os.path.exists(model_path_str) if is_local_path else 'N/A (HF model ID)'}\n"
                    f"Path is directory: {os.path.isdir(model_path_str) if is_local_path else 'N/A'}\n"
                    f"Original error: {error_msg}\n"
                    f"This may indicate a corrupted model, missing files, or incompatible transformers version."
                )
            raise
        print(f"[OK] Tokenizer loaded from base model", file=sys.stderr)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"[INFO] Set pad_token to eos_token: {tokenizer.eos_token}", file=sys.stderr)
        
        # Load base model
        # On Windows, bitsandbytes is unreliable - use FP16 instead
        if use_4bit and not IS_WINDOWS:
            print(f"[INFO] Loading with 4-bit quantization (non-Windows)", file=sys.stderr)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path_str,
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
                print("[WARN] 4-bit loading failed, falling back to FP16", file=sys.stderr)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path_str,
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
                    model_path_str,
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
    
    # Check if adapter_dir exists and has required files
    import os
    if not os.path.exists(adapter_dir):
        raise RuntimeError(f"Adapter directory not found: {adapter_dir}")
    
    # Check for adapter files (adapter_model.safetensors, adapter_model.bin, or pytorch_model.bin)
    adapter_files = ["adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"]
    has_adapter_files = any(os.path.exists(os.path.join(adapter_dir, f)) for f in adapter_files)
    
    if not has_adapter_files:
        print(f"[ERROR] Adapter directory '{adapter_dir}' exists but contains no adapter weights!")
        print(f"[ERROR] Expected files: {', '.join(adapter_files)}")
        print(f"[INFO] Directory contents: {os.listdir(adapter_dir)}")
        raise RuntimeError(
            f"Incomplete adapter checkpoint at '{adapter_dir}'.\n"
            f"The directory exists but contains no model weights.\n"
            f"Please complete the training or select a different checkpoint."
        )
    
    # Try loading tokenizer from adapter dir first, then base model
    try:
        print(f"[INFO] Loading tokenizer from adapter dir: {adapter_dir}", file=sys.stderr)
        if not isinstance(adapter_dir, str):
            raise ValueError(f"adapter_dir must be a string, got: {type(adapter_dir)} = {adapter_dir!r}")
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        print("[OK] Tokenizer loaded from adapter dir", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Could not load tokenizer from adapter dir: {e}", file=sys.stderr)
        print(f"[INFO] Loading tokenizer from base model: {model_path_str}", file=sys.stderr)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path_str)
            print("[OK] Tokenizer loaded from base model", file=sys.stderr)
        except Exception as e2:
            error_msg = str(e2)
            if "not a string" in error_msg.lower():
                raise ValueError(f"Invalid model path (not a string): base_model={model_path_str!r} (type: {type(model_path_str)}), adapter_dir={adapter_dir!r} (type: {type(adapter_dir)})")
            raise RuntimeError(f"Failed to load tokenizer from both adapter dir and base model: {e2}")
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[INFO] Set pad_token to eos_token: {tokenizer.eos_token}", file=sys.stderr)

    # Load base model without device_map to avoid accelerate compatibility issues
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        # Check if base model already has quantization - if so, just load it
        try:
            base = AutoModelForCausalLM.from_pretrained(
                model_path_str,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        except Exception:
            # Fallback: model already quantized, load without config
            base = AutoModelForCausalLM.from_pretrained(
                model_path_str,
                trust_remote_code=True,
            )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_path_str,
            trust_remote_code=True,
        )
        # Move to GPU if available and not quantized
        if torch.cuda.is_available():
            base = base.to("cuda")
            print(f"[INFO] Moved base model to GPU", file=sys.stderr)

    # Attach adapter (PEFT) to base model if adapter files exist; otherwise
    # treat adapter_dir as a merged model and load directly.
    try:
        # Prefer local-only loading to avoid treating the path as an HF repo id
        model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
        print(f"[OK] Adapter attached successfully", file=sys.stderr)
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
                    print(f"[OK] Fetched and saved config for merged model", file=sys.stderr)
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
                    print(f"[INFO] Moved merged model to GPU", file=sys.stderr)
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


def generate_text(tokenizer, model, prompt, max_new_tokens=128, temperature=0.7, model_type="base", system_prompt=""):
    """Generate text with proper formatting based on model type
    
    Args:
        tokenizer: The tokenizer
        model: The model
        prompt: The user's input text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        model_type: "instruct" or "base" - determines prompt formatting
        system_prompt: Optional system prompt for instruct models
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Format prompt based on model type
    if model_type == "instruct":
        # Use chat template for instruct models
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Use tokenizer's built-in chat template
            # For most models, the template includes the BOS token if needed
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback if tokenizer doesn't have chat template - manually format
            import sys
            print(f"[WARN] Chat template not available: {e}. Using manual formatting.", file=sys.stderr)
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                formatted_prompt = prompt
    else:
        # Base model - use plain prompt, but prepend system prompt if provided
        if system_prompt:
            formatted_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            formatted_prompt = prompt
    
    # For instruct models using chat templates, we often want add_special_tokens=False
    # because the template (like Gemma, Llama 3) already includes the BOS token.
    # For base models, we almost always want add_special_tokens=True.
    add_specials = True
    if model_type == "instruct":
        # Check if the formatted prompt already starts with a common BOS token or template start
        # This is a heuristic to avoid double BOS tokens.
        if formatted_prompt.startswith("<|begin_of_text|>") or formatted_prompt.startswith("<bos>") or formatted_prompt.startswith("<s>"):
            add_specials = False
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=add_specials)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    input_len = inputs["input_ids"].shape[1]
    
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
            print(f"[ERROR] Generation failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return ""
    
    # Return only newly generated tokens (exclude prompt)
    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    # If the model still somehow included the prompt, remove it
    # We do a case-insensitive check and also strip whitespace
    text_clean = text.strip()
    
    # Debug: Check if output is empty
    if not text_clean:
        print(f"[WARN] Empty generation. Input tokens: {input_len}, Total tokens: {len(out[0])}, New tokens: {len(gen_ids)}", file=sys.stderr)
    
    return text_clean


def main():
    import sys
    try:
        p = argparse.ArgumentParser()
        p.add_argument("--adapter-dir", default="./fine_tuned", help="Path to saved LoRA adapter directory (or specific adapter subdirectory)")
        p.add_argument("--base-model", default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit", help="Base model name or path")
        p.add_argument("--prompt", default="### Instruction:\nSay hello\n\n### Response:\n", help="Prompt to generate from")
        p.add_argument("--max-new-tokens", type=int, default=128)
        p.add_argument("--no-4bit", dest="use_4bit", action="store_false", help="Disable 4-bit quantization (requires more memory)")
        p.add_argument("--no-offload", dest="offload", action="store_false", help="Disable CPU offload for fp32 parts")
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--no-adapter", action="store_true", help="Load only base model without adapter")
        p.add_argument("--model-type", default="base", choices=["base", "instruct"], help="Model type: base or instruct (affects prompt formatting)")
        p.add_argument("--system-prompt", default="", help="System prompt for instruct models")
        args = p.parse_args()

        # convert literal "\n" sequences into real newlines (allows shell-friendly prompts)
        prompt = args.prompt.replace("\\n", "\n").strip()
        system_prompt = args.system_prompt.replace("\\n", "\n").strip() if args.system_prompt else ""

        # Set UTF-8 encoding for Windows console (do this first before any prints)
        if sys.platform == 'win32':
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        
        # Validate and normalize that base_model is a string
        if not isinstance(args.base_model, str):
            # Try to convert to string if it's not None
            if args.base_model is None:
                print(f"[ERROR] base_model argument is None", file=sys.stderr)
                print("\n--- OUTPUT ---\n", file=sys.stdout)
                print(f"[ERROR] Model loading failed: base model path is None", file=sys.stdout)
                sys.exit(1)
            else:
                args.base_model = str(args.base_model)
        
        if not args.base_model or not args.base_model.strip():
            print(f"[ERROR] base_model argument is empty: {args.base_model!r}", file=sys.stderr)
            print("\n--- OUTPUT ---\n", file=sys.stdout)
            print(f"[ERROR] Model loading failed: base model path is empty", file=sys.stdout)
            sys.exit(1)
        
        # Normalize base_model (remove any extra whitespace)
        args.base_model = args.base_model.strip()
        
        print(f"[INFO] Loading tokenizer and model (this may take a few minutes)...", file=sys.stderr)
        # If no-adapter flag is set, pass None as adapter_dir
        adapter_dir = None if args.no_adapter else args.adapter_dir
        
        # Validate and normalize adapter_dir if provided
        if adapter_dir is not None:
            if not isinstance(adapter_dir, str):
                # Try to convert to string
                adapter_dir = str(adapter_dir)
            if not adapter_dir or not adapter_dir.strip():
                print(f"[ERROR] adapter_dir argument is empty: {adapter_dir!r}", file=sys.stderr)
                print("\n--- OUTPUT ---\n", file=sys.stdout)
                print(f"[ERROR] Model loading failed: adapter directory path is empty", file=sys.stdout)
                sys.exit(1)
            adapter_dir = adapter_dir.strip()
        
        try:
            tokenizer, model = load_model(args.base_model, adapter_dir, use_4bit=args.use_4bit, offload=args.offload)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            print("\n--- OUTPUT ---\n", file=sys.stdout)
            print(f"[ERROR] Model loading failed: {str(e)}", file=sys.stdout)
            sys.exit(1)

        # Generate and print only the new text (no prompt echo)
        print(f"[INFO] Generating...", file=sys.stderr)
        try:
            out = generate_text(tokenizer, model, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, model_type=args.model_type, system_prompt=system_prompt)
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            print("\n--- OUTPUT ---\n", file=sys.stdout)
            print(f"[ERROR] Text generation failed: {str(e)}", file=sys.stdout)
            sys.exit(1)
        
        if out and out.strip():
            # Final clean of the output before printing to stdout
            # This is a safety measure to remove any prompt that might have leaked
            clean_out = out
            if prompt.lower() in out.lower():
                # Be more aggressive: replace the actual prompt text
                # Find the prompt in the output (case-insensitive)
                import re
                escaped_prompt = re.escape(prompt)
                clean_out = re.sub(escaped_prompt, "", out, flags=re.IGNORECASE).strip()
            
            print("\n--- OUTPUT ---\n", file=sys.stdout)
            print(clean_out, file=sys.stdout)
            sys.stdout.flush()
        else:
            print(f"[ERROR] No output generated. Model may have failed to produce text.", file=sys.stderr)
            print(f"[DEBUG] Check CUDA availability and model loading above.", file=sys.stderr)
            print("\n--- OUTPUT ---\n", file=sys.stdout)
            print(f"[ERROR] No text was generated. The model may have failed silently.", file=sys.stdout)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("\n--- OUTPUT ---\n", file=sys.stdout)
        print(f"[ERROR] Unexpected error occurred: {str(e)}", file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
