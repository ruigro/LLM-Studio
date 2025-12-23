#!/usr/bin/env python3
"""Multi-functional workflow script for finetuning, validation, and inference.

Usage:
  python workflow.py train [options]
  python workflow.py validate [options]
  python workflow.py run --prompt "Your prompt here" [options]
  python workflow.py all [options]  # Run train -> validate -> run pipeline
"""
import argparse
import sys
import os
import subprocess
from pathlib import Path


# Optional: import weave to enable W&B Weave tracing if installed
try:
    import weave  # type: ignore
    print("Weave imported in workflow: LLM call tracing enabled (local).")
except Exception:
    pass


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    # Prepare environment: default to offline W&B unless enabled at runtime
    env = os.environ.copy()
    if not getattr(run_command, "enable_wandb", False):
        env.setdefault("WANDB_MODE", "offline")
    else:
        env.pop("WANDB_MODE", None)

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n✅ {description} completed successfully")
    return result.returncode


def train(args):
    """Run finetuning training."""
    cmd = [
        "python", "finetune.py",
        "--data-path", args.data_path,
        "--output-dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    
    if args.max_examples:
        cmd.extend(["--max-examples", str(args.max_examples)])
    
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])
    
    run_command(cmd, "Training (Finetuning)")


def validate(args):
    """Run validation on trained model."""
    cmd = [
        "python", "validate_prompts.py",
        "--adapter-dir", args.adapter_dir,
        "--base-model", args.base_model,
        "--prompts", args.prompts_file,
        "--out", args.output_file,
    ]
    
    if args.max_new_tokens:
        cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    
    run_command(cmd, "Validation")
    
    # Print summary
    print(f"\n📊 Validation results saved to: {args.output_file}")


def run_inference(args):
    """Run inference with the trained model."""
    cmd = [
        "python", "run_adapter.py",
        "--adapter-dir", args.adapter_dir,
        "--base-model", args.base_model,
        "--prompt", args.prompt,
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    
    run_command(cmd, "Inference")


def run_all(args):
    """Run complete pipeline: train -> validate -> run."""
    print("\n" + "="*60)
    print("RUNNING COMPLETE PIPELINE: TRAIN → VALIDATE → RUN")
    print("="*60)
    
    # Train
    train(args)
    
    # Validate
    validate(args)
    
    # Run (if prompt provided)
    if args.prompt:
        run_inference(args)
    else:
        print("\n⚠️  Skipping inference (no --prompt provided)")
    
    print("\n" + "="*60)
    print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Workflow script for LLM finetuning, validation, and inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--enable-wandb", action="store_true", help="Enable online W&B logging (overrides offline default)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # === TRAIN command ===
    train_parser = subparsers.add_parser("train", help="Finetune the model")
    train_parser.add_argument("--data-path", default="train_data.jsonl", help="Training data JSONL file")
    train_parser.add_argument("--output-dir", default="./fine_tuned_adapter", help="Output directory for adapters")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    train_parser.add_argument("--max-examples", type=int, help="Limit dataset size for quick runs")
    train_parser.add_argument("--model-name", default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit", help="Base model name")
    
    # === VALIDATE command ===
    validate_parser = subparsers.add_parser("validate", help="Validate trained model")
    validate_parser.add_argument("--adapter-dir", default="./fine_tuned_adapter/M1Checkpoint1", help="Path to adapter checkpoint")
    validate_parser.add_argument("--base-model", default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit", help="Base model name")
    validate_parser.add_argument("--prompts-file", default="validation_prompts.jsonl", help="Validation prompts JSONL file")
    validate_parser.add_argument("--output-file", default="validation_results.jsonl", help="Output results JSONL file")
    validate_parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate")
    
    # === RUN command ===
    run_parser = subparsers.add_parser("run", help="Run inference with trained model")
    run_parser.add_argument("--adapter-dir", default="./fine_tuned_adapter/M1Checkpoint1", help="Path to adapter checkpoint")
    run_parser.add_argument("--base-model", default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit", help="Base model name")
    run_parser.add_argument("--prompt", required=True, help="Prompt for inference")
    run_parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    # === ALL command (pipeline) ===
    all_parser = subparsers.add_parser("all", help="Run complete pipeline: train -> validate -> run")
    # Training args
    all_parser.add_argument("--data-path", default="train_data.jsonl", help="Training data JSONL file")
    all_parser.add_argument("--output-dir", default="./fine_tuned_adapter", help="Output directory for adapters")
    all_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    all_parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    all_parser.add_argument("--max-examples", type=int, help="Limit dataset size for quick runs")
    all_parser.add_argument("--model-name", default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit", help="Base model name")
    # Validation args
    all_parser.add_argument("--adapter-dir", help="Path to adapter checkpoint (defaults to output-dir/checkpoint-1)")
    all_parser.add_argument("--base-model", default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit", help="Base model name")
    all_parser.add_argument("--prompts-file", default="validation_prompts.jsonl", help="Validation prompts JSONL file")
    all_parser.add_argument("--output-file", default="validation_results.jsonl", help="Output results JSONL file")
    # Run args
    all_parser.add_argument("--prompt", help="Prompt for inference (optional for all command)")
    all_parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate")
    all_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    # Propagate W&B flag to run_command
    run_command.enable_wandb = getattr(args, "enable_wandb", False)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Auto-set adapter_dir for 'all' command if not specified
    if args.command == "all" and not args.adapter_dir:
        args.adapter_dir = f"{args.output_dir}/checkpoint-1"
    
    # Execute command
    if args.command == "train":
        train(args)
    elif args.command == "validate":
        validate(args)
    elif args.command == "run":
        run_inference(args)
    elif args.command == "all":
        run_all(args)


if __name__ == "__main__":
    main()
