#!/usr/bin/env python3
"""Run validation prompts through the adapter + base model and report pass/fail.

Usage example:
    python validate_prompts.py \
        --adapter-dir ./fine_tuned_adapter/checkpoint-1 \
    --base-model unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit \
    --prompts validation_prompts.jsonl \
    --out results.jsonl
"""
import argparse
import json
from pathlib import Path

from run_adapter import load_model, generate_text


# Optional: import weave so W&B Weave tracing can be used during validation if installed
try:
    import weave  # type: ignore
    print("Weave imported in validate_prompts: LLM call tracing enabled (local).")
except Exception:
    pass

MENU_KEYWORDS = [
    "main menu",
    "please choose",
    "choose an option",
    "billing inquiry",
    "technical support",
    "account update",
    "menu",
]


def looks_like_menu(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(k in low for k in MENU_KEYWORDS)


def run_validation(adapter_dir, base_model, prompts_file, out_file, max_new_tokens=128, temperature=0.0, use_4bit=True, offload=True):
    tokenizer, model = load_model(base_model, adapter_dir, use_4bit=use_4bit, offload=offload)

    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))

    results = []
    total_prompts = len(prompts)
    passed_count = 0
    failed_count = 0
    
    print(f"\nRunning validation on {total_prompts} prompts...\n")
    
    for idx, p in enumerate(prompts, 1):
        pid = p.get("id")
        ptype = p.get("type")
        prompt = p.get("prompt")
        
        print(f"[{idx}/{total_prompts}] Testing prompt ID: {pid} (type: {ptype})...", end=" ")
        
        output = generate_text(tokenizer, model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        if ptype == "positive":
            passed = not looks_like_menu(output)
            note = "positive - should NOT be menu"
        else:
            passed = looks_like_menu(output)
            note = "negative - should be menu"

        if passed:
            print("✅ PASS")
            passed_count += 1
        else:
            print("❌ FAIL")
            failed_count += 1

        results.append({
            "id": pid,
            "type": ptype,
            "prompt": prompt,
            "output": output,
            "pass": passed,
            "note": note,
        })

    # write results JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompts:  {total_prompts}")
    print(f"✅ Passed:      {passed_count} ({passed_count/total_prompts*100:.1f}%)")
    print(f"❌ Failed:      {failed_count} ({failed_count/total_prompts*100:.1f}%)")
    print(f"Results saved:  {out_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--prompts", default="validation_prompts.jsonl")
    p.add_argument("--out", default="validation_results.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    p.add_argument("--no-offload", dest="offload", action="store_false")
    args = p.parse_args()

    run_validation(
        args.adapter_dir,
        args.base_model,
        args.prompts,
        args.out,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_4bit=args.use_4bit,
        offload=args.offload,
    )
