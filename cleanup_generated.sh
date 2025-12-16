#!/usr/bin/env bash
set -euo pipefail

# cleanup_generated.sh
# Dry-run by default. Pass --yes or -y to actually delete.

DRY_RUN=1
if [ "${1:-}" = "--yes" ] || [ "${1:-}" = "-y" ]; then
  DRY_RUN=0
fi

echo "Searching for generated files (dry-run=${DRY_RUN})..."

TMP_LIST=$(mktemp)

# Find directories that are usually safe to remove (caches, IDE folders, temp dirs)
find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ipynb_checkpoints" -o -name ".cache" -o -name ".vscode" -o -name ".idea" -o -name "tmp" -o -name "temp" -o -name "logs" -o -name "wandb" -o -name "mlruns" -o -name "tensorboard" -o -name "runs" -o -name "unsloth_compiled_cache" \) -print > "$TMP_LIST" || true

# Find some common generated files
find . -type f \( -name "*.pyc" -o -name ".DS_Store" -o -name "Thumbs.db" -o -name "*.log" \) -print >> "$TMP_LIST" || true

# Remove the single-line './.' if present
sed -i '/^\.$/d' "$TMP_LIST" || true

if [ ! -s "$TMP_LIST" ]; then
  echo "No generated files found."
  rm -f "$TMP_LIST"
  exit 0
fi

echo "Found the following files/directories:"
cat "$TMP_LIST"

if [ $DRY_RUN -eq 1 ]; then
  echo "Dry run complete. Re-run with --yes to delete these paths."
  rm -f "$TMP_LIST"
  exit 0
fi

echo "Proceeding to delete (skips likely important paths)..."

while IFS= read -r path; do
  # Skip obvious important directories or files
  case "$path" in
    *fine_tuned_adapter*|*data*|*dataset*|*checkpoints*|*Modelfile*|*train_data.jsonl*|*validation_prompts.jsonl*|*validation_results.jsonl*)
      echo "Skipping important path: $path"
      ;;
    *)
      echo "Removing: $path"
      rm -rf "$path"
      ;;
  esac
done < "$TMP_LIST"

rm -f "$TMP_LIST"
echo "Cleanup finished."
