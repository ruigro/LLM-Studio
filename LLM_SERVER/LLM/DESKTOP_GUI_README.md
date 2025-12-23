# Desktop GUI (PySide6) - No Streamlit

This repo originally shipped a Streamlit GUI (`gui.py`). This desktop GUI keeps the same backend scripts but replaces the UI with a native Qt app.

## Install
From the `LLM` folder:

```bash
pip install -r requirements.txt
```

## Run (Windows)
Double-click `run_desktop.bat`

Or:

```bash
python -m desktop_app.main
```

## What it reuses
- Training runs `finetune.py` (subprocess) and streams console output into the UI.
- Inference runs `run_adapter.py` (subprocess).
- Fine-tuned adapters are expected under `LLM/fine_tuned_adapter/`.
- Optional local Hugging Face downloads go to `LLM/hf_models/`.
