from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import snapshot_download, list_models
except Exception:  # pragma: no cover
    snapshot_download = None
    list_models = None


DEFAULT_BASE_MODELS: List[str] = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    "unsloth/Phi-4-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/gemma-2-27b-it-bnb-4bit",
]


@dataclass
class HFModelHit:
    model_id: str
    downloads: int | None = None
    likes: int | None = None
    last_modified: str | None = None


def get_app_root() -> Path:
    # .../LLM/core/models.py -> .../LLM
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def search_hf_models(query: str, limit: int = 20) -> List[HFModelHit]:
    """Search Hugging Face models by free-text query."""
    if list_models is None:
        raise RuntimeError("huggingface_hub is not available. Install requirements.txt")
    hits: List[HFModelHit] = []
    for m in list_models(search=query, limit=limit):
        hits.append(
            HFModelHit(
                model_id=getattr(m, "modelId", None) or getattr(m, "id", ""),
                downloads=getattr(m, "downloads", None),
                likes=getattr(m, "likes", None),
                last_modified=str(getattr(m, "lastModified", None) or ""),
            )
        )
    return hits


def download_hf_model(model_id: str, target_dir: Path) -> Path:
    """Download a HF model snapshot into target_dir/<model_id_slug>."""
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is not available. Install requirements.txt")
    target_dir = ensure_dir(target_dir)
    slug = model_id.replace("/", "__")
    dest = ensure_dir(target_dir / slug)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return dest


def list_local_adapters(adapter_root: Optional[Path] = None) -> List[str]:
    if adapter_root is None:
        adapter_root = get_app_root() / "fine_tuned_adapter"
    if not adapter_root.exists():
        return []
    return sorted([p.name for p in adapter_root.iterdir() if p.is_dir()])


def list_local_downloads(download_root: Optional[Path] = None) -> List[str]:
    if download_root is None:
        download_root = get_app_root() / "hf_models"
    if not download_root.exists():
        return []
    return sorted([p.name for p in download_root.iterdir() if p.is_dir()])
