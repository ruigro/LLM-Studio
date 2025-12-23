from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import os

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


def detect_model_capabilities(model_id=None, model_name=None, model_path=None):
    """Detect model capabilities (vision, tools, text, reasoning) from model ID, name, or config"""
    capabilities = []
    
    # Check model path if provided
    if model_path and os.path.exists(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Check model_type
                    model_type = config.get("model_type", "").lower()
                    arch = config.get("architectures", [])
                    arch_str = " ".join(arch).lower() if arch else ""
                    
                    # Vision detection
                    if any(keyword in model_type + arch_str for keyword in ["vision", "vl", "multimodal", "clip", "llava"]):
                        capabilities.append("vision")
                    
                    # Tools detection
                    if any(keyword in model_type + arch_str for keyword in ["tool", "function", "agent"]):
                        capabilities.append("tools")
                    
                    # Reasoning detection
                    if any(keyword in model_type + arch_str for keyword in ["reasoning", "r1", "o1", "deepseek", "cot"]):
                        capabilities.append("reasoning")
            except Exception:
                pass
    
    # Check model ID or name for keywords
    check_str = ""
    if model_id:
        check_str += model_id.lower() + " "
    if model_name:
        check_str += model_name.lower() + " "
    
    # Vision keywords
    if "vision" not in capabilities and any(keyword in check_str for keyword in ["vision", "vl", "multimodal", "llava", "clip"]):
        capabilities.append("vision")
    
    # Tools keywords
    if "tools" not in capabilities and any(keyword in check_str for keyword in ["tool", "function-calling", "function_calling", "agent"]):
        capabilities.append("tools")
    
    # Reasoning keywords
    if "reasoning" not in capabilities and any(keyword in check_str for keyword in ["reasoning", "r1", "deepseek-r1", "o1", "chain-of-thought", "cot", "-reasoning"]):
        capabilities.append("reasoning")
    
    # Default to text if no special capabilities
    if not capabilities:
        capabilities.append("text")
    
    return capabilities


def get_capability_icons(capabilities):
    """Get emoji icons for model capabilities"""
    icons = []
    if "vision" in capabilities:
        icons.append("👁️")
    if "tools" in capabilities:
        icons.append("🔧")
    if "reasoning" in capabilities:
        icons.append("🧠")
    if "text" in capabilities and len(capabilities) == 1:
        icons.append("📝")
    return " ".join(icons) if icons else "📝"


def get_model_size(model_path):
    """Get model size in human-readable format"""
    if not model_path or not os.path.exists(model_path):
        return "Unknown"
    
    try:
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                fp = os.path.join(root, file)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        
        # Convert to GB
        size_gb = total_size / (1024 ** 3)
        if size_gb < 1:
            return f"{total_size / (1024 ** 2):.1f}MB"
        return f"{size_gb:.1f}GB"
    except Exception:
        return "Unknown"

