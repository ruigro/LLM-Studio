from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import subprocess
import sys
import os


def get_app_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class InferenceConfig:
    prompt: str
    base_model: Optional[str] = None
    adapter_dir: Optional[Path] = None
    max_new_tokens: int = 256
    temperature: float = 0.7


def build_run_adapter_cmd(cfg: InferenceConfig) -> List[str]:
    cmd = [sys.executable, "-u", "run_adapter.py", "--prompt", cfg.prompt]
    if cfg.base_model:
        cmd += ["--base-model", cfg.base_model]
    if cfg.adapter_dir:
        cmd += ["--adapter-dir", str(cfg.adapter_dir)]
    cmd += ["--max-new-tokens", str(cfg.max_new_tokens), "--temperature", str(cfg.temperature)]
    return cmd


def run_inference(cfg: InferenceConfig, env: Optional[dict] = None) -> str:
    workdir = str(get_app_root())
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    cmd = build_run_adapter_cmd(cfg)
    p = subprocess.run(cmd, cwd=workdir, env=proc_env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return out
