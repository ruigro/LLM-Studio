"""
Environment Registry - Bridges EnvironmentManager and LLM Server System
Provides environment specifications with validated Python paths for each model.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import json
from typing import Optional
import subprocess


@dataclass
class EnvSpec:
    """Environment specification with validated python executable"""
    key: str  # Environment identifier
    python_executable: Path  # Path to python.exe in the environment (validated)
    metadata: dict  # Environment metadata from EnvironmentManager


class EnvRegistry:
    """Registry for managing per-model Python environments"""
    
    def __init__(self):
        from core.environment_manager import EnvironmentManager
        self.env_manager = EnvironmentManager()

    def _get_active_profile_data(self) -> Optional[dict]:
        """
        Resolve the active (selected/auto) hardware profile and return its JSON data.
        This is the single source of truth for torch build (+cuXXX) and torch_index.
        """
        try:
            llm_dir = Path(__file__).resolve().parent.parent.parent  # LLM/
            from setup_state import SetupStateManager
            from system_detector import SystemDetector
            from core.profile_selector import ProfileSelector

            state = SetupStateManager()
            override_profile_id = state.get_selected_profile()
            selected_gpu_index = state.get_selected_gpu_index()

            detector = SystemDetector()
            hw_profile = detector.get_hardware_profile(selected_gpu_index=selected_gpu_index)

            matrix_path = llm_dir / "metadata" / "compatibility_matrix.json"
            selector = ProfileSelector(matrix_path)
            profile_id, _pkg_versions, _warnings, _binary_pkgs = selector.select_profile(
                hw_profile,
                override_profile_id=override_profile_id,
            )

            profile_path = llm_dir / "profiles" / f"{profile_id}.json"
            if profile_path.exists():
                return json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None

    def _run_python(self, python_exe: Path, code: str, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(python_exe), "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _env_needs_cuda_torch(self, python_exe: Path, profile_data: Optional[dict]) -> bool:
        """
        Return True if the active profile expects CUDA torch, but this env does not have CUDA torch.
        """
        if not profile_data:
            return False
        try:
            torch_spec = str(profile_data.get("packages", {}).get("torch", ""))
            torch_index = str(profile_data.get("torch_index", ""))
            expects_cuda = ("+cu" in torch_spec) or ("/whl/cu" in torch_index)
            if not expects_cuda:
                return False

            # Check current env torch state
            r = self._run_python(
                python_exe,
                "import torch; print(torch.__version__); print('CUDA', torch.cuda.is_available())",
                timeout=20,
            )
            if r.returncode != 0:
                return True
            out = (r.stdout or "").lower()
            cuda_ok = "cuda true" in out
            has_cu = "+cu" in (r.stdout or "")
            return not (cuda_ok and has_cu)
        except Exception:
            return True

    def _ensure_profile_torch(self, python_exe: Path, profile_data: dict, log_callback=None) -> None:
        """
        Ensure the per-model environment has the correct torch stack for the active profile.
        Installs from profile_data['torch_index'] with the exact versions in profile_data['packages'].
        """
        def log(msg: str):
            if log_callback:
                log_callback(msg)
            import logging
            logging.info(msg)

        pkgs = profile_data.get("packages", {}) or {}
        torch_spec = pkgs.get("torch")
        torchvision_spec = pkgs.get("torchvision")
        torchaudio_spec = pkgs.get("torchaudio")
        torch_index = profile_data.get("torch_index")

        if not (torch_spec and torchvision_spec and torchaudio_spec and torch_index):
            log("Profile missing torch stack details; skipping CUDA torch enforcement.")
            return

        # Normalize to pip specs
        def _spec(name: str, ver: str) -> str:
            ver = str(ver).strip()
            if ver.startswith(("==", ">=", "<=", ">", "<")):
                return f"{name}{ver}"
            return f"{name}=={ver}"

        torch_pkg = _spec("torch", torch_spec)
        tv_pkg = _spec("torchvision", torchvision_spec)
        ta_pkg = _spec("torchaudio", torchaudio_spec)

        log(f"Ensuring CUDA torch stack in per-model env: {torch_pkg} ({torch_index})")

        # Uninstall any existing stack first (avoid mixed CPU/CUDA wheels)
        for pkg in ["torch", "torchvision", "torchaudio", "xformers", "triton", "triton-windows"]:
            try:
                subprocess.run(
                    [str(python_exe), "-m", "pip", "uninstall", "-y", pkg],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except Exception:
                pass

        # Install in order: torch -> torchvision -> torchaudio
        for pkg_spec in [torch_pkg, tv_pkg, ta_pkg]:
            log(f"Installing {pkg_spec} ...")
            r = subprocess.run(
                [
                    str(python_exe),
                    "-m",
                    "pip",
                    "install",
                    "--index-url",
                    str(torch_index),
                    "--force-reinstall",
                    "--no-deps",
                    pkg_spec,
                ],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip()
                raise RuntimeError(f"Failed to install {pkg_spec} in per-model env: {err[:800]}")

        # Verify torch is CUDA and matches expected build
        verify = self._run_python(
            python_exe,
            "import torch; print(torch.__version__); assert torch.cuda.is_available(); print('OK')",
            timeout=30,
        )
        if verify.returncode != 0:
            raise RuntimeError(
                f"Per-model env torch verification failed.\nSTDOUT:\n{verify.stdout}\nSTDERR:\n{verify.stderr}"
            )
    
    def get_env_for_model(self, model_path: str, log_callback=None) -> EnvSpec:
        """
        Get environment spec for a model, creating if needed.
        
        Args:
            model_path: Path to the model (base model path)
            log_callback: Optional function to call with log messages
            
        Returns:
            EnvSpec with validated python executable path
            
        Raises:
            RuntimeError: If environment is corrupted or python not found
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            import logging
            logging.info(msg)

        # Get environment path - this returns the directory path
        env_dir = self.env_manager.get_environment_path(model_path=model_path)
        
        # Check if environment exists
        env_created = False
        if not self.env_manager.environment_exists(model_path=model_path):
            # Create the environment
            log(f"Environment for {model_path} doesn't exist, creating...")
            try:
                # Create environment (this installs packages too)
                success, error = self.env_manager.create_environment(
                    model_path=model_path
                )
                if not success:
                    raise RuntimeError(f"Failed to create environment: {error}")
                env_created = True
            except Exception as e:
                raise RuntimeError(f"Failed to create environment: {e}")
        
        # Get Python executable path
        python_exe = self.env_manager.get_python_executable(model_path=model_path)

        # Determine active profile once (source of truth)
        profile_data = self._get_active_profile_data()

        # If the environment exists but has CPU torch while we expect CUDA, treat as missing deps.
        if python_exe and profile_data and self._env_needs_cuda_torch(python_exe, profile_data):
            env_created = True  # force reinstall path below
        
        # Install dependencies if environment was just created or if they're missing
        if env_created or not self._has_server_dependencies(python_exe):
            import subprocess
            from pathlib import Path
            
            llm_dir = Path(__file__).parent.parent.parent
            
            log(f"Installing dependencies for server environment...")
            try:
                # First install server framework (fast, needed first)
                log(f"Installing server framework (uvicorn, fastapi)...")
                result = subprocess.run(
                    [str(python_exe), "-m", "pip", "install", 
                     "uvicorn[standard]", "fastapi", "pydantic", "pyyaml", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode != 0:
                    log(f"Warning: Failed to install server framework: {result.stderr}")

                # Ensure torch stack matches the active profile (CUDA vs CPU)
                if profile_data:
                    try:
                        self._ensure_profile_torch(python_exe, profile_data, log_callback=log_callback)
                    except Exception as e:
                        raise RuntimeError(f"Failed to ensure profile torch stack: {e}")

                # Install minimal inference stack (avoid full UI requirements in per-model env).
                # This keeps per-model envs smaller and prevents resolver deadlocks/conflicts.
                pkgs = (profile_data or {}).get("packages", {}) if isinstance(profile_data, dict) else {}
                def _pkg(name: str, default: Optional[str] = None) -> Optional[str]:
                    v = pkgs.get(name)
                    if v is None:
                        return default
                    v = str(v).strip()
                    if v.startswith(("==", ">=", "<=", ">", "<")):
                        return f"{name}{v}"
                    return f"{name}=={v}"

                # Keep transformers pinned to a known-good build (avoids picking very new versions that
                # can conflict with tighter hub pins in older profiles).
                minimal_specs = [
                    _pkg("numpy", "numpy==1.26.4"),
                    _pkg("huggingface-hub", None),
                    "transformers==4.51.3",
                    "tokenizers==0.21.4",
                    _pkg("safetensors", "safetensors>=0.7.0,<0.8.0"),
                    _pkg("accelerate", "accelerate>=1.2.0,<1.3.0"),
                    _pkg("peft", "peft>=0.13.0,<0.16.0"),
                    _pkg("bitsandbytes", "bitsandbytes>=0.45.0,<0.50.0"),
                    _pkg("sentencepiece", "sentencepiece==0.2.0"),
                    _pkg("pyyaml", "pyyaml>=6.0.0,<7.0.0"),
                    _pkg("requests", "requests>=2.31.0,<3.0.0"),
                ]
                minimal_specs = [s for s in minimal_specs if s]
                log("Installing minimal inference stack in per-model env...")
                for spec in minimal_specs:
                    log(f"Installing {spec} ...")
                    r = subprocess.run(
                        [str(python_exe), "-m", "pip", "install", "--upgrade", spec],
                        capture_output=True,
                        text=True,
                        timeout=1800,
                    )
                    if r.returncode != 0:
                        err = (r.stderr or r.stdout or "").strip()
                        log(f"Warning: Failed to install {spec}: {err[:500]}")
                
                log("Dependencies installed (per-model inference env).")
            except Exception as e:
                log(f"Warning: Failed to install dependencies: {e}")
                # Non-fatal - might already be installed
            except Exception as e:
                logging.warning(f"Failed to install dependencies: {e}")
                # Non-fatal - might already be installed
        
        # VALIDATE: Ensure python executable exists
        if python_exe is None or not python_exe.exists():
            raise RuntimeError(
                f"Environment python not found: {python_exe}\n"
                f"Environment may be corrupted. Try recreating it.\n"
                f"Environment directory: {env_dir}"
            )
        
        # Validate it's actually executable (on Windows, check it's a file)
        if not python_exe.is_file():
            raise RuntimeError(
                f"Python path exists but is not a file: {python_exe}\n"
                f"Environment may be corrupted."
            )
        
        # Load metadata
        metadata_file = env_dir / "environment_metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
            except Exception as e:
                # Non-fatal: use empty metadata if we can't load it
                metadata = {"error": f"Failed to load metadata: {e}"}
        else:
            metadata = {"warning": "No metadata file found"}
        
        return EnvSpec(
            key=env_dir.name,
            python_executable=python_exe,
            metadata=metadata
        )
    
    def _has_server_dependencies(self, python_exe: Path) -> bool:
        """
        Check if server and ML dependencies are installed.
        
        Args:
            python_exe: Path to Python executable
            
        Returns:
            True if uvicorn, fastapi, transformers, peft, and torch are available.
            If the active profile expects CUDA torch, this also requires torch.cuda.is_available().
        """
        try:
            profile_data = self._get_active_profile_data()
            require_cuda = False
            if profile_data:
                torch_spec = str(profile_data.get("packages", {}).get("torch", ""))
                torch_index = str(profile_data.get("torch_index", ""))
                require_cuda = ("+cu" in torch_spec) or ("/whl/cu" in torch_index)

            code = "import uvicorn, fastapi, transformers, peft, torch\n"
            if require_cuda:
                code += "assert torch.cuda.is_available()\n"
                code += "assert '+cu' in torch.__version__\n"
            result = subprocess.run([str(python_exe), "-c", code], capture_output=True, timeout=30)
            return result.returncode == 0
        except Exception:
            return False
    
    def validate_env_spec(self, env_spec: EnvSpec) -> bool:
        """
        Validate an environment specification.
        
        Args:
            env_spec: The environment spec to validate
            
        Returns:
            True if valid, False otherwise
        """
        return (
            env_spec.python_executable.exists() and
            env_spec.python_executable.is_file()
        )
