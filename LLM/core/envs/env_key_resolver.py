"""
PHASE 2: Environment Key Resolver
Maps model requirements to shared environment keys (e.g., "torch-cu121-transformers-bnb").
Eliminates per-model environment explosion.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


class EnvKeyResolver:
    """Resolves environment keys from model requirements and hardware profiles"""
    
    def __init__(self):
        """Initialize resolver with profile data"""
        self.llm_dir = Path(__file__).parent.parent.parent
        self.profile_cache: Dict[str, dict] = {}
    
    def get_active_profile_data(self) -> Optional[dict]:
        """Get active hardware profile data (CUDA version, torch build, etc.)"""
        try:
            from setup_state import SetupStateManager
            from system_detector import SystemDetector
            from core.profile_selector import ProfileSelector
            
            state = SetupStateManager()
            override_profile_id = state.get_selected_profile()
            selected_gpu_index = state.get_selected_gpu_index()
            
            detector = SystemDetector()
            hw_profile = detector.get_hardware_profile(selected_gpu_index=selected_gpu_index)
            
            matrix_path = self.llm_dir / "metadata" / "compatibility_matrix.json"
            selector = ProfileSelector(matrix_path)
            profile_id, _pkg_versions, _warnings, _binary_pkgs = selector.select_profile(
                hw_profile,
                override_profile_id=override_profile_id,
            )
            
            # Cache profile data
            if profile_id not in self.profile_cache:
                profile_path = self.llm_dir / "profiles" / f"{profile_id}.json"
                if profile_path.exists():
                    self.profile_cache[profile_id] = json.loads(profile_path.read_text(encoding="utf-8"))
            
            return self.profile_cache.get(profile_id)
        except Exception as e:
            logger.warning(f"Failed to get active profile: {e}")
            return None
    
    def resolve_env_key(
        self,
        backend: str = "transformers",
        use_quantization: bool = True,
        model_path: Optional[str] = None,
        profile_data: Optional[dict] = None
    ) -> str:
        """
        Resolve environment key from model requirements.
        
        Args:
            backend: Backend type (transformers, vllm, llamacpp)
            use_quantization: Whether to use quantization (bitsandbytes)
            model_path: Optional model path for special cases
            profile_data: Optional profile data (if None, auto-detected)
        
        Returns:
            Environment key string (e.g., "torch-cu121-transformers-bnb")
        
        Examples:
            - torch-cu121-transformers-bnb (CUDA 12.1, transformers + bitsandbytes)
            - torch-cu124-transformers (CUDA 12.4, transformers only)
            - torch-cpu-transformers (CPU-only)
            - vllm-cu121 (vLLM with CUDA 12.1)
            - llamacpp-cpu (llama.cpp CPU)
        """
        if profile_data is None:
            profile_data = self.get_active_profile_data()
        
        # Determine CUDA version from profile
        cuda_version = "cpu"
        if profile_data:
            torch_spec = str(profile_data.get("packages", {}).get("torch", ""))
            torch_index = str(profile_data.get("torch_index", ""))
            
            # Extract CUDA version from torch spec or index URL
            if "+cu" in torch_spec:
                # e.g., "2.5.1+cu121" -> "cu121"
                cuda_part = torch_spec.split("+cu")[1]
                cuda_version = f"cu{cuda_part[:3]}"  # "cu121"
            elif "/whl/cu" in torch_index:
                # e.g., ".../whl/cu121" -> "cu121"
                cuda_part = torch_index.split("/whl/cu")[1].split("/")[0]
                cuda_version = f"cu{cuda_part[:3]}"
        
        # Build env_key based on backend
        parts = ["torch", cuda_version]
        
        if backend == "transformers":
            parts.append("transformers")
            if use_quantization:
                parts.append("bnb")  # bitsandbytes
        elif backend == "vllm":
            parts.append("vllm")
        elif backend == "llamacpp":
            parts = ["llamacpp", cuda_version]
        else:
            # Generic backend
            parts.append(backend)
        
        env_key = "-".join(parts)
        logger.debug(f"Resolved env_key: {env_key} (backend={backend}, quant={use_quantization}, cuda={cuda_version})")
        return env_key
    
    def parse_env_key(self, env_key: str) -> Dict[str, Any]:
        """
        Parse environment key back into components.
        
        Args:
            env_key: Environment key string
        
        Returns:
            Dict with backend, cuda_version, quantization flags
        
        Example:
            parse_env_key("torch-cu121-transformers-bnb") ->
            {"backend": "transformers", "cuda": "cu121", "quantization": True}
        """
        parts = env_key.split("-")
        
        result = {
            "backend": "unknown",
            "cuda": "cpu",
            "quantization": False,
            "framework": "torch"
        }
        
        # Parse parts
        if "llamacpp" in parts:
            result["framework"] = "llamacpp"
            result["backend"] = "llamacpp"
            if any(p.startswith("cu") for p in parts):
                result["cuda"] = next(p for p in parts if p.startswith("cu"))
        elif "vllm" in parts:
            result["backend"] = "vllm"
            if any(p.startswith("cu") for p in parts):
                result["cuda"] = next(p for p in parts if p.startswith("cu"))
        elif "transformers" in parts:
            result["backend"] = "transformers"
            if any(p.startswith("cu") for p in parts):
                result["cuda"] = next(p for p in parts if p.startswith("cu"))
            result["quantization"] = "bnb" in parts or "quantized" in parts
        
        return result
    
    def get_env_key_display_name(self, env_key: str) -> str:
        """
        Get human-readable name for environment key.
        
        Args:
            env_key: Environment key
        
        Returns:
            Display name
        
        Example:
            "torch-cu121-transformers-bnb" -> "Transformers + Quantization (CUDA 12.1)"
        """
        info = self.parse_env_key(env_key)
        
        # Build display name
        parts = []
        
        if info["backend"] == "transformers":
            parts.append("Transformers")
            if info["quantization"]:
                parts.append("+ Quantization")
        elif info["backend"] == "vllm":
            parts.append("vLLM")
        elif info["backend"] == "llamacpp":
            parts.append("llama.cpp")
        else:
            parts.append(info["backend"].title())
        
        # Add CUDA info
        cuda = info["cuda"]
        if cuda != "cpu":
            cuda_ver = cuda.replace("cu", "")
            if len(cuda_ver) == 3:
                # "121" -> "12.1"
                cuda_display = f"{cuda_ver[0:2]}.{cuda_ver[2]}"
            else:
                cuda_display = cuda_ver
            parts.append(f"(CUDA {cuda_display})")
        else:
            parts.append("(CPU)")
        
        return " ".join(parts)
