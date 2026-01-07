"""
Model Compatibility Detection and Runtime Capability Checking

This module provides generic detection of model types, library capabilities,
and version-aware parameter passing to avoid hardcoding version requirements.
"""

from typing import Dict, Optional, Tuple
import re


def detect_model_type(model_name: str) -> Dict[str, any]:
    """
    Detect model type and requirements from model name.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "unsloth/gemma-2-2b-it-bnb-4bit")
    
    Returns:
        Dict with:
        - is_unsloth: bool - Model is from unsloth namespace
        - is_quantized: bool - Model requires quantization (bnb-4bit, etc.)
        - model_family: str - Model family (qwen, llama, gemma, mistral, phi, etc.)
        - requires_unsloth: bool - Model explicitly requires unsloth
        - requires_quantization: bool - Model explicitly requires quantization
        - base_model_name: str - Base model name without unsloth/quantization suffixes
    """
    model_lower = model_name.lower()
    
    # Detect unsloth models
    is_unsloth = "/unsloth/" in model_name or "-unsloth-" in model_lower
    
    # Detect quantization
    is_quantized = (
        "bnb" in model_lower or 
        "4bit" in model_lower or 
        "8bit" in model_lower or
        "-4bit" in model_lower or
        "-8bit" in model_lower
    )
    
    # Extract model family
    model_family = _extract_model_family(model_name)
    
    # Determine base model name (remove unsloth/quantization suffixes)
    base_model_name = _extract_base_model_name(model_name)
    
    return {
        "is_unsloth": is_unsloth,
        "is_quantized": is_quantized,
        "model_family": model_family,
        "requires_unsloth": is_unsloth,  # Unsloth models require unsloth library
        "requires_quantization": is_quantized,  # Quantized models require bitsandbytes
        "base_model_name": base_model_name,
        "original_name": model_name,
    }


def _extract_model_family(model_name: str) -> str:
    """Extract model family from model name."""
    model_lower = model_name.lower()
    
    if "qwen" in model_lower:
        return "qwen"
    elif "llama" in model_lower:
        return "llama"
    elif "gemma" in model_lower:
        return "gemma"
    elif "mistral" in model_lower:
        return "mistral"
    elif "phi" in model_lower:
        return "phi"
    elif "hermes" in model_lower:
        return "hermes"
    else:
        return "unknown"


def _extract_base_model_name(model_name: str) -> str:
    """Extract base model name by removing unsloth/quantization suffixes."""
    base = model_name
    
    # Remove unsloth namespace
    if "/unsloth/" in base:
        base = base.replace("/unsloth/", "/")
    base = base.replace("-unsloth-", "-")
    
    # Remove quantization suffixes
    base = re.sub(r"-bnb-4bit$", "", base)
    base = re.sub(r"-bnb-8bit$", "", base)
    base = re.sub(r"-4bit$", "", base)
    base = re.sub(r"-8bit$", "", base)
    
    return base


def check_peft_capabilities() -> Dict[str, any]:
    """
    Check what peft features are available based on installed version.
    
    Returns:
        Dict with:
        - version: str - Installed peft version
        - available: bool - Whether peft is installed
        - supports_ensure_weight_tying: bool - Supports ensure_weight_tying parameter
        - supports_other_features: dict - Other feature flags
    """
    try:
        import peft
        from packaging import version as pkg_version
        
        peft_version = peft.__version__
        ver = pkg_version.parse(peft_version)
        
        # Check for ensure_weight_tying support (added in 0.14.0)
        supports_ensure_weight_tying = ver >= pkg_version.parse("0.14.0")
        
        # Check LoraConfig for available parameters
        try:
            from peft import LoraConfig
            import inspect
            lora_params = set(inspect.signature(LoraConfig.__init__).parameters.keys())
            supports_ensure_weight_tying = supports_ensure_weight_tying or "ensure_weight_tying" in lora_params
        except Exception:
            pass
        
        return {
            "version": peft_version,
            "available": True,
            "supports_ensure_weight_tying": supports_ensure_weight_tying,
            "supports_other_features": {
                # Add other feature checks here as needed
            }
        }
    except ImportError:
        return {
            "version": None,
            "available": False,
            "supports_ensure_weight_tying": False,
            "supports_other_features": {}
        }
    except Exception as e:
        # If version parsing fails, assume older version
        return {
            "version": "unknown",
            "available": True,
            "supports_ensure_weight_tying": False,
            "supports_other_features": {}
        }


def check_unsloth_capabilities() -> Dict[str, any]:
    """
    Check if unsloth is available and functional.
    
    Returns:
        Dict with:
        - available: bool - Whether unsloth is installed
        - functional: bool - Whether unsloth can be imported and used
        - version: str - Unsloth version if available
    """
    try:
        from unsloth import FastLanguageModel
        # Try to check version if available
        try:
            import unsloth
            version = getattr(unsloth, "__version__", "unknown")
        except:
            version = "unknown"
        
        return {
            "available": True,
            "functional": True,
            "version": version
        }
    except ImportError:
        return {
            "available": False,
            "functional": False,
            "version": None
        }
    except Exception as e:
        return {
            "available": True,
            "functional": False,
            "version": "unknown",
            "error": str(e)
        }


def check_bitsandbytes_capabilities() -> Dict[str, any]:
    """
    Check if bitsandbytes is available and functional.
    
    Returns:
        Dict with:
        - available: bool - Whether bitsandbytes is installed
        - functional: bool - Whether bitsandbytes can be used (requires triton.ops on Windows)
        - version: str - bitsandbytes version if available
    """
    try:
        import bitsandbytes
        from bitsandbytes.nn import Linear8bitLt
        
        version = getattr(bitsandbytes, "__version__", "unknown")
        
        # On Windows, check if triton.ops is available (required for bitsandbytes)
        functional = True
        import sys
        if sys.platform == "win32":
            try:
                import triton.ops
            except ImportError:
                functional = False
        
        return {
            "available": True,
            "functional": functional,
            "version": version
        }
    except ImportError:
        return {
            "available": False,
            "functional": False,
            "version": None
        }
    except Exception as e:
        return {
            "available": True,
            "functional": False,
            "version": "unknown",
            "error": str(e)
        }


def get_compatible_peft_params(
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    capabilities: Optional[Dict] = None
) -> Dict[str, any]:
    """
    Return PEFT parameters compatible with installed peft version.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: Target modules for LoRA
        bias: Bias type
        task_type: Task type
        capabilities: Optional pre-computed peft capabilities (if None, will check)
    
    Returns:
        Dict of parameters safe to pass to LoraConfig
    """
    if capabilities is None:
        capabilities = check_peft_capabilities()
    
    params = {
        "r": r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "lora_dropout": lora_dropout,
        "bias": bias,
        "task_type": task_type,
    }
    
    # Only add ensure_weight_tying if supported
    if capabilities.get("supports_ensure_weight_tying", False):
        params["ensure_weight_tying"] = True
    
    return params


def get_compatible_unsloth_params(
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list,
    use_gradient_checkpointing: str = "unsloth",
    capabilities: Optional[Dict] = None
) -> Dict[str, any]:
    """
    Return unsloth parameters compatible with installed peft version.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: Target modules for LoRA
        use_gradient_checkpointing: Gradient checkpointing mode
        capabilities: Optional pre-computed peft capabilities (if None, will check)
    
    Returns:
        Dict of parameters safe to pass to FastLanguageModel.get_peft_model
    """
    if capabilities is None:
        capabilities = check_peft_capabilities()
    
    params = {
        "r": r,
        "target_modules": target_modules,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "use_gradient_checkpointing": use_gradient_checkpointing,
    }
    
    # Only add ensure_weight_tying if supported
    if capabilities.get("supports_ensure_weight_tying", False):
        params["ensure_weight_tying"] = True
    
    return params


def get_optimal_loading_strategy(model_name: str) -> Tuple[str, Dict[str, any]]:
    """
    Determine optimal loading strategy for a model based on available capabilities.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        Tuple of (strategy_name, strategy_info)
        strategy_name: "unsloth", "peft", or "base"
        strategy_info: Dict with capabilities and fallback info
    """
    model_info = detect_model_type(model_name)
    peft_caps = check_peft_capabilities()
    unsloth_caps = check_unsloth_capabilities()
    bnb_caps = check_bitsandbytes_capabilities()
    
    # Strategy 1: Try unsloth if model requires it and it's available
    if model_info["requires_unsloth"] and unsloth_caps["functional"] and peft_caps["available"]:
        return ("unsloth", {
            "can_use": True,
            "reason": "Model requires unsloth and it's available",
            "fallback": "peft",
            "capabilities": {
                "peft": peft_caps,
                "unsloth": unsloth_caps,
                "bitsandbytes": bnb_caps
            }
        })
    
    # Strategy 2: Use standard PEFT if peft is available
    if peft_caps["available"]:
        return ("peft", {
            "can_use": True,
            "reason": "Standard PEFT available",
            "fallback": "base",
            "capabilities": {
                "peft": peft_caps,
                "bitsandbytes": bnb_caps
            }
        })
    
    # Strategy 3: Fall back to base transformers
    return ("base", {
        "can_use": True,
        "reason": "Falling back to base transformers (no PEFT)",
        "fallback": None,
        "capabilities": {
            "bitsandbytes": bnb_caps
        }
    })
