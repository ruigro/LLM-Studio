#!/usr/bin/env python3
"""
Lightweight dependency health check for launcher
Verifies critical packages are installed with correct versions
"""

import sys
import subprocess
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

try:
    from packaging.specifiers import SpecifierSet
    from packaging import version as pkg_version
    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False
    # Fallback: simple version comparison for == only
    pkg_version = None
    SpecifierSet = None


def check_package(pkg_name, version_spec=None):
    """Check if package is installed and matches version spec"""
    try:
        installed_ver = version(pkg_name)
        if not version_spec:
            return True, installed_ver
        
        if not PACKAGING_AVAILABLE:
            # Fallback: only support == for exact matches
            if "==" in version_spec:
                required = version_spec.split("==")[1].strip()
                if installed_ver == required:
                    return True, installed_ver
                else:
                    return False, f"{pkg_name} {installed_ver} != {required}"
            else:
                # Can't do complex comparison without packaging
                return True, installed_ver  # Assume OK if we can't verify
        
        # Use packaging library for robust version comparison
        spec = SpecifierSet(version_spec)
        if spec.contains(pkg_version.parse(installed_ver)):
            return True, installed_ver
        else:
            return False, f"{pkg_name} {installed_ver} does not match {version_spec}"
    except PackageNotFoundError:
        return False, f"{pkg_name} not installed"
    except Exception as e:
        return False, f"Error checking {pkg_name}: {str(e)}"


def check_pytorch_cuda():
    """Check if PyTorch has CUDA support (if GPU detected)"""
    try:
        # First check if torch package metadata exists
        try:
            torch_ver = version("torch")
        except PackageNotFoundError:
            return False, "PyTorch not installed (no package metadata)"
        
        # Try to import torch
        try:
            import torch
        except ImportError:
            return False, "PyTorch package exists but cannot import"
        
        # Check CUDA availability
        try:
            if torch.cuda.is_available():
                return True, f"CUDA available (torch {torch.__version__})"
            
            # Check if GPU exists but PyTorch is CPU-only
            try:
                result = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                if result.returncode == 0:
                    return False, "GPU detected but PyTorch is CPU-only"
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass  # nvidia-smi not available or failed, assume no GPU
            
            return True, f"CPU-only (torch {torch.__version__})"
        except Exception as e:
            return False, f"PyTorch import error: {str(e)}"
    except Exception as e:
        return False, f"PyTorch check failed: {str(e)}"


def _normalize_spec(spec: str) -> str:
    """
    Normalize a profile version entry into a pip/packaging specifier.
    Profiles may use:
      - exact: "1.26.4" or "2.5.1+cu121"
      - ranges: ">=4.51.0,<4.60.0"
    """
    spec = str(spec or "").strip()
    if not spec:
        return ""
    # Already a spec/range
    if any(op in spec for op in [">=", "<=", ">", "<", "!=", ","]) or spec.startswith("=="):
        return spec
    # Bare version -> exact
    return f"=={spec}"


def check_pyside6():
    """Check if PySide6 and shiboken6 are properly installed (matches launcher health check)"""
    try:
        # Check PySide6 package metadata
        try:
            pyside6_ver = version("PySide6")
        except PackageNotFoundError:
            return False, "PySide6 not installed (no package metadata)"
        
        # Check shiboken6 package metadata
        try:
            shiboken6_ver = version("shiboken6")
        except PackageNotFoundError:
            return False, "shiboken6 not installed (no package metadata)"
        
        # Try to import PySide6.QtCore (matches launcher health check exactly)
        try:
            import PySide6.QtCore
            return True, f"PySide6 {pyside6_ver}, shiboken6 {shiboken6_ver}"
        except ImportError as e:
            error_msg = str(e)
            if "shiboken" in error_msg.lower() or "does not exist" in error_msg:
                return False, f"PySide6/shiboken6 import failed: {error_msg}"
            return False, f"PySide6 import error: {error_msg}"
    except Exception as e:
        return False, f"PySide6 check failed: {str(e)}"


def verify_all():
    """Verify all critical dependencies"""
    # Single source of truth: derive checks from the selected hardware profile.
    # This avoids "repair loops" caused by hardcoded version ranges drifting from profiles.
    llm_dir = Path(__file__).parent
    matrix_path = llm_dir / "metadata" / "compatibility_matrix.json"

    try:
        from system_detector import SystemDetector
        from core.profile_selector import ProfileSelector
        from setup_state import SetupStateManager

        # Use persisted profile selection (same as InstallerV2)
        setup_state = SetupStateManager()
        selected_gpu_index = setup_state.get_selected_gpu_index()
        override_profile = setup_state.get_selected_profile()

        detector = SystemDetector()
        hardware_profile = detector.get_hardware_profile(selected_gpu_index=selected_gpu_index)
        selector = ProfileSelector(matrix_path)
        profile_name, package_versions, warnings, _binary = selector.select_profile(
            hardware_profile, override_profile_id=override_profile
        )

        # Report selected profile (useful in dependency_check.log)
        print(f"[PROFILE] Selected '{profile_name}' for dependency check")
        if override_profile:
            print(f"[PROFILE] Using user-selected profile override")
        if selected_gpu_index is not None:
            print(f"[PROFILE] Using user-selected GPU index {selected_gpu_index}")
        for w in warnings or []:
            print(f"[PROFILE] Warning: {w}")

        # Only check a small stable set; exact pins come from profiles.
        # (Don't hardcode newer ranges here — that's what caused the loop.)
        check_pkgs = ["numpy", "transformers", "tokenizers", "huggingface-hub", "datasets"]
        checks = []
        for pkg in check_pkgs:
            if pkg in package_versions:
                checks.append((pkg, _normalize_spec(package_versions[pkg])))
            else:
                # If profile doesn't specify it, skip quietly.
                pass
    except Exception as e:
        # Fallback: keep checks broad and avoid forcing repair loops
        print(f"[WARN] Could not load profile-based checks: {e}", file=sys.stderr)
        checks = [
            ("numpy", "<2.0.0"),
            ("transformers", ">=4.51.0,<4.60.0"),
            ("tokenizers", ">=0.21.0,<0.24.0"),
            # Keep hub compatible with current profiles (do NOT require >=0.30.0).
            ("huggingface-hub", ">=0.25.0,<1.0"),
            ("datasets", ">=2.11.0,<4.4.0"),
        ]
    
    all_ok = True
    for pkg, spec in checks:
        ok, msg = check_package(pkg, spec)
        if not ok:
            print(f"FAIL: {msg}", file=sys.stderr)
            all_ok = False
        else:
            print(f"OK: {pkg} {msg}")
    
    # Check PyTorch separately (more complex)
    pytorch_ok, pytorch_msg = check_pytorch_cuda()
    if not pytorch_ok:
        print(f"FAIL: PyTorch - {pytorch_msg}", file=sys.stderr)
        all_ok = False
    else:
        print(f"OK: PyTorch - {pytorch_msg}")
    
    # Check PySide6/shiboken6 (matches launcher health check - critical for preventing repair loops)
    pyside6_ok, pyside6_msg = check_pyside6()
    if not pyside6_ok:
        print(f"FAIL: PySide6 - {pyside6_msg}", file=sys.stderr)
        all_ok = False
    else:
        print(f"OK: PySide6 - {pyside6_msg}")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(verify_all())
