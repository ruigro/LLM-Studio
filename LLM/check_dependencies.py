#!/usr/bin/env python3
"""
Lightweight dependency health check for launcher
Verifies critical packages are installed with correct versions
"""

import sys
import subprocess
from importlib.metadata import version, PackageNotFoundError

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


def verify_all():
    """Verify all critical dependencies"""
    checks = [
        ("numpy", "<2.0.0"),
        ("transformers", ">=4.51.3,!=4.52.*,!=4.53.*,!=4.54.*,!=4.55.*,!=4.57.0,<4.58"),
        ("tokenizers", ">=0.22.0,<=0.23.0"),
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
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(verify_all())

