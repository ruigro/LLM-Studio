#!/usr/bin/env python3
"""
Post-Installation Verification Script
Tests installation and shows results
"""

import sys
import os
from pathlib import Path
from system_detector import SystemDetector

def verify_installation():
    """Verify that installation is correct"""
    print("=" * 60)
    print("LLM Fine-tuning Studio - Installation Verification")
    print("=" * 60)
    print()
    
    detector = SystemDetector()
    results = detector.detect_all()
    
    all_ok = True
    
    # Check Python
    python_info = results.get("python", {})
    if python_info.get("found"):
        print(f"✓ Python {python_info.get('version')} found")
        print(f"  Location: {python_info.get('executable')}")
        if python_info.get("pip_available"):
            print("  ✓ pip available")
        else:
            print("  ✗ pip not available")
            all_ok = False
    else:
        print("✗ Python not found")
        all_ok = False
    
    print()
    
    # Check PyTorch
    pytorch_info = results.get("pytorch", {})
    if pytorch_info.get("found"):
        print(f"✓ PyTorch {pytorch_info.get('version')} installed")
        if pytorch_info.get("cuda_available"):
            print(f"  ✓ CUDA available (Version {pytorch_info.get('cuda_version')})")
        else:
            print("  ⚠ CUDA not available (CPU mode)")
    else:
        print("✗ PyTorch not installed")
        print("  Run: pip install torch")
        all_ok = False
    
    print()
    
    # Check CUDA
    cuda_info = results.get("cuda", {})
    if cuda_info.get("found"):
        print(f"✓ CUDA {cuda_info.get('version')} detected")
        gpus = cuda_info.get("gpus", [])
        if gpus:
            print(f"  Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"    - {gpu.get('name')} ({gpu.get('memory')})")
    else:
        print("⚠ CUDA not detected (CPU mode will be used)")
    
    print()
    
    # Check Visual C++ Redistributables (Windows)
    if sys.platform == "win32":
        vcredist_info = results.get("vcredist", {})
        if vcredist_info.get("found"):
            print("✓ Visual C++ Redistributables found")
        else:
            print("⚠ Visual C++ Redistributables not found")
            print("  Some features may not work correctly")
            print("  Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    
    print()
    
    # Check application files
    app_dir = Path(__file__).parent
    required_files = ["gui.py", "finetune.py", "run_adapter.py", "validate_prompts.py"]
    print("Checking application files...")
    for file in required_files:
        file_path = app_dir / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} not found")
            all_ok = False
    
    print()
    
    # Check dependencies
    print("Checking key dependencies...")
    dependencies = {
        "streamlit": "Streamlit",
        "transformers": "Transformers",
        "peft": "PEFT",
        "accelerate": "Accelerate",
        "huggingface_hub": "Hugging Face Hub"
    }
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} not installed")
            all_ok = False
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✓ Installation verification passed!")
        print("You can now launch the application.")
    else:
        print("⚠ Installation verification found some issues.")
        print("Please install missing components before using the application.")
    
    print("=" * 60)
    
    return all_ok

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)

