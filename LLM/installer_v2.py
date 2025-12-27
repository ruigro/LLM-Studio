#!/usr/bin/env python3
"""
Installer V2 - Main Coordinator
Immutable installer for LLM Fine-tuning Studio
"""

import sys
import os
import json
from pathlib import Path
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.wheelhouse import WheelhouseManager
from core.immutable_installer import ImmutableInstaller, InstallationFailed
from system_detector import SystemDetector


class InstallerV2:
    """
    Main installer coordinator for immutable installation.
    Orchestrates detection, wheel download, and atomic installation.
    """
    
    def __init__(self):
        """Initialize installer coordinator"""
        self.root = Path(__file__).parent
        self.manifest_path = self.root / "metadata" / "dependencies.json"
        self.compat_matrix_path = self.root / "metadata" / "compatibility_matrix.json"
        self.wheelhouse = self.root / "wheelhouse"
        self.venv = self.root / ".venv"
        
        # Verify manifest exists and load it
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        
        # Check if compatibility matrix exists (for hardware-adaptive mode)
        self.use_adaptive = self.compat_matrix_path.exists()
        if not self.use_adaptive:
            self.log("⚠ Compatibility matrix not found. Using legacy fixed-version mode.")
    
    def log(self, message: str):
        """Log message to console"""
        print(f"[INSTALLER-V2] {message}")
    
    def install(self, skip_wheelhouse: bool = False) -> bool:
        """
        Run full installation.
        
        Args:
            skip_wheelhouse: If True, skip wheelhouse preparation (use existing)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.log("=" * 60)
            self.log("LLM Fine-tuning Studio - Immutable Installer v2.0")
            self.log("=" * 60)
            
            # PHASE 0: Detection
            self.log("\nPHASE 0: Hardware and Platform Detection")
            self.log("-" * 60)
            
            detector = SystemDetector()
            results = detector.detect_all()
            
            # Display detection results
            self._display_detection_results(results)
            
            # Validate Python version BEFORE determining CUDA config
            python_version = (sys.version_info.major, sys.version_info.minor)
            min_py = tuple(map(int, self.manifest["python_min"].split('.')))
            max_py = tuple(map(int, self.manifest["python_max"].split('.')))
            
            if python_version < min_py or python_version > max_py:
                raise ValueError(
                    f"\n✗ Python {python_version[0]}.{python_version[1]} is not supported.\n"
                    f"  Required: Python {self.manifest['python_min']} - {self.manifest['python_max']}\n"
                    f"  Current: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n\n"
                    f"  Please use a supported Python version.\n"
                    f"  You can install Python 3.10, 3.11, or 3.12 from:\n"
                    f"  https://www.python.org/downloads/"
                )
            
            # Hardware-adaptive mode: Use ProfileSelector
            if self.use_adaptive:
                self.log("\n🎯 Using hardware-adaptive installation")
                
                from core.profile_selector import ProfileSelector
                
                # Get hardware profile
                hw_profile = detector.get_hardware_profile()
                
                # Select optimal profile
                selector = ProfileSelector(self.compat_matrix_path)
                try:
                    profile_name, package_versions, warnings = selector.select_profile(hw_profile)
                    
                    self.log(f"\n✓ Selected profile: {profile_name}")
                    self.log(f"  {selector.get_profile_description(profile_name)}")
                    
                    for warning in warnings:
                        self.log(f"  ⚠ {warning}")
                    
                    # Determine CUDA config from selected profile
                    cuda_config = self._extract_cuda_config(package_versions.get("torch", ""))
                    
                except Exception as e:
                    raise ValueError(f"Profile selection failed: {str(e)}")
            
            # Legacy mode: Use fixed versions from manifest
            else:
                self.log("\n⚠ Using legacy fixed-version installation")
                cuda_config = self._determine_cuda_config(results)
                package_versions = None  # Will use manifest
            
            self.log(f"\n✓ Target configuration: {cuda_config}")
            
            # PHASE 1: Prepare wheelhouse (unless skipped)
            if not skip_wheelhouse:
                self.log("\nPHASE 1: Wheelhouse Preparation")
                self.log("-" * 60)
                
                wheelhouse_mgr = WheelhouseManager(self.manifest_path, self.wheelhouse)
                
                python_version = (sys.version_info.major, sys.version_info.minor)
                success, error = wheelhouse_mgr.prepare_wheelhouse(
                    cuda_config, 
                    python_version,
                    package_versions  # Pass hardware-specific versions or None
                )
                
                if not success:
                    self.log(f"\n✗ Wheelhouse preparation failed:")
                    self.log(f"  {error}")
                    return False
                
                self.log("\n✓ Wheelhouse ready")
            else:
                self.log("\nPHASE 1: Wheelhouse Preparation (SKIPPED)")
                self.log(f"  Using existing wheelhouse at: {self.wheelhouse}")
                
                if not self.wheelhouse.exists():
                    self.log("\n✗ Wheelhouse directory not found!")
                    return False
                
                wheel_count = len(list(self.wheelhouse.glob("*.whl")))
                if wheel_count == 0:
                    self.log("\n✗ Wheelhouse is empty!")
                    return False
                
                self.log(f"  ✓ Found {wheel_count} wheels")
            
            # PHASE 2-6: Install
            self.log("\nPHASE 2-6: Environment Installation")
            self.log("-" * 60)
            
            installer = ImmutableInstaller(self.venv, self.wheelhouse, self.manifest_path)
            success, error = installer.install(cuda_config)
            
            if not success:
                self.log(f"\n✗ Installation failed:")
                self.log(f"  {error}")
                return False
            
            self.log("\n" + "=" * 60)
            self.log("✓ Installation complete!")
            self.log("=" * 60)
            self.log(f"\nVirtual environment: {self.venv}")
            self.log(f"Python executable: {self.venv / 'Scripts' / 'python.exe' if sys.platform == 'win32' else self.venv / 'bin' / 'python'}")
            self.log("\nYou can now launch the application.")
            
            return True
            
        except KeyboardInterrupt:
            self.log("\n\nInstallation interrupted by user")
            return False
        except Exception as e:
            self.log(f"\n✗ Installation failed with exception:")
            self.log(f"  {type(e).__name__}: {str(e)}")
            
            import traceback
            self.log("\nFull traceback:")
            self.log(traceback.format_exc())
            
            return False
    
    def _display_detection_results(self, results: dict):
        """Display hardware detection results"""
        # Python
        python_info = results.get("python", {})
        if python_info.get("found"):
            self.log(f"  Python: {python_info.get('version')} at {python_info.get('executable')}")
        
        # CUDA
        cuda_info = results.get("cuda", {})
        if cuda_info.get("found"):
            gpus = cuda_info.get("gpus", [])
            self.log(f"  CUDA: {cuda_info.get('version')} with {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                self.log(f"    GPU {i}: {gpu.get('name')} ({gpu.get('memory_mb', 0)} MB)")
        else:
            self.log("  CUDA: Not detected")
        
        # Hardware
        hw_info = results.get("hardware", {})
        cpu = hw_info.get("cpu", "Unknown")
        ram_gb = hw_info.get("ram_gb", 0)
        self.log(f"  CPU: {cpu}")
        self.log(f"  RAM: {ram_gb:.1f} GB")
    
    def _determine_cuda_config(self, detection_results: dict) -> str:
        """
        Determine which CUDA config to use based on detection.
        
        Args:
            detection_results: Results from SystemDetector
        
        Returns:
            CUDA config key (e.g., "cu124")
        
        Raises:
            ValueError: If no suitable CUDA configuration found
        """
        cuda_info = detection_results.get("cuda", {})
        
        if not cuda_info.get("found"):
            raise ValueError(
                "No CUDA GPU detected. This application requires CUDA.\n"
                "Please ensure:\n"
                "  1. NVIDIA GPU is installed\n"
                "  2. NVIDIA drivers are up to date\n"
                "  3. CUDA toolkit is installed"
            )
        
        cuda_version = cuda_info.get("version", "")
        driver_version = cuda_info.get("driver_version", "")
        
        # Handle missing CUDA version - try to infer from driver
        if not cuda_version or cuda_version == "None":
            if driver_version:
                try:
                    driver_major = int(driver_version.split('.')[0])
                    # Map driver version to CUDA version
                    # Driver 560+ supports CUDA 12.6+
                    # Driver 550+ supports CUDA 12.4+
                    # Driver 520+ supports CUDA 12.1+
                    if driver_major >= 550:
                        cuda_version = "12.4"
                        self.log(f"  Inferred CUDA 12.4+ from driver {driver_version}")
                    elif driver_major >= 520:
                        cuda_version = "12.1"
                        self.log(f"  Inferred CUDA 12.1+ from driver {driver_version}")
                    elif driver_major >= 470:
                        cuda_version = "11.8"
                        self.log(f"  Inferred CUDA 11.8+ from driver {driver_version}")
                    else:
                        raise ValueError(
                            f"Driver version {driver_version} is too old.\n"
                            f"Please update NVIDIA drivers to version 520+ for CUDA 12 support."
                        )
                except ValueError as e:
                    raise e
                except Exception:
                    pass
            
            if not cuda_version or cuda_version == "None":
                raise ValueError(
                    "Could not detect CUDA version. This application requires CUDA.\n"
                    "Please ensure:\n"
                    "  1. NVIDIA GPU is installed\n"
                    "  2. NVIDIA drivers are up to date (version 520+ recommended)\n"
                    "  3. Run 'nvidia-smi' in terminal to verify driver installation"
                )
        
        # Map CUDA version to config
        if cuda_version.startswith("12.6") or cuda_version.startswith("12.5") or cuda_version.startswith("12.4"):
            return "cu124"
        elif cuda_version.startswith("12.3") or cuda_version.startswith("12.2") or cuda_version.startswith("12.1"):
            return "cu121"
        elif cuda_version.startswith("11.8"):
            return "cu118"
        else:
            # Try to find closest match
            try:
                major, minor = map(int, cuda_version.split(".")[:2])
                if major == 12:
                    # For CUDA 12.x, use cu124 as default
                    self.log(f"  WARNING: CUDA {cuda_version} not explicitly supported, using cu124")
                    return "cu124"
                elif major == 11 and minor >= 8:
                    self.log(f"  WARNING: CUDA {cuda_version} not explicitly supported, using cu118")
                    return "cu118"
            except:
                pass
            
            raise ValueError(
                f"Unsupported CUDA version: {cuda_version}\n"
                f"Supported versions: 11.8, 12.1-12.3, 12.4-12.6\n"
                f"Please update your NVIDIA drivers"
            )
    
    def _extract_cuda_config(self, torch_version: str) -> str:
        """Extract CUDA config from torch version string like '2.5.1+cu124' → 'cu124'"""
        if "+cu" in torch_version:
            parts = torch_version.split("+cu")
            if len(parts) > 1:
                return "cu" + parts[1][:3]  # Extract '124' from 'cu124' or just 'cu124'
        # Default fallback based on most common
        return "cu121"
    
    def verify_installation(self) -> bool:
        """
        Verify an existing installation.
        
        Returns:
            True if installation is valid, False otherwise
        """
        try:
            self.log("Verifying installation...")
            
            # Check venv exists
            if not self.venv.exists():
                self.log("✗ Virtual environment not found")
                return False
            
            # Get venv Python
            if sys.platform == 'win32':
                venv_python = self.venv / "Scripts" / "python.exe"
            else:
                venv_python = self.venv / "bin" / "python"
            
            if not venv_python.exists():
                self.log("✗ Python executable not found in venv")
                return False
            
            # Run verification
            from core.verification import VerificationSystem
            
            verifier = VerificationSystem(self.manifest, venv_python)
            success, error = verifier.run_quick_verify()
            
            if success:
                self.log("✓ Installation verified")
                return True
            else:
                self.log(f"✗ Verification failed: {error}")
                return False
                
        except Exception as e:
            self.log(f"✗ Verification error: {str(e)}")
            return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM Fine-tuning Studio - Immutable Installer v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python installer_v2.py                    # Full installation
  python installer_v2.py --skip-wheelhouse  # Skip download, use existing wheels
  python installer_v2.py --verify           # Verify existing installation
"""
    )
    
    parser.add_argument(
        "--skip-wheelhouse",
        action="store_true",
        help="Skip wheelhouse preparation (use existing wheels)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing installation instead of installing"
    )
    
    args = parser.parse_args()
    
    try:
        installer = InstallerV2()
        
        if args.verify:
            # Verification mode
            success = installer.verify_installation()
        else:
            # Installation mode
            success = installer.install(skip_wheelhouse=args.skip_wheelhouse)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

