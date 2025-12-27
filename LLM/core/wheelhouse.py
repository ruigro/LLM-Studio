#!/usr/bin/env python3
"""
Wheelhouse Manager - Downloads and verifies Python wheels
Part of the Immutable Installer system
"""

import sys
import json
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


class BlacklistedPackageError(Exception):
    """Raised when a blacklisted package is found in wheelhouse"""
    pass


class WheelhouseManager:
    """
    Manages wheel download and verification for offline installation.
    Downloads exact wheels to wheelhouse, checks for blacklisted packages.
    """
    
    def __init__(self, manifest_path: Path, wheelhouse_dir: Path):
        """
        Initialize wheelhouse manager.
        
        Args:
            manifest_path: Path to dependencies.json manifest
            wheelhouse_dir: Path to wheelhouse directory for storing wheels
        """
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        
        self.wheelhouse = wheelhouse_dir
        self.wheelhouse.mkdir(parents=True, exist_ok=True)
        
        # Windows subprocess flags to prevent CMD window flashing
        self.subprocess_flags = {}
        if sys.platform == 'win32':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            self.subprocess_flags = {
                'startupinfo': startupinfo,
                'creationflags': subprocess.CREATE_NO_WINDOW
            }
    
    def log(self, message: str):
        """Log message to console"""
        print(f"[WHEELHOUSE] {message}")
    
    def prepare_wheelhouse(self, cuda_config: str, python_version: Tuple[int, int], package_versions: dict = None) -> Tuple[bool, str]:
        """
        Download all wheels to wheelhouse with exact versions.
        
        Args:
            cuda_config: CUDA configuration key (e.g., "cu124")
            python_version: Python version tuple (e.g., (3, 12))
            package_versions: Optional dict of {package_name: exact_version} from ProfileSelector.
                            If provided, uses these instead of manifest.
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            self.log(f"Preparing wheelhouse for {cuda_config}, Python {python_version[0]}.{python_version[1]}")
            
            # Clear existing wheelhouse
            self._clear_wheelhouse()
            
            # If package_versions provided, use hardware-adaptive installation
            if package_versions:
                return self._prepare_from_profile(cuda_config, python_version, package_versions)
            
            # Otherwise, use legacy manifest-based installation
            return self._prepare_from_manifest(cuda_config, python_version)
            
            # Get CUDA-specific packages
            if cuda_config not in self.manifest["cuda_configs"]:
                return False, f"Unknown CUDA config: {cuda_config}"
            
            cuda_packages = self.manifest["cuda_configs"][cuda_config]["packages"]
            torch_index = self.manifest["cuda_configs"][cuda_config]["torch_index"]
            
            # Phase 1: Download torch stack from CUDA index
            self.log("Phase 1: Downloading torch stack from CUDA index")
            for pkg_name, pkg_version in cuda_packages.items():
                self.log(f"  Downloading {pkg_name}=={pkg_version}")
                success, error = self._download_wheel(
                    package=f"{pkg_name}=={pkg_version}",
                    index_url=torch_index,
                    no_deps=True,  # Critical: no deps for torch
                    python_version=python_version
                )
                if not success:
                    return False, f"Failed to download {pkg_name}: {error}"
            
            # Phase 2: Download core dependencies in order
            self.log("Phase 2: Downloading core dependencies")
            deps = sorted(self.manifest["core_dependencies"], key=lambda x: x["order"])
            
            for dep in deps:
                pkg_name = dep["name"]
                
                # Skip CUDA packages (already downloaded)
                if dep["version"] == "FROM_CUDA_CONFIG":
                    continue
                
                # Skip platform-specific packages if not on that platform
                if "platform" in dep and dep["platform"] != sys.platform:
                    self.log(f"  Skipping {pkg_name} (platform: {dep['platform']}, current: {sys.platform})")
                    continue
                
                version_spec = dep["version"]
                
                # Determine install args
                install_args = dep.get("install_args", [])
                no_deps = "--no-deps" in install_args
                
                self.log(f"  Downloading {pkg_name}{version_spec}")
                success, error = self._download_wheel(
                    package=f"{pkg_name}{version_spec}",
                    index_url=None,  # Use default PyPI
                    no_deps=no_deps,
                    python_version=python_version
                )
                if not success:
                    # Non-critical packages can fail
                    if dep.get("critical", False):
                        return False, f"Failed to download critical package {pkg_name}: {error}"
                    else:
                        self.log(f"  WARNING: Failed to download optional package {pkg_name}: {error}")
            
            # Phase 3: Verify no blacklisted packages
            self.log("Phase 3: Verifying no blacklisted packages")
            success, error = self._verify_no_blacklist()
            if not success:
                return False, error
            
            # Phase 4: Verify all critical packages present
            self.log("Phase 4: Verifying all wheels present")
            success, error = self._verify_wheels_present(deps, cuda_packages)
            if not success:
                return False, error
            
            self.log("✓ Wheelhouse preparation complete")
            return True, ""
            
        except Exception as e:
            return False, f"Wheelhouse preparation exception: {str(e)}"
    
    def _prepare_from_profile(self, cuda_config: str, python_version: Tuple[int, int], package_versions: dict) -> Tuple[bool, str]:
        """
        Prepare wheelhouse from ProfileSelector package versions (hardware-adaptive).
        
        Args:
            cuda_config: CUDA config for torch index URL
            python_version: Python version tuple
            package_versions: Dict of {package_name: exact_version}
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Get torch index URL
            torch_index = None
            if cuda_config in self.manifest.get("cuda_configs", {}):
                torch_index = self.manifest["cuda_configs"][cuda_config]["torch_index"]
            
            # Phase 1: Download torch stack from CUDA index
            self.log("Phase 1: Downloading torch stack from CUDA index")
            torch_packages = ["torch", "torchvision", "torchaudio"]
            
            for pkg_name in torch_packages:
                if pkg_name in package_versions:
                    pkg_version = package_versions[pkg_name]
                    self.log(f"  Downloading {pkg_name}=={pkg_version}")
                    success, error = self._download_wheel(
                        package=f"{pkg_name}=={pkg_version}",
                        index_url=torch_index,
                        no_deps=True,  # Critical: no deps for torch
                        python_version=python_version
                    )
                    if not success:
                        return False, f"Failed to download {pkg_name}: {error}"
            
            # Phase 2: Download all other packages
            self.log("Phase 2: Downloading all other packages")
            
            # Sort packages by priority (critical packages first)
            critical_order = [
                "typing-extensions", "numpy", "safetensors", "tokenizers",
                "huggingface-hub", "transformers", "peft", "bitsandbytes",
                "accelerate", "datasets"
            ]
            
            remaining = [pkg for pkg in package_versions if pkg not in torch_packages]
            ordered_packages = []
            
            # Add critical packages in order
            for critical in critical_order:
                if critical in remaining:
                    ordered_packages.append(critical)
            
            # Add remaining packages
            for pkg in remaining:
                if pkg not in ordered_packages:
                    ordered_packages.append(pkg)
            
            for pkg_name in ordered_packages:
                pkg_version = package_versions[pkg_name]
                self.log(f"  Downloading {pkg_name}=={pkg_version}")
                
                success, error = self._download_wheel(
                    package=f"{pkg_name}=={pkg_version}",
                    index_url=None,  # Use PyPI
                    no_deps=False,  # Allow deps for non-torch packages
                    python_version=python_version
                )
                if not success:
                    # Check if it's critical
                    if pkg_name in critical_order:
                        return False, f"Failed to download critical package {pkg_name}: {error}"
                    else:
                        self.log(f"  WARNING: Failed to download optional package {pkg_name}: {error}")
            
            # Phase 3: Verify no blacklisted packages
            self.log("Phase 3: Verifying no blacklisted packages")
            success, error = self._verify_no_blacklist()
            if not success:
                return False, error
            
            # Phase 4: Count wheels
            wheel_count = len(list(self.wheelhouse.glob("*.whl")))
            self.log(f"Phase 4: Verification complete ({wheel_count} wheels)")
            self.log("✓ Wheelhouse preparation complete")
            
            return True, ""
            
        except Exception as e:
            return False, f"Profile-based preparation failed: {str(e)}"
    
    def _prepare_from_manifest(self, cuda_config: str, python_version: Tuple[int, int]) -> Tuple[bool, str]:
        """
        Legacy manifest-based preparation (for backwards compatibility).
        """
        try:
            # Get CUDA-specific packages
            if cuda_config not in self.manifest["cuda_configs"]:
                return False, f"Unknown CUDA config: {cuda_config}"
            
            cuda_packages = self.manifest["cuda_configs"][cuda_config]["packages"]
            torch_index = self.manifest["cuda_configs"][cuda_config]["torch_index"]
            
            # Phase 1: Download torch stack from CUDA index
            self.log("Phase 1: Downloading torch stack from CUDA index")
            for pkg_name, pkg_version in cuda_packages.items():
                self.log(f"  Downloading {pkg_name}=={pkg_version}")
                success, error = self._download_wheel(
                    package=f"{pkg_name}=={pkg_version}",
                    index_url=torch_index,
                    no_deps=True,
                    python_version=python_version
                )
                if not success:
                    return False, f"Failed to download {pkg_name}: {error}"
            
            # Phase 2: Download core dependencies
            self.log("Phase 2: Downloading core dependencies")
            deps = sorted(self.manifest["core_dependencies"], key=lambda x: x["order"])
            
            for dep in deps:
                pkg_name = dep["name"]
                
                if dep["version"] == "FROM_CUDA_CONFIG":
                    continue
                
                if "platform" in dep and dep["platform"] != sys.platform:
                    self.log(f"  Skipping {pkg_name} (platform mismatch)")
                    continue
                
                version_spec = dep["version"]
                no_deps = "--no-deps" in dep.get("install_args", [])
                
                self.log(f"  Downloading {pkg_name}{version_spec}")
                success, error = self._download_wheel(
                    package=f"{pkg_name}{version_spec}",
                    index_url=None,
                    no_deps=no_deps,
                    python_version=python_version
                )
                if not success:
                    if dep.get("critical", False):
                        return False, f"Failed to download critical package {pkg_name}: {error}"
                    else:
                        self.log(f"  WARNING: Failed to download optional package {pkg_name}: {error}")
            
            # Phase 3: Verify no blacklisted packages
            self.log("Phase 3: Verifying no blacklisted packages")
            success, error = self._verify_no_blacklist()
            if not success:
                return False, error
            
            # Phase 4: Verify all critical packages present
            self.log("Phase 4: Verifying all wheels present")
            success, error = self._verify_wheels_present(deps, cuda_packages)
            if not success:
                return False, error
            
            self.log("✓ Wheelhouse preparation complete")
            return True, ""
            
        except Exception as e:
            return False, f"Manifest-based preparation failed: {str(e)}"
    
    def _clear_wheelhouse(self):
        """Clear existing wheelhouse"""
        if self.wheelhouse.exists():
            for item in self.wheelhouse.glob("*"):
                if item.is_file():
                    item.unlink()
            self.log("Existing wheelhouse cleared")
    
    def _download_wheel(
        self, 
        package: str, 
        index_url: Optional[str] = None, 
        no_deps: bool = False,
        python_version: Tuple[int, int] = None
    ) -> Tuple[bool, str]:
        """
        Download single package to wheelhouse.
        
        Args:
            package: Package specifier (e.g., "torch==2.5.1+cu124")
            index_url: Custom index URL (for torch)
            no_deps: If True, don't download dependencies
            python_version: Target Python version
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        cmd = [
            sys.executable, "-m", "pip", "download",
            "--dest", str(self.wheelhouse),
            "--no-cache-dir"  # Critical: don't use cache
        ]
        
        if index_url:
            cmd.extend(["--index-url", index_url])
        
        if no_deps:
            cmd.append("--no-deps")
        
        # Add python version constraint if specified
        if python_version:
            py_ver = f"{python_version[0]}.{python_version[1]}"
            cmd.extend([
                "--python-version", py_ver,
                "--only-binary=:all:"  # Required when using --python-version
            ])
        
        cmd.append(package)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes max per package
                **self.subprocess_flags
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                return False, error_msg[:500]  # Truncate long errors
            
            return True, ""
            
        except subprocess.TimeoutExpired:
            return False, f"Download timeout for {package}"
        except Exception as e:
            return False, str(e)
    
    def _verify_no_blacklist(self) -> Tuple[bool, str]:
        """
        Verify no blacklisted packages in wheelhouse.
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        blacklist = set(pkg.lower().replace("_", "-") for pkg in self.manifest["global_blacklist"])
        
        found_blacklisted = []
        
        for wheel in self.wheelhouse.glob("*.whl"):
            # Parse wheel filename: package-version-pyver-abi-platform.whl
            # Example: torch-2.5.1+cu124-cp312-cp312-win_amd64.whl
            wheel_name_parts = wheel.stem.split("-")
            if len(wheel_name_parts) >= 2:
                pkg_name = wheel_name_parts[0].lower().replace("_", "-")
                
                if pkg_name in blacklist:
                    found_blacklisted.append((pkg_name, wheel.name))
        
        if found_blacklisted:
            error_lines = [
                "CRITICAL: Blacklisted packages found in wheelhouse!",
                "These packages cause conflicts and must not be installed:",
                ""
            ]
            for pkg_name, wheel_file in found_blacklisted:
                error_lines.append(f"  - {pkg_name} ({wheel_file})")
            
            error_lines.extend([
                "",
                "These were likely pulled in as dependencies.",
                "The installation cannot proceed with these packages.",
                "Clearing wheelhouse and aborting."
            ])
            
            # Clear wheelhouse to prevent accidental use
            self._clear_wheelhouse()
            
            return False, "\n".join(error_lines)
        
        self.log(f"✓ No blacklisted packages found ({len(blacklist)} checked)")
        return True, ""
    
    def _verify_wheels_present(self, deps: List[Dict], cuda_packages: Dict[str, str]) -> Tuple[bool, str]:
        """
        Verify all critical packages have wheels in wheelhouse.
        
        Args:
            deps: List of dependency dicts
            cuda_packages: Dict of CUDA package names to versions
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        wheels_present = set()
        
        # Get list of all wheel files
        for wheel in self.wheelhouse.glob("*.whl"):
            wheel_name_parts = wheel.stem.split("-")
            if len(wheel_name_parts) >= 1:
                pkg_name = wheel_name_parts[0].lower().replace("_", "-")
                wheels_present.add(pkg_name)
        
        # Check critical packages
        missing_critical = []
        
        # Check CUDA packages
        for pkg_name in cuda_packages.keys():
            pkg_normalized = pkg_name.lower().replace("_", "-")
            if pkg_normalized not in wheels_present:
                missing_critical.append(pkg_name)
        
        # Check core dependencies
        for dep in deps:
            if dep.get("critical", False):
                pkg_name = dep["name"]
                pkg_normalized = pkg_name.lower().replace("_", "-")
                
                # Skip FROM_CUDA_CONFIG (already checked above)
                if dep["version"] == "FROM_CUDA_CONFIG":
                    continue
                
                # Skip platform-specific
                if "platform" in dep and dep["platform"] != sys.platform:
                    continue
                
                if pkg_normalized not in wheels_present:
                    missing_critical.append(pkg_name)
        
        if missing_critical:
            return False, f"Missing critical packages in wheelhouse: {', '.join(missing_critical)}"
        
        self.log(f"✓ All critical packages present ({len(wheels_present)} wheels total)")
        return True, ""
    
    def get_wheel_info(self, package_name: str) -> Optional[Path]:
        """
        Get wheel file path for a package.
        
        Args:
            package_name: Package name
        
        Returns:
            Path to wheel file or None if not found
        """
        pkg_normalized = package_name.lower().replace("_", "-")
        
        for wheel in self.wheelhouse.glob("*.whl"):
            wheel_name_parts = wheel.stem.split("-")
            if len(wheel_name_parts) >= 1:
                wheel_pkg = wheel_name_parts[0].lower().replace("_", "-")
                if wheel_pkg == pkg_normalized:
                    return wheel
        
        return None


def main():
    """Test wheelhouse manager"""
    import sys
    from pathlib import Path
    
    # Test setup
    manifest = Path(__file__).parent.parent / "metadata" / "dependencies.json"
    wheelhouse = Path(__file__).parent.parent / "wheelhouse_test"
    
    if not manifest.exists():
        print(f"ERROR: Manifest not found at {manifest}")
        sys.exit(1)
    
    manager = WheelhouseManager(manifest, wheelhouse)
    
    print("Testing wheelhouse preparation...")
    success, error = manager.prepare_wheelhouse("cu124", (3, 12))
    
    if success:
        print("✓ Test passed!")
    else:
        print(f"✗ Test failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()

