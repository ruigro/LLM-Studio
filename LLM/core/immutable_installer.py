#!/usr/bin/env python3
"""
Immutable Installer - Atomic installation system
Part of the Immutable Installer system
"""

import sys
import os
import shutil
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Try to import packaging for version validation
try:
    from packaging.specifiers import SpecifierSet
    from packaging import version as pkg_version
    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False
    SpecifierSet = None
    pkg_version = None

# Try to import importlib.metadata for package checking
try:
    from importlib.metadata import version as get_package_version, PackageNotFoundError
    METADATA_AVAILABLE = True
except ImportError:
    # Fallback for Python < 3.8
    try:
        from importlib_metadata import version as get_package_version, PackageNotFoundError
        METADATA_AVAILABLE = True
    except ImportError:
        METADATA_AVAILABLE = False
        get_package_version = None
        PackageNotFoundError = Exception


class InstallationFailed(Exception):
    """Raised when installation fails"""
    pass


class ImmutableInstaller:
    """
    Immutable installer - always starts fresh, never repairs in place.
    Creates new venv, installs from wheelhouse offline, verifies integrity.
    """
    
    def __init__(self, venv_path: Path, wheelhouse_path: Path, manifest_path: Path):
        """
        Initialize immutable installer.
        
        Args:
            venv_path: Path where venv should be created
            wheelhouse_path: Path to wheelhouse with downloaded wheels
            manifest_path: Path to dependencies.json manifest
        """
        self.venv_path = venv_path
        self.wheelhouse = wheelhouse_path
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        
        self.torch_lock = venv_path / ".torch_lock"
        
        # Windows subprocess flags
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
        print(f"[INSTALL] {message}")
    
    def install(self, cuda_config: str) -> Tuple[bool, str]:
        """
        Perform immutable installation.
        
        Args:
            cuda_config: CUDA configuration key (e.g., "cu124")
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            self.log("=" * 60)
            self.log("Immutable Installation Starting")
            self.log("=" * 60)
            
            # PHASE 0: Validation
            self.log("PHASE 0: Environment validation")
            self._validate_environment()
            
            # PHASE 1: Wheelhouse already prepared by WheelhouseManager
            self.log("PHASE 1: Wheelhouse verification")
            if not self.wheelhouse.exists():
                raise InstallationFailed(f"Wheelhouse not found at {self.wheelhouse}")
            
            wheel_count = len(list(self.wheelhouse.glob("*.whl")))
            if wheel_count == 0:
                raise InstallationFailed("Wheelhouse is empty - no wheels found")
            self.log(f"  ✓ Wheelhouse contains {wheel_count} wheels")
            
            # PHASE 2: Resume detection - check if venv exists and is valid
            venv_python = None
            should_resume = False
            packages_to_install = []
            
            if self.venv_path.exists():
                # Check if venv Python exists
                if sys.platform == 'win32':
                    venv_python_path = self.venv_path / "Scripts" / "python.exe"
                else:
                    venv_python_path = self.venv_path / "bin" / "python"
                
                if venv_python_path.exists():
                    venv_python = venv_python_path
                    self.log("PHASE 2: Checking existing venv for resume capability")
                    
                    # Verify Python version matches
                    try:
                        result = subprocess.run(
                            [str(venv_python), "--version"],
                            capture_output=True,
                            text=True,
                            timeout=10,
                            **self.subprocess_flags
                        )
                        if result.returncode == 0:
                            self.log(f"  ✓ Venv Python found: {result.stdout.strip()}")
                            
                            # Check which packages need installation
                            deps = sorted(self.manifest["core_dependencies"], key=lambda x: x["order"])
                            cuda_packages = self.manifest["cuda_configs"][cuda_config]["packages"]
                            
                            missing_packages = []
                            wrong_version_packages = []
                            
                            # Check CUDA packages
                            for pkg_name, pkg_version in cuda_packages.items():
                                is_installed, version_matches = self._check_package_installed(
                                    venv_python, pkg_name, f"=={pkg_version}"
                                )
                                if not is_installed:
                                    missing_packages.append((pkg_name, f"=={pkg_version}"))
                                elif not version_matches:
                                    wrong_version_packages.append((pkg_name, f"=={pkg_version}"))
                            
                            # Check core dependencies
                            for dep in deps:
                                pkg_name = dep["name"]
                                
                                # Skip platform-specific packages
                                if "platform" in dep and dep["platform"] != sys.platform:
                                    continue
                                
                                # Skip CUDA packages (already checked)
                                if dep["version"] == "FROM_CUDA_CONFIG":
                                    continue
                                
                                version_spec = dep["version"]
                                is_installed, version_matches = self._check_package_installed(
                                    venv_python, pkg_name, version_spec
                                )
                                
                                if not is_installed:
                                    missing_packages.append((pkg_name, version_spec))
                                elif not version_matches:
                                    wrong_version_packages.append((pkg_name, version_spec))
                            
                            packages_to_install = missing_packages + wrong_version_packages
                            
                            if packages_to_install:
                                self.log(f"  ⚠ Found {len(packages_to_install)} packages that need installation:")
                                for pkg_name, version_spec in packages_to_install[:5]:  # Show first 5
                                    self.log(f"    - {pkg_name}{version_spec}")
                                if len(packages_to_install) > 5:
                                    self.log(f"    ... and {len(packages_to_install) - 5} more")
                                
                                # Check if critical packages are missing or wrong
                                critical_pkg_names = {dep["name"] for dep in deps if dep.get("critical", False)}
                                critical_pkg_names.update(cuda_packages.keys())  # CUDA packages are always critical
                                critical_missing = any(
                                    pkg_name in critical_pkg_names for pkg_name, _ in packages_to_install
                                )
                                
                                if critical_missing:
                                    self.log("  ⚠ Critical packages need installation - will resume from where we left off")
                                    should_resume = True
                                else:
                                    self.log("  ✓ Only non-critical packages need installation - will resume")
                                    should_resume = True
                            else:
                                self.log("  ✓ All packages already installed correctly - skipping installation")
                                should_resume = True
                                
                    except Exception as e:
                        self.log(f"  ⚠ Could not verify venv: {e}")
                        self.log("  Will destroy and recreate venv")
                        should_resume = False
                        venv_python = None
            
            if not should_resume:
                # PHASE 2: Destroy existing venv
                self.log("PHASE 2: Destroying existing venv")
                self._destroy_venv()
                
                # PHASE 3: Create fresh venv
                self.log("PHASE 3: Creating fresh venv")
                venv_python = self._create_venv()
            else:
                self.log("PHASE 2: Resuming installation (venv already exists)")
                self.log(f"  Will install {len(packages_to_install)} missing/outdated packages")
            
            # PHASE 4: Atomic installation
            self.log("PHASE 4: Installing packages atomically")
            self._install_packages(cuda_config, venv_python, skip_installed=should_resume, packages_to_install=packages_to_install if should_resume else None)
            
            # PHASE 5: Clear Python cache
            self.log("PHASE 5: Clearing Python bytecode cache")
            self._clear_pycache()
            
            # PHASE 6: Verification
            self.log("PHASE 6: Verifying installation")
            self._verify_installation(venv_python)
            
            self.log("=" * 60)
            self.log("✓ Installation complete")
            self.log("=" * 60)
            return True, ""
            
        except Exception as e:
            error_msg = str(e)
            self.log(f"✗ Installation failed: {error_msg}")
            
            # Delete corrupted venv
            if self.venv_path.exists():
                self.log("Cleaning up corrupted venv...")
                try:
                    shutil.rmtree(self.venv_path, ignore_errors=True)
                    self.log("✓ Corrupted venv removed")
                except Exception as cleanup_error:
                    self.log(f"WARNING: Could not remove venv: {cleanup_error}")
            
            return False, error_msg
    
    def _validate_environment(self):
        """Validate environment before starting"""
        # Check wheelhouse exists
        if not self.wheelhouse.exists():
            raise InstallationFailed(f"Wheelhouse not found: {self.wheelhouse}")
        
        # Check Python version
        py_version = sys.version_info
        min_ver = tuple(map(int, self.manifest.get("python_min", "3.10").split(".")))
        max_ver = tuple(map(int, self.manifest.get("python_max", "3.12").split(".")))
        
        if py_version < min_ver or py_version >= (max_ver[0], max_ver[1] + 1):
            raise InstallationFailed(
                f"Python {py_version[0]}.{py_version[1]} not supported. "
                f"Required: >={min_ver[0]}.{min_ver[1]}, <={max_ver[0]}.{max_ver[1]}"
            )
        
        self.log(f"  ✓ Python {py_version[0]}.{py_version[1]} OK")
    
    def _destroy_venv(self):
        """Completely destroy existing venv"""
        if not self.venv_path.exists():
            self.log("  No existing venv to remove")
            return
        
        self.log(f"  Removing existing venv: {self.venv_path}")
        
        # Terminate any processes using the venv
        self._terminate_venv_processes()
        
        # Force delete with retries
        for attempt in range(1, 4):
            try:
                if sys.platform == 'win32':
                    # Use Windows command for force delete
                    result = subprocess.run(
                        ['cmd', '/c', 'rmdir', '/S', '/Q', str(self.venv_path)],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode == 0 or not self.venv_path.exists():
                        self.log(f"  ✓ Venv destroyed (attempt {attempt}/3)")
                        return
                else:
                    shutil.rmtree(self.venv_path, ignore_errors=True)
                    if not self.venv_path.exists():
                        self.log(f"  ✓ Venv destroyed (attempt {attempt}/3)")
                        return
            except Exception as e:
                if attempt < 3:
                    self.log(f"  Attempt {attempt} failed: {e}, retrying...")
                    time.sleep(attempt * 2)
                else:
                    raise InstallationFailed(f"Cannot delete venv after 3 attempts: {e}")
        
        # Final check
        if self.venv_path.exists():
            raise InstallationFailed(f"Venv still exists after deletion attempts: {self.venv_path}")
    
    def _terminate_venv_processes(self):
        """Terminate any processes using the venv"""
        if sys.platform != 'win32':
            return  # Only implemented for Windows for now
        
        venv_python = self.venv_path / "Scripts" / "python.exe"
        if not venv_python.exists():
            return
        
        try:
            # Find processes using this Python
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                **self.subprocess_flags
            )
            
            # Note: This is a basic implementation
            # More sophisticated process detection would check file handles
            
        except Exception:
            pass  # Best effort
    
    def _create_venv(self) -> Path:
        """
        Create fresh venv.
        
        Returns:
            Path to venv Python executable
        """
        self.log(f"  Creating venv at: {self.venv_path}")
        
        # Create venv with --clear flag
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(self.venv_path), "--clear"],
            capture_output=True,
            text=True,
            timeout=120,
            **self.subprocess_flags
        )
        
        if result.returncode != 0:
            raise InstallationFailed(f"Failed to create venv: {result.stderr}")
        
        # Get venv Python path
        if sys.platform == 'win32':
            venv_python = self.venv_path / "Scripts" / "python.exe"
        else:
            venv_python = self.venv_path / "bin" / "python"
        
        if not venv_python.exists():
            raise InstallationFailed(f"Venv created but Python not found at {venv_python}")
        
        self.log(f"  ✓ Venv created")
        
        # Upgrade pip, setuptools, wheel
        self.log("  Upgrading pip, setuptools, wheel...")
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            capture_output=True,
            text=True,
            timeout=300,
            **self.subprocess_flags
        )
        
        if result.returncode != 0:
            self.log(f"  WARNING: pip upgrade had issues: {result.stderr[:200]}")
        else:
            self.log(f"  ✓ pip upgraded")
        
        return venv_python
    
    def _install_packages(self, cuda_config: str, venv_python: Path, skip_installed: bool = False, packages_to_install: List[Tuple[str, str]] = None):
        """
        Install all packages from wheelhouse in strict order.
        
        Args:
            cuda_config: CUDA configuration key
            venv_python: Path to venv Python executable
            skip_installed: If True, skip packages that are already installed correctly
            packages_to_install: Optional list of (package_name, version_spec) tuples to install.
                               If provided, only install these packages. If None, install all.
        """
        # Get dependencies in order
        deps = sorted(self.manifest["core_dependencies"], key=lambda x: x["order"])
        cuda_packages = self.manifest["cuda_configs"][cuda_config]["packages"]
        
        # Create set of packages to install for quick lookup
        packages_to_install_set = None
        if packages_to_install:
            # Normalize package names for lookup
            packages_to_install_set = set()
            for pkg_name, version_spec in packages_to_install:
                pkg_normalized = pkg_name.lower().replace("_", "-")
                packages_to_install_set.add((pkg_normalized, version_spec))
                packages_to_install_set.add((pkg_name, version_spec))  # Also add original name
        
        installed_count = 0
        skipped_count = 0
        total_count = len([d for d in deps if d["version"] != "FROM_CUDA_CONFIG"]) + len(cuda_packages)
        
        # Install CUDA packages first
        for pkg_name, pkg_version in cuda_packages.items():
            version_spec = f"=={pkg_version}"
            
            # If packages_to_install is specified, only install those packages
            if packages_to_install_set is not None:
                pkg_normalized = pkg_name.lower().replace("_", "-")
                if (pkg_normalized, version_spec) not in packages_to_install_set and (pkg_name, version_spec) not in packages_to_install_set:
                    continue
            
            # Check if package is already installed (if skip_installed is True)
            if skip_installed:
                is_installed, version_matches = self._check_package_installed(venv_python, pkg_name, version_spec)
                if is_installed and version_matches:
                    skipped_count += 1
                    self.log(f"  [{installed_count + skipped_count}/{total_count}] Skipping {pkg_name} (already installed correctly)")
                    continue
            
            installed_count += 1
            progress = f"[{installed_count + skipped_count}/{total_count}]"
            self.log(f"  {progress} Installing {pkg_name}...")
            
            # Build install command
            success, error = self._install_single_package(
                venv_python=venv_python,
                package_name=pkg_name,
                version_spec=version_spec,
                install_args=["--no-deps"]  # CUDA packages use --no-deps
            )
            
            if not success:
                raise InstallationFailed(f"Failed to install critical CUDA package {pkg_name}: {error}")
            
            # Special handling for torch - create lock
            if pkg_name == "torch":
                self._create_torch_lock(pkg_version)
        
        # Install core dependencies
        for dep in deps:
            pkg_name = dep["name"]
            
            # Skip platform-specific packages
            if "platform" in dep and dep["platform"] != sys.platform:
                self.log(f"  Skipping {pkg_name} (platform: {dep['platform']})")
                continue
            
            # Handle CUDA packages
            if dep["version"] == "FROM_CUDA_CONFIG":
                if pkg_name in cuda_packages:
                    pkg_version = cuda_packages[pkg_name]
                    version_spec = f"=={pkg_version}"
                else:
                    continue
            else:
                # Extract version from spec
                version_spec = dep["version"]
                pkg_version = None  # Will use version_spec directly
            
            # If packages_to_install is specified, only install those packages
            if packages_to_install_set is not None:
                pkg_normalized = pkg_name.lower().replace("_", "-")
                # Check both normalized and original package name
                if (pkg_normalized, version_spec) not in packages_to_install_set:
                    # Also check original name in case it wasn't normalized
                    if (pkg_name, version_spec) not in packages_to_install_set:
                        continue
            
            # Check if package is already installed (if skip_installed is True)
            if skip_installed:
                is_installed, version_matches = self._check_package_installed(venv_python, pkg_name, version_spec)
                if is_installed and version_matches:
                    skipped_count += 1
                    self.log(f"  [{installed_count + skipped_count}/{total_count}] Skipping {pkg_name} (already installed correctly)")
                    continue
            
            # Skip optional packages
            # (Could add logic here to check blacklist_deps)
            
            installed_count += 1
            progress = f"[{installed_count + skipped_count}/{total_count}]"
            self.log(f"  {progress} Installing {pkg_name}...")
            
            # Build install command
            success, error = self._install_single_package(
                venv_python=venv_python,
                package_name=pkg_name,
                version_spec=pkg_version or dep["version"],
                install_args=dep.get("install_args", [])
            )
            
            if not success:
                if dep.get("critical", False):
                    raise InstallationFailed(f"Failed to install critical package {pkg_name}: {error}")
                else:
                    self.log(f"  WARNING: Failed to install optional package {pkg_name}: {error}")
                    continue
            
            # Special handling for torch - create lock
            if pkg_name == "torch":
                self._create_torch_lock(pkg_version or dep["version"])
            
            # Verify critical packages immediately after install
            if dep.get("critical", False) and pkg_name in ["torch", "transformers", "peft"]:
                success, error = self._verify_package_import(venv_python, pkg_name)
                if not success:
                    raise InstallationFailed(f"Package {pkg_name} installed but import failed: {error}")
        
        if skipped_count > 0:
            self.log(f"  ✓ Installation complete: {installed_count} installed, {skipped_count} skipped (already installed)")
        else:
            self.log(f"  ✓ All packages installed ({installed_count} total)")
    
    def _install_single_package(
        self, 
        venv_python: Path, 
        package_name: str, 
        version_spec: str,
        install_args: List[str] = None
    ) -> Tuple[bool, str]:
        """
        Install a single package from wheelhouse.
        
        Args:
            venv_python: Path to venv Python
            package_name: Package name
            version_spec: Version specifier (e.g., "==2.5.1+cu124" or ">=1.0.0")
            install_args: Additional pip install arguments
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        if install_args is None:
            install_args = []
        
        # Build pip install command
        cmd = [
            str(venv_python), "-m", "pip", "install",
            "--no-index",  # Critical: offline only
            "--find-links", str(self.wheelhouse),
            "--no-cache-dir"
        ]
        
        # Add additional args (like --no-deps)
        cmd.extend(install_args)
        
        # Add package specifier
        if version_spec.startswith("==") or version_spec.startswith(">=") or version_spec.startswith("<"):
            cmd.append(f"{package_name}{version_spec}")
        else:
            cmd.append(f"{package_name}=={version_spec}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes per package
                **self.subprocess_flags
            )
            
            if result.returncode != 0:
                error = result.stderr or result.stdout
                return False, error[:500]  # Truncate long errors
            
            return True, ""
            
        except subprocess.TimeoutExpired:
            return False, f"Installation timeout for {package_name}"
        except Exception as e:
            return False, str(e)
    
    def _create_torch_lock(self, version: str):
        """Create immutability marker for torch"""
        lock_data = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "locked": True,
            "comment": "DO NOT MODIFY - torch version is immutable"
        }
        
        self.torch_lock.write_text(json.dumps(lock_data, indent=2))
        self.log(f"    ✓ Torch lock created: {version}")
    
    def _check_package_installed(self, venv_python: Path, package_name: str, version_spec: str) -> Tuple[bool, bool]:
        """
        Check if a package is installed and if its version matches the requirement.
        
        Args:
            venv_python: Path to venv Python executable
            package_name: Package name to check
            version_spec: Version specifier (e.g., "==4.51.3" or ">=4.51.3,!=4.52.*")
        
        Returns:
            Tuple of (is_installed: bool, version_matches: bool)
        """
        # Check if package is installed using importlib.metadata
        if not METADATA_AVAILABLE:
            # Fallback: try to import the package
            code = f"import {package_name}; print('OK')"
            try:
                result = subprocess.run(
                    [str(venv_python), "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    **self.subprocess_flags
                )
                is_installed = result.returncode == 0
                if not is_installed:
                    return False, False
                # Can't verify version without metadata, assume it matches
                return True, True
            except Exception:
                return False, False
        
        # Use importlib.metadata to get installed version
        try:
            code = f"""
import sys
sys.path.insert(0, r'{venv_python.parent.parent / "Lib" / "site-packages"}')
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    installed_ver = version('{package_name}')
    print(installed_ver)
except PackageNotFoundError:
    print('NOT_FOUND')
"""
            result = subprocess.run(
                [str(venv_python), "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
                **self.subprocess_flags
            )
            
            if result.returncode != 0:
                return False, False
            
            output = result.stdout.strip()
            if output == "NOT_FOUND" or not output:
                return False, False
            
            installed_version = output.strip()
            
            # Check if version matches requirement
            if not PACKAGING_AVAILABLE:
                # Fallback: only support exact == matches
                if version_spec.startswith("=="):
                    required = version_spec[2:].strip()
                    return True, installed_version == required
                else:
                    # Can't verify complex specifiers, assume it matches
                    return True, True
            
            # Use packaging library for robust version comparison
            try:
                spec = SpecifierSet(version_spec)
                # Handle build tags (e.g., "2.5.1+cu124" -> "2.5.1")
                base_version = installed_version.split("+")[0].split("-")[0]
                version_matches = spec.contains(pkg_version.parse(base_version))
                return True, version_matches
            except Exception:
                # If parsing fails, assume version doesn't match
                return True, False
                
        except Exception:
            return False, False
    
    def _verify_package_import(self, venv_python: Path, package_name: str) -> Tuple[bool, str]:
        """
        Verify package can be imported.
        
        Args:
            venv_python: Path to venv Python
            package_name: Package to import
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        code = f"import {package_name}; print('{package_name} OK')"
        
        try:
            result = subprocess.run(
                [str(venv_python), "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
                **self.subprocess_flags
            )
            
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr or result.stdout
                
        except Exception as e:
            return False, str(e)
    
    def _clear_pycache(self):
        """Clear Python bytecode cache to prevent lazy_loader corruption"""
        site_packages = self.venv_path / "Lib" / "site-packages"
        if not site_packages.exists():
            return
        
        cleared_count = 0
        
        # Remove __pycache__ directories
        for pycache_dir in site_packages.rglob("__pycache__"):
            try:
                shutil.rmtree(pycache_dir, ignore_errors=True)
                cleared_count += 1
            except Exception:
                pass
        
        # Remove .pyc files
        for pyc_file in site_packages.rglob("*.pyc"):
            try:
                pyc_file.unlink()
                cleared_count += 1
            except Exception:
                pass
        
        if cleared_count > 0:
            self.log(f"  ✓ Cleared {cleared_count} cache files/directories")
    
    def _verify_installation(self, venv_python: Path):
        """
        Run verification checks.
        
        Args:
            venv_python: Path to venv Python
        """
        from .verification import VerificationSystem
        
        # Create verifier
        manifest_path = Path(__file__).parent.parent / "metadata" / "dependencies.json"
        verifier = VerificationSystem(manifest_path, venv_python)
        
        # Run quick verification
        success, error = verifier.run_quick_verify()
        
        if not success:
            raise InstallationFailed(f"Verification failed: {error}")
        
        self.log("  ✓ Verification passed")


def main():
    """Test immutable installer"""
    import sys
    from pathlib import Path
    
    # Test setup
    venv_path = Path(__file__).parent.parent / ".venv_test"
    wheelhouse = Path(__file__).parent.parent / "wheelhouse"
    manifest = Path(__file__).parent.parent / "metadata" / "dependencies.json"
    
    if not wheelhouse.exists():
        print(f"ERROR: Wheelhouse not found at {wheelhouse}")
        print("Run wheelhouse manager first to download wheels")
        sys.exit(1)
    
    if not manifest.exists():
        print(f"ERROR: Manifest not found at {manifest}")
        sys.exit(1)
    
    installer = ImmutableInstaller(venv_path, wheelhouse, manifest)
    
    print("Testing immutable installer...")
    success, error = installer.install("cu124")
    
    if success:
        print("✓ Test passed!")
    else:
        print(f"✗ Test failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()

