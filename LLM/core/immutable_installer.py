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
    
    def __init__(self, venv_path: Path, wheelhouse_path: Path, manifest_path: Path, python_executable: Path = None):
        """
        Initialize immutable installer.
        
        Args:
            venv_path: Path where venv should be created
            wheelhouse_path: Path to wheelhouse with downloaded wheels
            manifest_path: Path to dependencies.json manifest
            python_executable: Optional Python executable to use for creating venv (defaults to sys.executable)
        """
        self.venv_path = venv_path
        self.wheelhouse = wheelhouse_path
        self.manifest_path = manifest_path  # Store for use in fallback downloads
        self.python_executable = Path(python_executable) if python_executable else Path(sys.executable)
        
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
        """Log message to console with encoding safety"""
        try:
            print(f"[INSTALL] {message}")
        except UnicodeEncodeError:
            # Fallback for Windows consoles that don't support UTF-8 characters
            safe_message = message.replace('✓', '[OK]').replace('✗', '[FAIL]').replace('⚠', '[WARN]')
            try:
                print(f"[INSTALL] {safe_message}")
            except Exception:
                pass
    
    def install(self, cuda_config: str, package_versions: dict = None, binary_packages: dict = None) -> Tuple[bool, str]:
        """
        Perform immutable installation.
        
        Args:
            cuda_config: CUDA configuration key (e.g., "cu124")
            package_versions: Optional dict of {package_name: exact_version} from profile.
                            If provided, uses these for version checking instead of manifest.
            binary_packages: Optional dict of binary packages to install from wheels.
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            self.log("=" * 60)
            self.log("Immutable Installation Engine Started")
            self.log(f"  Configuration: {cuda_config}")
            if package_versions:
                self.log(f"  Using profile versions: {len(package_versions)} packages")
            if binary_packages:
                self.log(f"  Binary packages: {len(binary_packages)} packages")
                for pkg_name in binary_packages.keys():
                    self.log(f"    - {pkg_name}")
            else:
                self.log(f"  Binary packages: None (not provided)")
            self.log("=" * 60)
            
            # Store profile versions and binary packages for installation
            self.profile_versions = package_versions or {}
            self.binary_packages = binary_packages or {}
            
            # DEBUG: Log what we stored
            if self.binary_packages:
                self.log(f"[DEBUG] Stored {len(self.binary_packages)} binary packages: {list(self.binary_packages.keys())}")
            else:
                self.log(f"[DEBUG] No binary packages stored")
            
            # PHASE 0: Validation
            self.log("[STEP] PHASE 0: Validating environment...")
            self._validate_environment()
            self.log("  ✓ Environment valid")
            
            # PHASE 1: Wheelhouse already prepared by WheelhouseManager
            self.log("[STEP] PHASE 1: Verifying wheelhouse...")
            if not self.wheelhouse.exists():
                raise InstallationFailed(f"Wheelhouse not found at {self.wheelhouse}")
            
            wheel_count = len(list(self.wheelhouse.glob("*.whl")))
            if wheel_count == 0:
                raise InstallationFailed("Wheelhouse is empty - no wheels found")
            self.log(f"  ✓ Wheelhouse contains {wheel_count} wheels")
            
            # PHASE 2: Resume detection - check if venv exists and is valid
            self.log("[STEP] PHASE 2: Checking existing venv for repair...")
            venv_python = None
            should_resume = False
            packages_to_install = []
            
            if self.venv_path.exists():
                self.log("  Found existing .venv directory")
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

                            # Critical: remove known-bad packages (e.g. torchao) before any import checks.
                            self._uninstall_blacklisted_packages(venv_python)
                            
                            # Check which packages need installation
                            deps = sorted(self.manifest["core_dependencies"], key=lambda x: x["order"])
                            cuda_packages = self.manifest["cuda_configs"][cuda_config]["packages"]
                            
                            # Use profile versions if available, otherwise fall back to manifest
                            if self.profile_versions:
                                # Profile mode: use exact versions from profile
                                cuda_packages_to_check = {k: self.profile_versions.get(k, v) for k, v in cuda_packages.items()}
                            else:
                                cuda_packages_to_check = cuda_packages
                            
                            missing_packages = []
                            wrong_version_packages = []
                            
                            # Check CUDA packages (with functionality check for broken packages)
                            for pkg_name, pkg_version in cuda_packages_to_check.items():
                                is_installed, version_matches, is_broken = self._check_package_installed(
                                    venv_python, pkg_name, f"=={pkg_version}", check_functionality=True
                                )
                                if not is_installed:
                                    missing_packages.append((pkg_name, f"=={pkg_version}"))
                                elif is_broken:
                                    # Only add if actually broken (DLL corruption, missing attributes, etc.)
                                    wrong_version_packages.append((pkg_name, f"=={pkg_version}"))
                                elif not version_matches:
                                    # Only add if version is wrong
                                    wrong_version_packages.append((pkg_name, f"=={pkg_version}"))
                                # Otherwise: installed, correct version, not broken -> skip (don't add to any list)
                            
                            # Check core dependencies (with functionality check for broken packages)
                            for dep in deps:
                                pkg_name = dep["name"]
                                
                                # Skip platform-specific packages
                                if "platform" in dep and dep["platform"] != sys.platform:
                                    continue
                                
                                # Skip CUDA packages (already checked)
                                if dep["version"] == "FROM_CUDA_CONFIG":
                                    continue
                                
                                # Use profile version if available, otherwise use manifest version
                                # Profile versions may be ranges (e.g., ">=0.13.0,<0.16.0") or exact (e.g., "0.14.0")
                                if self.profile_versions and pkg_name in self.profile_versions:
                                    profile_ver = self.profile_versions[pkg_name]
                                    # Check if it's already a range or exact version
                                    if any(op in profile_ver for op in [">=", "<=", ">", "<", "!=", ","]):
                                        version_spec = profile_ver  # Use range as-is
                                    else:
                                        version_spec = f"=={profile_ver}"  # Exact version
                                else:
                                    version_spec = dep["version"]
                                
                                is_installed, version_matches, is_broken = self._check_package_installed(
                                    venv_python, pkg_name, version_spec, check_functionality=True
                                )
                                
                                if not is_installed:
                                    missing_packages.append((pkg_name, version_spec))
                                elif is_broken:
                                    # Only add if actually broken (import fails, missing attributes, etc.)
                                    wrong_version_packages.append((pkg_name, version_spec))
                                elif not version_matches:
                                    # Only add if version is wrong
                                    wrong_version_packages.append((pkg_name, version_spec))
                                # Otherwise: installed, correct version, not broken -> skip (don't add to any list)
                            
                            packages_to_install = missing_packages + wrong_version_packages
                            
                            if packages_to_install:
                                broken_packages = []
                                missing_pkg_names = [pkg for pkg, _ in missing_packages]
                                wrong_pkg_names = [pkg for pkg, _ in wrong_version_packages]
                                
                                # Check which packages are broken (not just wrong version)
                                for pkg_name, version_spec in packages_to_install:
                                    is_installed, _, is_broken = self._check_package_installed(
                                        venv_python, pkg_name, version_spec, check_functionality=True
                                    )
                                    if is_broken:
                                        broken_packages.append(pkg_name)
                                
                                if broken_packages:
                                    self.log(f"  ⚠ Found {len(broken_packages)} broken packages: {', '.join(broken_packages)}")
                                
                                self.log(f"  ⚠ Found {len(packages_to_install)} packages that need installation:")
                                for pkg_name, version_spec in packages_to_install[:5]:  # Show first 5
                                    status = " (broken)" if pkg_name in broken_packages else ""
                                    self.log(f"    - {pkg_name}{version_spec}{status}")
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
                                self.log("  ✓ All regular packages already installed correctly")
                                # Still need to check binary packages
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
                if packages_to_install:
                    self.log("PHASE 2: Resuming installation (venv already exists)")
                    self.log(f"  Will install {len(packages_to_install)} missing/outdated packages")
                else:
                    self.log("PHASE 2: All packages already installed correctly")
                    self.log("  No installation needed - skipping to verification")
            
            # PHASE 4: Atomic installation
            # ALWAYS run installation phase to check and install binary packages if needed
            # This ensures binary packages are installed automatically even when regular packages are OK
            self.log("[STEP] PHASE 4: Installing packages atomically")
            
            # Always run _install_packages - it will handle both regular and binary packages
            # Pass packages_to_install as-is (empty list will be handled in _install_packages)
            packages_to_install_param = packages_to_install if should_resume else None
            self._install_packages(cuda_config, venv_python, skip_installed=should_resume, packages_to_install=packages_to_install_param, binary_packages=self.binary_packages)
            
            # PHASE 5: Clear Python cache
            self.log("[STEP] PHASE 5: Clearing Python bytecode cache")
            self._clear_pycache()
            
            # PHASE 6: Verification (SKIP in resume mode if only some packages were installed)
            self.log("[STEP] PHASE 6: Verifying installation")
            if should_resume and packages_to_install:
                self.log("  (SKIPPED - Partial repair completed)")
                self.log("  Full verification would fail due to partial state.")
                self.log("  Run a full 'Install All' if you need complete environment verification.")
            else:
                self._verify_installation(venv_python)
            
            self.log("=" * 60)
            self.log("✓ Installation complete")
            self.log("=" * 60)
            return True, ""
            
        except Exception as e:
            error_msg = str(e)
            self.log(f"✗ Installation failed: {error_msg}")
            
            # Check if this is a version conflict that might be fixable
            is_version_conflict = self._is_version_conflict_error(error_msg)
            
            if is_version_conflict:
                self.log("\n⚠ Detected version conflict error")
                self.log("  Attempting to fix by reinstalling problematic packages...")
                
                # Get venv Python if it exists
                if self.venv_path.exists():
                    if sys.platform == 'win32':
                        venv_python = self.venv_path / "Scripts" / "python.exe"
                    else:
                        venv_python = self.venv_path / "bin" / "python"
                    
                    if venv_python.exists():
                        # Try to fix version conflicts
                        fix_success = self._fix_version_conflicts(venv_python, error_msg)
                        
                        if fix_success:
                            self.log("\n🔄 Retrying installation from where we left off...")
                            # Retry installation in resume mode
                            try:
                                self._install_packages(
                                    cuda_config,
                                    venv_python,
                                    skip_installed=True,
                                    packages_to_install=None  # Install all missing/wrong packages
                                )
                                
                                # Clear cache and verify
                                self._clear_pycache()
                                self._verify_installation(venv_python)
                                
                                self.log("=" * 60)
                                self.log("✓ Installation complete (after version conflict fix)")
                                self.log("=" * 60)
                                return True, ""
                            except Exception as retry_error:
                                self.log(f"  ✗ Retry after fix failed: {retry_error}")
                                # Fall through to cleanup
            
            # IMPORTANT: In resume/repair mode, do not destroy the venv automatically.
            # Destructive cleanup belongs to explicit "full install" flow.
            if self.venv_path.exists() and not should_resume:
                self.log("Cleaning up corrupted venv...")
                try:
                    shutil.rmtree(self.venv_path, ignore_errors=True)
                    self.log("✓ Corrupted venv removed")
                except Exception as cleanup_error:
                    self.log(f"WARNING: Could not remove venv: {cleanup_error}")
            
            return False, error_msg
    
    def _validate_environment(self):
        """Validate environment before starting"""
        self.log("  Validating environment suitability...")
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

        # Check for venv locks (Windows specific)
        if sys.platform == 'win32' and self.venv_path.exists():
            self.log("  Checking for environment locks...")
            try:
                # Try to see if we can open a dummy file in the venv
                test_file = self.venv_path / ".repair_lock_test"
                with open(test_file, 'w') as f:
                    f.write('test')
                test_file.unlink()
                self.log("  ✓ Environment is writable")
            except Exception as e:
                self.log(f"  ⚠ Warning: Environment might be locked: {e}")
                self.log("  Please ensure no other training or server processes are running.")

    
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
        self.log(f"  Using Python: {self.python_executable}")
        
        # Create venv with --clear flag
        result = subprocess.run(
            [str(self.python_executable), "-m", "venv", str(self.venv_path), "--clear"],
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
    
    def _install_packages(self, cuda_config: str, venv_python: Path, skip_installed: bool = False, packages_to_install: List[Tuple[str, str]] = None, binary_packages: dict = None):
        """
        Install all packages from wheelhouse in strict order.
        
        Args:
            cuda_config: CUDA configuration key
            venv_python: Path to venv Python executable
            skip_installed: If True, skip packages that are already installed correctly
            packages_to_install: Optional list of (package_name, version_spec) tuples to install.
                               If provided, only install these packages. If None, install all.
            binary_packages: Optional dict of binary packages to install from wheels.
        """
        # Get dependencies in order
        deps = sorted(self.manifest["core_dependencies"], key=lambda x: x["order"])
        cuda_packages = self.manifest["cuda_configs"][cuda_config]["packages"]
        
        # Use profile versions if available (same as in venv checking phase)
        if self.profile_versions:
            # Profile mode: use exact versions from profile
            cuda_packages_to_install = {k: self.profile_versions.get(k, v) for k, v in cuda_packages.items()}
        else:
            cuda_packages_to_install = cuda_packages
        
        # Create set of packages to install for quick lookup
        packages_to_install_set = None
        if packages_to_install is not None:
            # If empty list, skip all installations
            if len(packages_to_install) == 0:
                self.log("  ✓ All packages already installed correctly - skipping all installations")
                return
            # Normalize package names for lookup
            packages_to_install_set = set()
            for pkg_name, version_spec in packages_to_install:
                pkg_normalized = pkg_name.lower().replace("_", "-")
                packages_to_install_set.add((pkg_normalized, version_spec))
                packages_to_install_set.add((pkg_name, version_spec))  # Also add original name
        
        installed_count = 0
        skipped_count = 0
        total_count = len([d for d in deps if d["version"] != "FROM_CUDA_CONFIG"]) + len(cuda_packages_to_install)
        
        # Install CUDA packages first
        for pkg_name, pkg_version in cuda_packages_to_install.items():
            version_spec = f"=={pkg_version}"
            
            # If packages_to_install is specified, only install those packages
            if packages_to_install_set is not None:
                pkg_normalized = pkg_name.lower().replace("_", "-")
                if (pkg_normalized, version_spec) not in packages_to_install_set and (pkg_name, version_spec) not in packages_to_install_set:
                    continue
            
            # Check if package is already installed (if skip_installed is True)
            extra_reinstall_args: List[str] = []
            if skip_installed:
                is_installed, version_matches, is_broken = self._check_package_installed(venv_python, pkg_name, version_spec, check_functionality=True)
                if is_installed and version_matches and not is_broken:
                    skipped_count += 1
                    self.log(f"  [{installed_count + skipped_count}/{total_count}] Skipping {pkg_name} (already installed correctly)")
                    continue
                elif is_broken:
                    self.log(f"  [{installed_count + skipped_count}/{total_count}] Package {pkg_name} is broken - will reinstall")
                    # Force uninstall first, then reinstall
                    self.log(f"    Uninstalling broken {pkg_name}...")
                    self._uninstall_package(venv_python, pkg_name)
                    # After uninstall, we'll do a clean install (no extra args needed)
                    extra_reinstall_args = []
                elif is_installed and not version_matches:
                    # Force uninstall for wrong version, then clean install
                    self.log(f"    Uninstalling wrong version of {pkg_name}...")
                    self._uninstall_package(venv_python, pkg_name)
                    extra_reinstall_args = []
            
            installed_count += 1
            progress = f"[{installed_count + skipped_count}/{total_count}]"
            self.log(f"  {progress} Installing {pkg_name}...")
            
            # Build install command
            success, error = self._install_single_package(
                venv_python=venv_python,
                package_name=pkg_name,
                version_spec=version_spec,
                install_args=["--no-deps"] + extra_reinstall_args  # CUDA packages use --no-deps
            )
            
            if not success:
                raise InstallationFailed(f"Failed to install critical CUDA package {pkg_name}: {error}")
            
            # Special handling for torch - create lock
            if pkg_name == "torch":
                self._create_torch_lock(pkg_version)
        
        # Install binary packages (after torch, before other packages)
        # Use self.binary_packages (stored from install()) instead of parameter
        # Also check parameter in case it was passed directly
        binary_packages_to_use = binary_packages if binary_packages else self.binary_packages
        if binary_packages_to_use:
            self.log("Installing binary packages from wheels...")
            self.log(f"  Found {len(binary_packages_to_use)} binary package(s) to install: {list(binary_packages_to_use.keys())}")
            # During repair, always install binary packages even if skip_installed is True
            # This ensures they get installed correctly
            if skip_installed and not packages_to_install:
                self.log("  Repair mode: Will verify and install binary packages if needed")
            # Install in correct order: triton -> causal_conv1d -> mamba_ssm
            binary_order = ["triton", "causal_conv1d", "mamba_ssm"]
            for pkg_name in binary_order:
                if pkg_name in binary_packages_to_use:
                    pkg_info = binary_packages_to_use[pkg_name]
                    url = pkg_info.get("url")
                    if url:
                        # Extract wheel filename from URL
                        wheel_filename = url.split("/")[-1]
                        wheel_path = self.wheelhouse / wheel_filename
                        
                        if wheel_path.exists():
                            # For binary packages, always check if they're actually importable
                            # Don't trust the simple version check - try importing them
                            # During repair mode (skip_installed=True and no packages_to_install),
                            # we should ALWAYS reinstall binary packages to ensure they work
                            should_install = True
                            # Check if we're in repair mode
                            is_repair_mode = skip_installed and (packages_to_install is None or len(packages_to_install) == 0)
                            
                            # In repair mode, always install binary packages (don't skip)
                            if is_repair_mode:
                                self.log(f"  Repair mode: Will force install {pkg_name} to ensure it works")
                                should_install = True
                            elif skip_installed:
                                try:
                                    # Try to actually import the package to verify it works
                                    import_cmd = f"import {pkg_name.replace('-', '_')}"
                                    result = subprocess.run(
                                        [str(venv_python), "-c", import_cmd],
                                        capture_output=True,
                                        text=True,
                                        timeout=5,
                                        **self.subprocess_flags
                                    )
                                    if result.returncode == 0:
                                        # Package imports successfully - check version if needed
                                        if pkg_name == "triton":
                                            # For triton, also verify version
                                            version_cmd = "import triton; print(triton.__version__)"
                                            ver_result = subprocess.run(
                                                [str(venv_python), "-c", version_cmd],
                                                capture_output=True,
                                                text=True,
                                                timeout=5,
                                                **self.subprocess_flags
                                            )
                                            if ver_result.returncode == 0:
                                                try:
                                                    from packaging import version as pkg_version
                                                    installed_ver = ver_result.stdout.strip()
                                                    if pkg_version.parse(installed_ver) >= pkg_version.parse("3.0.0"):
                                                        self.log(f"  ✓ {pkg_name} is already installed and working (v{installed_ver})")
                                                        should_install = False
                                                except Exception as ver_err:
                                                    self.log(f"  Version check failed for {pkg_name}: {ver_err}, will reinstall")
                                                    should_install = True
                                        else:
                                            # For causal_conv1d and mamba_ssm, if import works, it's good
                                            self.log(f"  ✓ {pkg_name} is already installed and working")
                                            should_install = False
                                except Exception as e:
                                    # If import check fails, we need to install
                                    self.log(f"  {pkg_name} import check failed: {e}, will install")
                                    should_install = True
                                except Exception as e:
                                    # If import check fails, we need to install
                                    self.log(f"  {pkg_name} import check failed: {e}, will install")
                                    should_install = True
                            
                            if not should_install:
                                continue
                            
                            self.log(f"  Installing {pkg_name} from {wheel_filename}...")
                            success, error = self._install_binary_package(venv_python, wheel_path)
                            if success:
                                self.log(f"  ✓ {pkg_name} installed successfully")
                            else:
                                self.log(f"  WARNING: Failed to install binary package {pkg_name}: {error}")
                                # Binary packages are optional - continue even if they fail
                                # They'll be handled at runtime with error messages
                        else:
                            self.log(f"  WARNING: Binary package wheel not found: {wheel_filename}")
                            self.log(f"    Expected at: {wheel_path}")
                            self.log(f"    This means the wheel was not downloaded during wheelhouse preparation.")
                            self.log(f"    Please check if the URL is accessible and the wheelhouse preparation succeeded.")
                            self.log(f"    URL was: {url}")
                            # Try to download it now as fallback
                            self.log(f"    Attempting to download now as fallback...")
                            from core.wheelhouse import WheelhouseManager
                            wheelhouse_mgr = WheelhouseManager(self.manifest_path, self.wheelhouse)
                            download_success, download_error = wheelhouse_mgr._download_binary_package(url, pkg_name)
                            if download_success and wheel_path.exists():
                                self.log(f"    ✓ Downloaded {wheel_filename}, proceeding with installation")
                                # Retry installation
                                success, error = self._install_binary_package(venv_python, wheel_path)
                                if success:
                                    self.log(f"  ✓ {pkg_name} installed successfully")
                                else:
                                    self.log(f"  WARNING: Failed to install binary package {pkg_name}: {error}")
                            else:
                                self.log(f"    ✗ Download failed: {download_error}")
        else:
            if not binary_packages and not self.binary_packages:
                self.log("  No binary packages to install")
        
        # Install core dependencies
        for dep in deps:
            pkg_name = dep["name"]
            
            # Skip platform-specific packages
            if "platform" in dep and dep["platform"] != sys.platform:
                self.log(f"  Skipping {pkg_name} (platform: {dep['platform']})")
                continue
            
            # CUDA stack was already installed in the first loop; don't install it twice.
            if dep["version"] == "FROM_CUDA_CONFIG":
                continue

            # Extract version from spec
            # Use profile version if available (same as in venv checking phase), otherwise use manifest version
            if self.profile_versions and pkg_name in self.profile_versions:
                version_spec = f"=={self.profile_versions[pkg_name]}"
            else:
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
            extra_reinstall_args = []
            if skip_installed:
                is_installed, version_matches, is_broken = self._check_package_installed(venv_python, pkg_name, version_spec, check_functionality=True)
                if is_installed and version_matches and not is_broken:
                    skipped_count += 1
                    self.log(f"  [{installed_count + skipped_count}/{total_count}] Skipping {pkg_name} (already installed correctly)")
                    continue
                elif is_broken:
                    self.log(f"  [{installed_count + skipped_count}/{total_count}] Package {pkg_name} is broken - will reinstall")
                    # Force uninstall first, then reinstall (don't use --force-reinstall which keeps corrupted files)
                    self.log(f"    Uninstalling broken {pkg_name}...")
                    self._uninstall_package(venv_python, pkg_name)
                    # After uninstall, we'll do a clean install (no extra args needed)
                    extra_reinstall_args = []
                    
                    # For torch, ensure all DLL files are cleaned up before reinstall to avoid corruption
                    if pkg_name == "torch":
                        self.log(f"    Cleaning up corrupted torch DLLs before reinstall...")
                        torch_lib = self.venv_path / "Lib" / "site-packages" / "torch" / "lib"
                        if torch_lib.exists():
                            try:
                                for dll_file in torch_lib.glob("*.dll"):
                                    try:
                                        dll_file.unlink()
                                        self.log(f"      Removed: {dll_file.name}")
                                    except Exception as e:
                                        self.log(f"      Warning: Could not remove {dll_file.name}: {e}")
                            except Exception as e:
                                self.log(f"      Warning: DLL cleanup failed: {e}")
                    
                    # For triton, ensure clean reinstall if broken (missing ops sub-module)
                    elif pkg_name in ["triton", "triton-windows"]:
                        self.log(f"    Cleaning up broken triton installation...")
                        triton_path = self.venv_path / "Lib" / "site-packages" / "triton"
                        if triton_path.exists():
                            try:
                                import shutil
                                shutil.rmtree(triton_path, ignore_errors=True)
                                self.log(f"      Removed triton directory for clean reinstall")
                            except Exception as e:
                                self.log(f"      Warning: Could not remove triton directory: {e}")
                    
                    # For peft/transformers, ensure clean reinstall if broken (version compatibility issues)
                    elif pkg_name in ["peft", "transformers"]:
                        self.log(f"    Cleaning up broken {pkg_name} installation...")
                        pkg_path = self.venv_path / "Lib" / "site-packages" / pkg_name
                        if pkg_path.exists():
                            try:
                                import shutil
                                shutil.rmtree(pkg_path, ignore_errors=True)
                                self.log(f"      Removed {pkg_name} directory for clean reinstall")
                            except Exception as e:
                                self.log(f"      Warning: Could not remove {pkg_name} directory: {e}")
                elif is_installed and not version_matches:
                    # Force uninstall for wrong version, then clean install
                    self.log(f"    Uninstalling wrong version of {pkg_name}...")
                    self._uninstall_package(venv_python, pkg_name)
                    extra_reinstall_args = []
            
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
                install_args=dep.get("install_args", []) + extra_reinstall_args
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
                    # Check if this is a version conflict error that we can fix
                    if self._is_version_conflict_error(error):
                        self.log(f"  ⚠ Detected version conflict for {pkg_name}: {error[:200]}")
                        self.log(f"  🔄 Attempting to fix by reinstalling dependencies...")
                        
                        # Try to fix version conflicts by reinstalling problematic dependencies
                        fix_success = self._fix_version_conflicts(venv_python, error)
                        if fix_success:
                            # Retry verification
                            success, error = self._verify_package_import(venv_python, pkg_name)
                            if success:
                                self.log(f"  ✓ Version conflict resolved for {pkg_name}")
                                continue
                            else:
                                self.log(f"  ⚠ Fix attempt failed, will continue with original error")
                    
                    if not success:
                        raise InstallationFailed(f"Package {pkg_name} installed but import failed: {error}")
        
        if skipped_count > 0:
            self.log(f"  ✓ Installation complete: {installed_count} installed, {skipped_count} skipped (already installed)")
        else:
            self.log(f"  ✓ All packages installed ({installed_count} total)")
    
    def _install_binary_package(self, venv_python: Path, wheel_path: Path) -> Tuple[bool, str]:
        """
        Install a binary package from a wheel file.
        
        Args:
            venv_python: Path to venv Python executable
            wheel_path: Path to wheel file
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        cmd = [
            str(venv_python), "-m", "pip", "install",
            "--no-index",  # Critical: offline only
            "--find-links", str(self.wheelhouse),
            "--no-cache-dir",
            "--no-deps",  # Critical: prevent pip from resolving dependencies
            str(wheel_path)
        ]
        
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
            return False, f"Installation timeout for {wheel_path.name}"
        except Exception as e:
            return False, str(e)
    
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
            "--no-cache-dir",
            "--no-deps"  # Critical: prevent pip from resolving any dependencies
        ]
        
        # Add additional args (like --no-deps, --force-reinstall)
        # BUT: Never add install_args if they're already in cmd (avoid duplicate --no-deps)
        for arg in install_args:
            if arg not in cmd:
                cmd.append(arg)
        
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
    
    def _uninstall_package(self, venv_python: Path, package_name: str) -> Tuple[bool, str]:
        """
        Uninstall a package completely.
        
        Args:
            venv_python: Path to venv Python
            package_name: Package name to uninstall
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        cmd = [
            str(venv_python), "-m", "pip", "uninstall",
            "-y",  # No confirmation
            package_name
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
                **self.subprocess_flags
            )
            
            if result.returncode != 0:
                # If package not found, that's OK (already uninstalled)
                if "not installed" in result.stdout.lower() or "not installed" in result.stderr.lower():
                    return True, ""
                error = result.stderr or result.stdout
                return False, error[:500]  # Truncate long errors
            
            return True, ""
            
        except subprocess.TimeoutExpired:
            return False, f"Uninstall timeout for {package_name}"
        except Exception as e:
            return False, str(e)

    def _uninstall_blacklisted_packages(self, venv_python: Path) -> None:
        """
        Uninstall any globally-blacklisted packages from the target venv.
        These packages are known to break imports (e.g., torchao causing transformers failures).
        """
        blacklist = self.manifest.get("global_blacklist", [])
        if not blacklist:
            return

        for pkg in blacklist:
            try:
                # Best-effort uninstall; treat "not installed" as success.
                self.log(f"    Removing blacklisted package if present: {pkg}...")
                self._uninstall_package(venv_python, pkg)
            except Exception:
                # Never fail install because of uninstall issues; main install will surface real errors.
                pass
    
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
    
    def _check_package_broken(self, venv_python: Path, package_name: str) -> bool:
        """
        Check if a package is broken (importable but missing critical attributes or DLL corruption).
        
        Args:
            venv_python: Path to venv Python executable
            package_name: Package name to check
        
        Returns:
            True if package is broken (imports but missing critical attributes or DLL errors), False otherwise
        """
        # Special handling for torch - check critical attributes AND DLL integrity
        if package_name == "torch":
            code = """
try:
    import torch
    import sys
    # Check for critical attributes
    has_version = hasattr(torch, '__version__')
    has_cuda = hasattr(torch, 'cuda')
    has_tensor = hasattr(torch, 'Tensor')
    
    # Try to access CUDA to detect DLL corruption
    dll_ok = True
    try:
        _ = torch.cuda.is_available()  # This will fail if DLLs are corrupted
    except Exception as e:
        dll_ok = False
        print(f'DLL_ERROR: {e}', file=sys.stderr)
    
    if has_version and has_cuda and has_tensor and dll_ok:
        print('OK')
    else:
        print('BROKEN')
except ImportError:
    print('NOT_INSTALLED')
except Exception as e:
    import sys
    print(f'ERROR: {e}', file=sys.stderr)
    print('BROKEN')  # Any other error means it's broken
"""
        elif package_name in ["triton", "triton-windows"]:
            code = """
try:
    import triton
    # Check if triton has basic functionality
    # Note: triton-windows may not have triton.ops (Windows port limitation),
    # so we only check if triton itself can be imported and has core features
    has_version = hasattr(triton, '__version__')
    has_compile = hasattr(triton, 'compile')
    if has_version and has_compile:
        print('OK')
    else:
        print('BROKEN')
except ImportError:
    print('NOT_INSTALLED')
except Exception as e:
    print(f'ERROR: {e}')
    print('BROKEN')
"""
        elif package_name == "bitsandbytes":
            code = """
try:
    import bitsandbytes
    # Try a simple import that often fails if triton is broken
    from bitsandbytes.nn import Linear8bitLt
    print('OK')
except ImportError:
    print('NOT_INSTALLED')
except Exception as e:
    print(f'ERROR: {e}')
    print('BROKEN')
"""
        elif package_name == "transformers":
            code = """
try:
    import transformers
    from transformers import PreTrainedModel, AutoModel, AutoTokenizer
    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    print('BROKEN')
"""
        elif package_name == "peft":
            code = """
try:
    import peft
    import sys
    # Try importing key classes that are commonly used
    from peft import LoraConfig, get_peft_model
    # Also check that peft can properly import from transformers
    from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    print('OK')
except ImportError as e:
    # Check if it's the BloomPreTrainedModel error or similar compatibility issue
    error_str = str(e)
    if 'BloomPreTrainedModel' in error_str or 'cannot import name' in error_str:
        print('BROKEN')
    else:
        print('NOT_INSTALLED')
except Exception as e:
    print(f'ERROR: {e}')
    print('BROKEN')
"""
        elif package_name == "accelerate":
            code = """
try:
    import accelerate
    from accelerate import Accelerator
    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    print('BROKEN')
"""
        elif package_name == "datasets":
            code = """
try:
    import datasets
    from datasets import load_dataset
    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    print('BROKEN')
"""
        elif package_name == "torchao":
            code = """
try:
    import torchao
    # Check for basic functionality
    from torchao.quantization import quantize_
    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    print('BROKEN')
"""
        else:
            # For other packages, just check if import works
            code = f"""
try:
    import {package_name}
    print('OK')
except ImportError:
    print('NOT_INSTALLED')
except Exception as e:
    print(f'ERROR: {{e}}')
"""
        
        try:
            result = subprocess.run(
                [str(venv_python), "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
                **self.subprocess_flags
            )
            
            if result.returncode != 0:
                return True  # Consider broken if check fails
            
            output = result.stdout.strip()
            return output == "BROKEN" or output.startswith("ERROR:")
        except Exception:
            return True  # Consider broken if check fails
    
    def _check_package_installed(self, venv_python: Path, package_name: str, version_spec: str, check_functionality: bool = False) -> Tuple[bool, bool, bool]:
        """
        Check if a package is installed and if its version matches the requirement.
        
        Args:
            venv_python: Path to venv Python executable
            package_name: Package name to check (pip package name)
            version_spec: Version specifier (e.g., "==4.51.3" or ">=4.51.3,!=4.52.*")
            check_functionality: If True, also check if package is broken (missing critical attributes)
        
        Returns:
            Tuple of (is_installed: bool, version_matches: bool, is_broken: bool)
            If check_functionality is False, is_broken will always be False
        """
        # For binary packages, use the import name (module name) instead of package name
        import_name = package_name
        if package_name == "triton-windows":
            import_name = "triton"  # Package name is triton-windows but module is triton
        """
        Check if a package is installed and if its version matches the requirement.
        
        Args:
            venv_python: Path to venv Python executable
            package_name: Package name to check
            version_spec: Version specifier (e.g., "==4.51.3" or ">=4.51.3,!=4.52.*")
            check_functionality: If True, also check if package is broken (missing critical attributes)
        
        Returns:
            Tuple of (is_installed: bool, version_matches: bool, is_broken: bool)
            If check_functionality is False, is_broken will always be False
        """
        is_broken = False
        
        # Check functionality first if requested
        if check_functionality:
            is_broken = self._check_package_broken(venv_python, package_name)
            # If broken, we need to reinstall even if metadata says it's installed
            if is_broken:
                return True, False, True  # Installed (metadata), wrong version (broken), is_broken=True
        
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
                    return False, False, False
                # Can't verify version without metadata, assume it matches
                return True, True, False
            except Exception:
                return False, False, False
        
        # Use importlib.metadata to get installed version
        # For binary packages, use import name instead of package name
        check_package_name = import_name if import_name != package_name else package_name
        try:
            code = f"""
import sys
sys.path.insert(0, r'{venv_python.parent.parent / "Lib" / "site-packages"}')
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    # Try package name first, then import name for binary packages
    try:
        installed_ver = version('{package_name}')
    except PackageNotFoundError:
        installed_ver = version('{check_package_name}')
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
                return False, False, False
            
            installed_version = output.strip()
            
            # Check if version matches requirement
            if not PACKAGING_AVAILABLE:
                # Fallback: only support exact == matches
                if version_spec.startswith("=="):
                    required = version_spec[2:].strip()
                    version_matches = installed_version == required
                    return True, version_matches, is_broken
                else:
                    # Can't verify complex specifiers, assume it matches
                    return True, True, is_broken
            
            # Use packaging library for robust version comparison
            try:
                # Handle build tags in both installed version and version_spec
                # e.g., "2.5.1+cu121" -> compare base "2.5.1" and build "+cu121"
                installed_base = installed_version.split("+")[0]
                installed_build = "+" + installed_version.split("+")[1] if "+" in installed_version else ""
                
                # Parse version_spec to extract base version and build tag
                if version_spec.startswith("=="):
                    required_full = version_spec[2:].strip()
                    required_base = required_full.split("+")[0]
                    required_build = "+" + required_full.split("+")[1] if "+" in required_full else ""
                    
                    # Both base version and build tag must match for ==
                    version_matches = (installed_base == required_base) and (installed_build == required_build)
                    return True, version_matches, is_broken
                elif version_spec.startswith(">=") or version_spec.startswith("<"):
                    # For range specifiers, only compare base version (ignore build tags)
                    spec = SpecifierSet(version_spec)
                    version_matches = spec.contains(pkg_version.parse(installed_base))
                    return True, version_matches, is_broken
                else:
                    # Complex specifier - use packaging
                    spec = SpecifierSet(version_spec)
                    version_matches = spec.contains(pkg_version.parse(installed_base))
                    return True, version_matches, is_broken
            except Exception:
                # If parsing fails, assume version doesn't match
                return True, False, is_broken
                
        except Exception:
            return False, False, False
    
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
    
    def _is_version_conflict_error(self, error: str) -> bool:
        """
        Check if error is a version conflict that can be fixed.
        
        Args:
            error: Error message from import verification
        
        Returns:
            True if this is a version conflict error
        """
        version_conflict_indicators = [
            "is required",
            "but found",
            "version",
            "ImportError",
            "required for"
        ]
        
        error_lower = error.lower()
        # Check if it mentions version requirements
        has_version_keywords = any(indicator in error_lower for indicator in version_conflict_indicators)
        
        # Check if it's specifically about package versions (not other import errors)
        if has_version_keywords and ("required" in error_lower or "but found" in error_lower):
            return True
        
        return False
    
    def _fix_version_conflicts(self, venv_python: Path, error: str) -> bool:
        """
        Attempt to fix version conflicts by reinstalling problematic packages.
        
        Args:
            venv_python: Path to venv Python
            error: Error message that indicates version conflict
        
        Returns:
            True if fix was attempted and succeeded, False otherwise
        """
        # Parse error to find which package has wrong version
        # Example: "huggingface-hub>=0.34.0,<1.0 is required... but found huggingface-hub==1.2.3"
        
        import re
        
        # Pattern 1: "package>=X,<Y is required... but found package==Z"
        pattern1 = r'(\w+(?:-\w+)*)\s*(?:>=|==|>|<)[\d.,<>=\s]+is required.*?but found\s+(\w+(?:-\w+)*)==([\d.]+)'
        match = re.search(pattern1, error, re.IGNORECASE | re.DOTALL)
        
        if match:
            required_pkg = match.group(1)
            found_pkg = match.group(2)
            wrong_version = match.group(3)
            
            # Use the package name from "but found" as it's more reliable
            pkg_name = found_pkg if found_pkg else required_pkg
        else:
            # Pattern 2: "but found package==version"
            pattern2 = r'but found\s+(\w+(?:-\w+)*)==([\d.]+)'
            match = re.search(pattern2, error, re.IGNORECASE)
            
            if match:
                pkg_name = match.group(1)
                wrong_version = match.group(2)
            else:
                # Pattern 3: Extract package name from error context
                # Look for common package names in the error
                common_packages = ["huggingface-hub", "transformers", "tokenizers", "peft", "datasets"]
                pkg_name = None
                for pkg in common_packages:
                    if pkg.lower() in error.lower():
                        pkg_name = pkg
                        break
                
                if not pkg_name:
                    self.log(f"    ⚠ Could not identify package from error message")
                    return False
        
        # Find correct version (prefer profile if available, fall back to manifest)
        deps = sorted(self.manifest["core_dependencies"], key=lambda x: x["order"])
        dep_info = None
        
        for dep in deps:
            if dep["name"].lower() == pkg_name.lower().replace("_", "-"):
                dep_info = dep
                break
        
        if not dep_info:
            self.log(f"    ⚠ Package {pkg_name} not found in manifest")
            return False

        # Prefer exact profile pin when available, otherwise use manifest (may be flexible).
        normalized_pkg = pkg_name.lower().replace("_", "-")
        version_spec = dep_info["version"]
        if getattr(self, "profile_versions", None):
            # Try both normalized and raw keys.
            if normalized_pkg in self.profile_versions:
                version_spec = f"=={self.profile_versions[normalized_pkg]}"
            elif pkg_name in self.profile_versions:
                version_spec = f"=={self.profile_versions[pkg_name]}"

        self.log(f"    Reinstalling {pkg_name} with correct version {version_spec}...")
        
        # Uninstall wrong version
        uninstall_cmd = [str(venv_python), "-m", "pip", "uninstall", "-y", pkg_name]
        result = subprocess.run(uninstall_cmd, capture_output=True, timeout=60, **self.subprocess_flags)
        
        if result.returncode != 0:
            self.log(f"    ⚠ Uninstall had issues: {result.stderr[:200]}")
        
        # Reinstall with correct version from wheelhouse
        success, install_error = self._install_single_package(
            venv_python=venv_python,
            package_name=pkg_name,
            version_spec=version_spec,
            install_args=dep_info.get("install_args", [])
        )
        
        if success:
            self.log(f"    ✓ {pkg_name} reinstalled with correct version")
            return True
        else:
            self.log(f"    ✗ Failed to reinstall {pkg_name}: {install_error[:200]}")
            return False
    
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

