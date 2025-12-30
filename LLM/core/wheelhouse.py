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

# Try to import packaging for version validation
try:
    from packaging.specifiers import SpecifierSet
    from packaging import version as pkg_version
    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False
    SpecifierSet = None
    pkg_version = None


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
    
    def _extract_version_from_wheel(self, wheel_path: Path) -> Optional[str]:
        """
        Extract version from wheel filename.
        
        Args:
            wheel_path: Path to wheel file
            
        Returns:
            Version string or None if parsing fails
        """
        # Wheel filename format: package-version-pyver-abi-platform.whl
        # Example: transformers-4.57.3-cp312-cp312-win_amd64.whl
        wheel_name_parts = wheel_path.stem.split("-")
        if len(wheel_name_parts) >= 2:
            # Version is typically the second part, but may contain + for build tags
            # Handle cases like: torch-2.5.1+cu124-cp312...
            version_part = wheel_name_parts[1]
            # Check if next part is a build tag (starts with + or is a number)
            if len(wheel_name_parts) >= 3:
                # Check if third part looks like a build tag (contains + or is part of version)
                third_part = wheel_name_parts[2]
                if "+" in third_part or third_part.replace(".", "").isdigit():
                    # Version might span multiple parts
                    # Try to reconstruct: version+build
                    potential_version = f"{version_part}-{third_part}"
                    # Validate it looks like a version
                    if re.match(r'^[\d.]+[\+\-]?[\w.]*$', potential_version):
                        return potential_version.replace("-", "+", 1) if "+" not in potential_version else potential_version
            
            # Simple case: version is just the second part
            if re.match(r'^[\d.]+[\+\-]?[\w.]*$', version_part):
                return version_part
        
        return None
    
    def _validate_wheel_against_requirement(self, package_name: str, wheel_version: str, requirement_spec: str) -> bool:
        """
        Validate that a wheel version satisfies a requirement specifier.
        
        Args:
            package_name: Package name (for logging)
            wheel_version: Version extracted from wheel filename
            requirement_spec: Requirement specifier (e.g., ">=4.51.3,!=4.52.*,!=4.53.*")
            
        Returns:
            True if wheel version satisfies requirement, False otherwise
        """
        if not PACKAGING_AVAILABLE:
            # Fallback: only support exact == matches
            if requirement_spec.startswith("=="):
                required = requirement_spec[2:].strip()
                return wheel_version == required
            else:
                # Can't validate complex specifiers without packaging
                # Assume OK if we can't verify (conservative approach)
                return True
        
        try:
            # Parse requirement specifier
            spec = SpecifierSet(requirement_spec)
            
            # Parse wheel version (handle build tags like +cu124)
            # Extract base version before + if present
            base_version = wheel_version.split("+")[0].split("-")[0]
            
            # Check if version satisfies specifier
            return spec.contains(pkg_version.parse(base_version))
        except Exception as e:
            # If parsing fails, log warning and assume invalid
            self.log(f"  ⚠ Could not validate {package_name} {wheel_version} against {requirement_spec}: {e}")
            return False
    
    def _get_wheel_version(self, package_name: str) -> Optional[str]:
        """
        Get version of a package from wheelhouse.
        
        Args:
            package_name: Package name (normalized)
            
        Returns:
            Version string or None if not found
        """
        pkg_normalized = package_name.lower().replace("_", "-")
        
        for wheel in self.wheelhouse.glob("*.whl"):
            wheel_name_parts = wheel.stem.split("-")
            if len(wheel_name_parts) >= 1:
                wheel_pkg = wheel_name_parts[0].lower().replace("_", "-")
                if wheel_pkg == pkg_normalized:
                    return self._extract_version_from_wheel(wheel)
        
        return None
    
    def _validate_wheelhouse_requirements(self, package_versions: dict = None) -> Tuple[bool, str]:
        """
        Validate that existing wheels in wheelhouse satisfy current requirements.
        
        Args:
            package_versions: Optional dict of {package_name: exact_version} for profile-based mode.
                            If None, validates against manifest requirements.
        
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        validation_errors = []
        
        if package_versions:
            # Profile-based mode: validate against exact profile versions
            # Also check if wheels satisfy flexible requirements from manifest if they exist
            critical_packages = ["torch", "transformers", "tokenizers", "numpy", "peft", "accelerate"]
            
            for pkg_name in critical_packages:
                if pkg_name in package_versions:
                    expected_version = package_versions[pkg_name]
                    wheel_version = self._get_wheel_version(pkg_name)
                    
                    if not wheel_version:
                        validation_errors.append(f"{pkg_name}: wheel not found")
                        continue
                    
                    # Check exact match first (profile requirement)
                    if wheel_version != expected_version:
                        # Check if it satisfies flexible requirement from manifest
                        dep = next((d for d in self.manifest["core_dependencies"] if d["name"] == pkg_name), None)
                        if dep and dep.get("version") != "FROM_CUDA_CONFIG":
                            requirement_spec = dep["version"]
                            # If requirement is flexible (not ==), validate against it
                            if not requirement_spec.startswith("=="):
                                if self._validate_wheel_against_requirement(pkg_name, wheel_version, requirement_spec):
                                    # Wheel satisfies flexible requirement, that's OK
                                    continue
                        
                        validation_errors.append(f"{pkg_name}: wheel version {wheel_version} != expected {expected_version}")
        else:
            # Manifest-based mode: validate against requirements from dependencies.json
            deps = sorted(self.manifest["core_dependencies"], key=lambda x: x["order"])
            
            for dep in deps:
                pkg_name = dep["name"]
                
                # Skip CUDA packages (handled separately)
                if dep["version"] == "FROM_CUDA_CONFIG":
                    continue
                
                # Skip platform-specific packages
                if "platform" in dep and dep["platform"] != sys.platform:
                    continue
                
                # Only validate critical packages
                if not dep.get("critical", False):
                    continue
                
                version_spec = dep["version"]
                wheel_version = self._get_wheel_version(pkg_name)
                
                if not wheel_version:
                    validation_errors.append(f"{pkg_name}: wheel not found")
                    continue
                
                # Validate wheel version against requirement
                if not self._validate_wheel_against_requirement(pkg_name, wheel_version, version_spec):
                    validation_errors.append(f"{pkg_name}: wheel version {wheel_version} does not satisfy {version_spec}")
        
        if validation_errors:
            error_msg = "Wheelhouse validation failed:\n  " + "\n  ".join(validation_errors)
            return False, error_msg
        
        return True, ""
    
    def _validate_dependency_compatibility(self, package_versions: dict = None) -> Tuple[bool, str]:
        """
        Validate dependency compatibility BEFORE downloading anything.
        Checks known compatibility rules and ensures versions are compatible.
        
        Args:
            package_versions: Optional dict of {package_name: exact_version} for profile-based mode.
                            If None, validates against manifest requirements.
        
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        compatibility_errors = []
        
        # Known compatibility rules (package -> required dependency version)
        known_requirements = {
            "transformers": {
                "4.51.3": {"tokenizers": ">=0.22.0,<=0.23.0"},
                "4.57.3": {"tokenizers": ">=0.22.0,<=0.23.0"},
                # Add more as needed
            }
        }
        
        # Get package versions to check
        if package_versions:
            # Profile-based mode: use exact versions from profile
            versions_to_check = package_versions.copy()
        else:
            # Manifest-based mode: extract versions from dependencies.json
            versions_to_check = {}
            deps = sorted(self.manifest["core_dependencies"], key=lambda x: x["order"])
            for dep in deps:
                if dep["version"] != "FROM_CUDA_CONFIG" and dep.get("critical", False):
                    # For flexible requirements, we'll check compatibility rules
                    # but can't validate exact version without downloading
                    versions_to_check[dep["name"]] = dep["version"]
        
        # Check transformers -> tokenizers compatibility
        # This is the most common compatibility issue
        if "transformers" in versions_to_check and "tokenizers" in versions_to_check:
            transformers_spec = versions_to_check["transformers"]
            tokenizers_spec = versions_to_check["tokenizers"]
            
            # Extract minimum transformers version from spec (e.g., ">=4.51.3" -> "4.51.3")
            min_transformers_ver = None
            if isinstance(transformers_spec, str):
                if transformers_spec.startswith(">="):
                    min_transformers_ver = transformers_spec.split(">=")[1].split(",")[0].strip()
                elif transformers_spec.startswith("=="):
                    min_transformers_ver = transformers_spec.split("==")[1].strip()
            
            # Check if transformers version requires tokenizers>=0.22.0
            if min_transformers_ver:
                # Transformers 4.51.3+ requires tokenizers>=0.22.0,<=0.23.0
                requires_tokenizers_022 = False
                if PACKAGING_AVAILABLE:
                    try:
                        if pkg_version.parse(min_transformers_ver) >= pkg_version.parse("4.51.3"):
                            requires_tokenizers_022 = True
                    except Exception:
                        # Fallback: string comparison
                        if "4.51" in min_transformers_ver or "4.57" in min_transformers_ver:
                            requires_tokenizers_022 = True
                else:
                    # Fallback: simple string check
                    if "4.51" in min_transformers_ver or "4.57" in min_transformers_ver:
                        requires_tokenizers_022 = True
                
                if requires_tokenizers_022:
                    # Check if tokenizers spec allows 0.22.0+
                    if isinstance(tokenizers_spec, str):
                        # Check for common incompatible patterns
                        if "<0.22" in tokenizers_spec or "0.21" in tokenizers_spec:
                            compatibility_errors.append(
                                f"❌ DEPENDENCY CONFLICT DETECTED:\n"
                                f"   transformers {min_transformers_ver}+ requires tokenizers>=0.22.0,<=0.23.0\n"
                                f"   but dependencies.json specifies: tokenizers {tokenizers_spec}\n"
                                f"   This will cause installation to fail when transformers tries to import tokenizers."
                            )
                        elif not self._check_version_compatibility("tokenizers", tokenizers_spec, ">=0.22.0,<=0.23.0"):
                            compatibility_errors.append(
                                f"❌ DEPENDENCY CONFLICT DETECTED:\n"
                                f"   transformers {min_transformers_ver}+ requires tokenizers>=0.22.0,<=0.23.0\n"
                                f"   but dependencies.json specifies: tokenizers {tokenizers_spec}\n"
                                f"   These version ranges are incompatible."
                            )
        
        # Also check exact version matches in profile mode
        if package_versions:
            transformers_ver = package_versions.get("transformers", "")
            tokenizers_ver = package_versions.get("tokenizers", "")
            
            # Check if transformers version requires specific tokenizers version
            if transformers_ver and tokenizers_ver:
                # Extract base transformers version (remove +cu124 etc)
                base_tf_ver = transformers_ver.split("+")[0]
                
                # Check known requirements
                for known_ver, deps in known_requirements.get("transformers", {}).items():
                    if PACKAGING_AVAILABLE:
                        try:
                            if pkg_version.parse(base_tf_ver) >= pkg_version.parse(known_ver):
                                required_tokenizers = deps.get("tokenizers")
                                if required_tokenizers:
                                    # Check if tokenizers version satisfies requirement
                                    if not self._check_version_compatibility("tokenizers", tokenizers_ver, required_tokenizers):
                                        compatibility_errors.append(
                                            f"❌ DEPENDENCY CONFLICT: transformers {base_tf_ver} requires {required_tokenizers}, "
                                            f"but profile specifies tokenizers {tokenizers_ver}"
                                        )
                        except Exception:
                            pass
        
        if compatibility_errors:
            error_msg = "Dependency compatibility check failed BEFORE installation:\n\n"
            error_msg += "\n".join(compatibility_errors)
            error_msg += "\n\nPlease fix dependencies.json or compatibility_matrix.json before proceeding."
            return False, error_msg
        
        return True, ""
    
    def _check_version_compatibility(self, package_name: str, version_spec: str, required_spec: str) -> bool:
        """
        Check if a version specifier is compatible with a required specifier.
        
        Args:
            package_name: Package name (for logging)
            version_spec: Version specifier from manifest (e.g., ">=0.21.0,<0.22")
            required_spec: Required version specifier (e.g., ">=0.22.0,<=0.23.0")
        
        Returns:
            True if compatible, False otherwise
        """
        if not PACKAGING_AVAILABLE:
            # Fallback: simple string checks
            if "0.21" in version_spec and "0.22" in required_spec:
                return False
            return True
        
        try:
            # Parse both specifiers
            version_set = SpecifierSet(version_spec)
            required_set = SpecifierSet(required_spec)
            
            # Check if there's any overlap between the two specifiers
            # We need to find a version that satisfies both
            # This is a simplified check - we'll test a few candidate versions
            test_versions = ["0.21.0", "0.22.0", "0.22.1", "0.23.0", "0.24.0"]
            
            for test_ver in test_versions:
                try:
                    parsed_ver = pkg_version.parse(test_ver)
                    if version_set.contains(parsed_ver) and required_set.contains(parsed_ver):
                        # Found a version that satisfies both - they're compatible
                        return True
                except Exception:
                    continue
            
            # If no overlap found, they're incompatible
            return False
        except Exception as e:
            # If parsing fails, assume incompatible to be safe
            self.log(f"  ⚠ Could not check compatibility: {e}")
            return False
    
    def prepare_wheelhouse(self, cuda_config: str, python_version: Tuple[int, int], package_versions: dict = None, force_redownload: bool = False) -> Tuple[bool, str]:
        """
        Download all wheels to wheelhouse with exact versions.
        
        Args:
            cuda_config: CUDA configuration key (e.g., "cu124")
            python_version: Python version tuple (e.g., (3, 12))
            package_versions: Optional dict of {package_name: exact_version} from ProfileSelector.
                            If provided, uses these instead of manifest.
            force_redownload: If True, clear wheelhouse and re-download everything.
                            If False (default), skip if wheelhouse is already complete.
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            self.log(f"Preparing wheelhouse for {cuda_config}, Python {python_version[0]}.{python_version[1]}")
            
            # CRITICAL: Validate dependency compatibility BEFORE downloading anything
            self.log("Validating dependency compatibility...")
            is_compatible, error_msg = self._validate_dependency_compatibility(package_versions)
            if not is_compatible:
                self.log("")
                self.log("=" * 60)
                self.log("✗ DEPENDENCY COMPATIBILITY CHECK FAILED")
                self.log("=" * 60)
                self.log("")
                self.log(error_msg)
                self.log("")
                return False, error_msg
            
            self.log("✓ Dependency compatibility check passed")
            
            # Check if wheelhouse already has wheels (skip re-download unless forced)
            existing_wheels = list(self.wheelhouse.glob("*.whl"))
            if existing_wheels and not force_redownload:
                # Validate wheelhouse against current requirements
                self.log("Validating existing wheelhouse against current requirements...")
                is_valid, error_msg = self._validate_wheelhouse_requirements(package_versions)
                
                if not is_valid:
                    self.log(f"⚠ Wheelhouse validation failed:")
                    for line in error_msg.split("\n"):
                        if line.strip():
                            self.log(f"  {line}")
                    self.log("Clearing wheelhouse and re-downloading with correct versions...")
                    self._clear_wheelhouse()
                else:
                    self.log(f"✓ Wheelhouse validation passed - {len(existing_wheels)} wheels satisfy current requirements")
                    self.log("Skipping re-download (use force_redownload=True to force)")
                    return True, ""
            
            # Clear existing wheelhouse only if forced or if starting fresh
            elif force_redownload:
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
                
                # Fallback for transformers: if exact version fails, try flexible specifier
                if not success and pkg_name == "transformers" and pkg_version == "4.51.3":
                    self.log(f"  ⚠ Exact version {pkg_version} not available, trying flexible specifier...")
                    flexible_spec = "transformers>=4.51.3,!=4.52.*,!=4.53.*,!=4.54.*,!=4.55.*,!=4.57.0,<4.58"
                    success, error = self._download_wheel(
                        package=flexible_spec,
                        index_url=None,
                        no_deps=False,
                        python_version=python_version
                    )
                    if success:
                        self.log(f"  ✓ Downloaded transformers with flexible version specifier")
                
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
                
                # Note: version_spec is already flexible for transformers (from dependencies.json update)
                # So no fallback needed here - pip will automatically select 4.57.3 if 4.51.3 isn't available
                
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

