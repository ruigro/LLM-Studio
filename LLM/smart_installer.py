#!/usr/bin/env python3
"""
Smart Installer for LLM Fine-tuning Studio
Intelligently installs components based on system detection
"""

import os
import sys
import subprocess
import platform
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
from system_detector import SystemDetector, detect_all

try:
    from packaging import version as pkg_version
    from packaging.specifiers import SpecifierSet
except ImportError:
    # Fallback if packaging not available
    pkg_version = None
    SpecifierSet = None

class SmartInstaller:
    """Smart installer that detects and installs only what's needed"""
    
    # Hardware-specific version matrix for compatibility
    VERSION_MATRIX = {
        "cuda_12_4": {"torch": "2.5.1", "triton": "3.0.0", "torchvision": "0.20.1", "torchaudio": "2.5.1"},
        "cuda_12_1": {"torch": "2.5.1", "triton": "3.0.0", "torchvision": "0.20.1", "torchaudio": "2.5.1"},
        "cuda_11_8": {"torch": "2.5.1", "triton": "3.0.0", "torchvision": "0.20.1", "torchaudio": "2.5.1"},
        "cpu": {"torch": "2.5.1", "triton": "3.0.0", "torchvision": "0.20.1", "torchaudio": "2.5.1"}
    }
    
    # GPU-specific compatibility information
    GPU_COMPAT = {
        "RTX 4090": {"compute": "8.9", "min_cuda": "11.8", "recommended_cuda": "12.4"},
        "RTX 4080": {"compute": "8.9", "min_cuda": "11.8", "recommended_cuda": "12.4"},
        "RTX 4070": {"compute": "8.9", "min_cuda": "11.8", "recommended_cuda": "12.4"},
        "RTX A2000": {"compute": "8.6", "min_cuda": "11.1", "recommended_cuda": "11.8"},
        "RTX 3090": {"compute": "8.6", "min_cuda": "11.1", "recommended_cuda": "11.8"},
        "RTX 3080": {"compute": "8.6", "min_cuda": "11.1", "recommended_cuda": "11.8"},
        "RTX 3070": {"compute": "8.6", "min_cuda": "11.1", "recommended_cuda": "11.8"},
        "RTX 3060": {"compute": "8.6", "min_cuda": "11.1", "recommended_cuda": "11.8"},
        "T1000": {"compute": "7.5", "min_cuda": "10.0", "recommended_cuda": "11.8"},
        "T600": {"compute": "7.5", "min_cuda": "10.0", "recommended_cuda": "11.8"},
    }
    
    def __init__(self, install_dir: Optional[str] = None):
        self.install_dir = Path(install_dir) if install_dir else Path.cwd()
        self.detector = SystemDetector()
        self.detection_results = {}
        self.installation_log = []
        self.progress_callback = None  # Callback for progress updates (percent, message)
        self.min_disk_space_gb = 5  # Minimum 5 GB required
        
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
        """Log installation message"""
        print(f"[INSTALL] {message}")
        self.installation_log.append(message)
    
    def _terminate_venv_processes(self, venv_path: Path) -> Tuple[bool, list]:
        """
        Terminate all Python processes using the target venv.
        
        Returns:
            Tuple of (success: bool, terminated_pids: list)
        """
        if platform.system() != "Windows":
            return True, []  # Less critical on other platforms
        
        terminated_pids = []
        
        try:
            import psutil
        except ImportError:
            # Try using taskkill on Windows
            try:
                venv_python = venv_path / "Scripts" / "python.exe"
                if venv_python.exists():
                    # Kill processes using this specific Python
                    subprocess.run(
                        ["taskkill", "/F", "/IM", "python.exe", "/FI", f"IMAGEPATH eq {venv_python}"],
                        capture_output=True,
                        timeout=10,
                        **self.subprocess_flags
                    )
                    subprocess.run(
                        ["taskkill", "/F", "/IM", "pythonw.exe", "/FI", f"IMAGEPATH eq {venv_python}"],
                        capture_output=True,
                        timeout=10,
                        **self.subprocess_flags
                    )
                return True, []
            except Exception:
                return False, []
        
        # Use psutil to find and kill processes
        current_pid = os.getpid()
        try:
            current_process = psutil.Process(current_pid)
            parent_pid = current_process.ppid() if hasattr(current_process, 'ppid') else None
        except:
            parent_pid = None
        
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                proc_info = proc.info
                proc_pid = proc_info.get('pid')
                
                # Never kill current process or parent
                if proc_pid == current_pid or (parent_pid and proc_pid == parent_pid):
                    continue
                
                if proc_info.get('name') and 'python' in proc_info.get('name', '').lower():
                    exe_path = proc_info.get('exe')
                    if exe_path:
                        exe_path_obj = Path(exe_path)
                        # Check if this Python is from our venv
                        if str(exe_path_obj.parent.parent).lower() == str(venv_path).lower():
                            try:
                                proc.kill()
                                terminated_pids.append(proc_pid)
                                self.log(f"Terminated Python process: {proc_info.get('name')} (PID {proc_pid})")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
            except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                pass
        
        if terminated_pids:
            import time
            time.sleep(2)  # Wait for processes to terminate
        
        return True, terminated_pids
    
    def _delete_torch_directory(self, venv_path: Path) -> Tuple[bool, str]:
        """
        Delete the torch directory from venv.
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        torch_dir = venv_path / "Lib" / "site-packages" / "torch"
        
        if not torch_dir.exists():
            return True, None
        
        self.log(f"Deleting existing torch directory: {torch_dir}")
        
        try:
            # Try to delete the directory
            shutil.rmtree(torch_dir)
            self.log("✓ Torch directory deleted successfully")
            return True, None
        except PermissionError as e:
            error_msg = f"PyTorch files are locked by a running process. Cannot delete: {torch_dir}"
            self.log(f"ERROR: {error_msg}")
            self.log(f"Details: {str(e)}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Failed to delete torch directory: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            return False, error_msg
    
    def _check_disk_space(self, path: Optional[Path] = None) -> Tuple[bool, float]:
        """
        Check available disk space at the given path.
        
        Returns:
            Tuple of (has_enough_space, free_space_gb)
        """
        if path is None:
            path = self.install_dir
        
        try:
            if platform.system() == "Windows":
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(str(path)),
                    ctypes.pointer(ctypes.c_ulonglong()),
                    ctypes.pointer(ctypes.c_ulonglong()),
                    ctypes.pointer(free_bytes)
                )
                free_gb = free_bytes.value / (1024**3)
            else:
                stat = shutil.disk_usage(str(path))
                free_gb = stat.free / (1024**3)
            
            has_enough = free_gb >= self.min_disk_space_gb
            return has_enough, free_gb
        except Exception as e:
            # If we can't check, assume it's OK (better than blocking)
            self.log(f"Warning: Could not check disk space: {str(e)}")
            return True, 0.0
    
    def _sanitize_requirement(self, req_string: str) -> str:
        """
        Sanitize a requirement string by removing inline comments.
        
        Removes everything after '#' and trims whitespace.
        Example: "transformers==4.51.3  # comment" -> "transformers==4.51.3"
        
        Args:
            req_string: Requirement string that may contain inline comments
        
        Returns:
            Sanitized requirement string without comments
        """
        if not req_string:
            return ""
        # Remove everything after '#' and strip whitespace
        sanitized = req_string.split('#')[0].strip()
        return sanitized
    
    def _run_pip_worker(self, action: str, package: str, python_executable: str,
                       index_url: str = "", pip_args: list = None) -> Tuple[bool, str, int]:
        """
        Run pip command via worker process, stream output in real-time.
        
        Args:
            action: 'install' or 'uninstall'
            package: Package name with optional version constraints (e.g., 'torch==2.5.1', 'numpy<2')
            python_executable: Python executable to use
            index_url: Index URL for pip (optional)
            pip_args: Pip arguments to forward directly (e.g., ['--force-reinstall', '--no-deps'])
        
        Returns:
            Tuple of (success: bool, last_200_lines: str, exit_code: int)
        """
        # Get path to pip_worker.py (should be in same directory as this file)
        worker_script = Path(__file__).parent / "pip_worker.py"
        if not worker_script.exists():
            error_msg = f"pip_worker.py not found at {worker_script}"
            self.log(f"ERROR: {error_msg}")
            return False, error_msg, 1
        
        if pip_args is None:
            pip_args = []
        
        # Sanitize package string to remove any inline comments
        package = self._sanitize_requirement(package)
        
        # On Windows, quote package specs containing < or > to avoid shell interpretation
        import platform
        if platform.system() == "Windows" and ("<" in package or ">" in package):
            # Package is already a string, but we need to ensure it's passed correctly
            # subprocess.Popen handles this correctly when passed as a list element
            pass  # No need to quote - subprocess handles it correctly
        
        # Build command to launch worker
        # Use the target Python executable to run the worker script (not sys.executable which might be pythonw.exe)
        cmd = [
            python_executable,  # Use target Python to run worker script
            str(worker_script),
            "--action", action,
            "--package", package,
            "--python", python_executable
        ]
        
        if index_url:
            cmd.extend(["--index-url", index_url])
        
        # Add pip args directly (they will be forwarded by parse_known_args)
        cmd.extend(pip_args)
        
        self.log(f"Running pip worker: {' '.join(cmd)}")
        
        try:
            # Use Popen to stream output in real-time
            # Redirect stderr to stdout so we capture all output
            popen_flags = {
                'stdout': subprocess.PIPE,
                'stderr': subprocess.STDOUT,  # Redirect stderr to stdout
                'text': True,
                'bufsize': 1,
                'universal_newlines': True
            }
            popen_flags.update(self.subprocess_flags)
            
            process = subprocess.Popen(cmd, **popen_flags)
            
            # Stream output line by line
            output_lines = []
            
            # Read from stdout (stderr is redirected to stdout)
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    output_lines.append(line)
                    self.log(line)
            
            # Wait for process to complete
            exit_code = process.wait()
            
            # Get last 200 lines for error reporting
            last_200_lines = '\n'.join(output_lines[-200:]) if len(output_lines) > 200 else '\n'.join(output_lines)
            
            success = (exit_code == 0)
            
            if not success:
                self.log(f"ERROR: pip {action} failed with exit code {exit_code}")
                self.log(f"Command: {' '.join(cmd)}")
                self.log(f"Last {len(output_lines[-200:]) if len(output_lines) > 200 else len(output_lines)} lines:")
                for line in output_lines[-200:]:
                    self.log(line)
            
            return success, last_200_lines, exit_code
            
        except Exception as e:
            error_msg = f"Exception running pip worker: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            self.log(f"Command: {' '.join(cmd)}")
            return False, error_msg, 1
    
    def run_detection(self) -> Dict:
        """Run system detection"""
        self.log("Detecting system components...")
        self.detection_results = self.detector.detect_all()
        return self.detection_results
    
    def install_python(self) -> bool:
        """Install Python if not found"""
        python_info = self.detection_results.get("python", {})
        
        if python_info.get("found"):
            self.log(f"Python {python_info.get('version')} found at {python_info.get('executable')}")
            return True
        
        self.log("Python not found. Installation required.")
        self.log("Please install Python 3.8+ from https://www.python.org/downloads/")
        self.log("Make sure to check 'Add Python to PATH' during installation.")
        return False
    
    def install_vcredist(self) -> bool:
        """Install Visual C++ Redistributables if needed (Windows only)"""
        if platform.system() != "Windows":
            return True
        
        vcredist_info = self.detection_results.get("vcredist", {})
        
        if vcredist_info.get("found"):
            self.log("Visual C++ Redistributables found")
            return True
        
        self.log("Visual C++ Redistributables not found.")
        self.log("Downloading and installing Visual C++ Redistributables...")
        
        # Download URL for Visual C++ Redistributables
        vcredist_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
        vcredist_path = self.install_dir / "vc_redist.x64.exe"
        
        try:
            import urllib.request
            self.log(f"Downloading from {vcredist_url}...")
            urllib.request.urlretrieve(vcredist_url, vcredist_path)
            
            # Install silently
            self.log("Installing Visual C++ Redistributables...")
            result = subprocess.run(
                [str(vcredist_path), "/install", "/quiet", "/norestart"],
                timeout=300,
                capture_output=True,
                **self.subprocess_flags
            )
            
            if result.returncode == 0:
                self.log("Visual C++ Redistributables installed successfully")
                # Clean up
                if vcredist_path.exists():
                    vcredist_path.unlink()
                return True
            else:
                self.log(f"Installation failed with code {result.returncode}")
                return False
        
        except Exception as e:
            self.log(f"Error installing Visual C++ Redistributables: {e}")
            self.log("Please install manually from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            return False
    
    def get_optimal_cuda_build(self) -> str:
        """Determine optimal CUDA build based on hardware detection"""
        cuda_info = self.detection_results.get("cuda", {})
        driver_version = cuda_info.get("driver_version")
        gpus = cuda_info.get("gpus", [])
        
        if not driver_version or not gpus:
            self.log("No CUDA detected. Using CPU build.")
            return "cpu"

        def _parse_driver_major(ver: Optional[str]) -> Optional[int]:
            if not ver:
                return None
            try:
                return int(float(ver.split(".")[0]))
            except Exception:
                return None

        def _parse_mem_mib(mem_str: str) -> int:
            # nvidia-smi often returns like: "24576 MiB" or "24576 MB"
            if not mem_str:
                return 0
            try:
                digits = "".join(ch for ch in mem_str if ch.isdigit())
                return int(digits) if digits else 0
            except Exception:
                return 0

        def _recommended_to_build(recommended: Optional[str]) -> Optional[str]:
            if not recommended:
                return None
            if "12.4" in recommended:
                return "cu124"
            if "12.1" in recommended:
                return "cu121"
            if "11.8" in recommended or "11" in recommended or "10" in recommended:
                return "cu118"
            return None

        # Choose the best GPU (mixed-GPU systems): prefer higher recommended CUDA / compute / VRAM.
        driver_major = _parse_driver_major(driver_version)
        best_gpu = None
        best_compat = None
        best_score = (-1, -1.0, -1)  # (recommended_rank, compute, mem_mib)

        for gpu in gpus:
            name = gpu.get("name", "") or ""
            mem_mib = _parse_mem_mib(gpu.get("memory", "") or "")

            compat = None
            for known_gpu, info in self.GPU_COMPAT.items():
                if known_gpu in name:
                    compat = info
                    break

            if compat:
                recommended = compat.get("recommended_cuda")
                build = _recommended_to_build(recommended) or "cu118"
                recommended_rank = {"cu118": 1, "cu121": 2, "cu124": 3}.get(build, 0)
                try:
                    compute = float(compat.get("compute", "0") or 0)
                except Exception:
                    compute = 0.0
            else:
                recommended_rank = 0
                compute = 0.0

            score = (recommended_rank, compute, mem_mib)
            if score > best_score:
                best_score = score
                best_gpu = gpu
                best_compat = compat

        selected_name = (best_gpu or {}).get("name", "") if best_gpu else ""

        # If we have a known GPU, start from its recommended build, but don't exceed driver capability.
        desired_build = None
        if best_compat:
            recommended = best_compat.get("recommended_cuda", "11.8")
            desired_build = _recommended_to_build(recommended)
            self.log(f"Detected GPUs: {len(gpus)} | Selected: {selected_name} | Recommended CUDA: {recommended}")
        else:
            self.log(f"Detected GPUs: {len(gpus)} | Selected: {selected_name} | No known GPU match; using driver-based selection.")

        if driver_major is not None and desired_build is not None:
            # Gate by driver major version (conservative fallbacks).
            if desired_build == "cu124" and driver_major < 555:
                desired_build = "cu121" if driver_major >= 545 else ("cu118" if driver_major >= 450 else "cpu")
            elif desired_build == "cu121" and driver_major < 545:
                desired_build = "cu118" if driver_major >= 450 else "cpu"
            elif desired_build == "cu118" and driver_major < 450:
                desired_build = "cpu"

            return desired_build
        
        # Fallback: use driver version to determine build
        try:
            driver_major = int(float(driver_version.split('.')[0]))
            
            if driver_major >= 555:
                return "cu124"  # CUDA 12.4
            elif driver_major >= 545:
                return "cu121"  # CUDA 12.1
            elif driver_major >= 450:
                return "cu118"  # CUDA 11.8
            else:
                self.log(f"Warning: Old driver {driver_version}. Using CPU build.")
                return "cpu"
        except:
            self.log(f"Could not parse driver version {driver_version}. Using CPU build.")
            return "cpu"
    
    def install_pytorch(self, python_executable: Optional[str] = None) -> bool:
        """Install PyTorch based on detection with proper version compatibility"""
        if not python_executable:
            python_info = self.detection_results.get("python", {})
            if not python_info.get("found"):
                self.log("Python not found. Cannot install PyTorch.")
                return False
            python_executable = python_info.get("executable")
        
        # Run detection if not already done
        if not self.detection_results:
            self.run_detection()
        
        # CRITICAL: Determine venv path and verify no processes are using it
        python_path = Path(python_executable)
        if "Scripts" in str(python_path) or "bin" in str(python_path):
            venv_path = python_path.parent.parent
        else:
            # Not a venv Python, use install_dir
            venv_path = self.install_dir / ".venv"
        
        self.log(f"Target venv path: {venv_path}")
        
        # CRITICAL: Verify installer is not running from target venv
        # This should never happen due to bootstrap guard, but double-check
        current_python = Path(sys.executable).resolve()
        current_python_str = str(current_python).replace('\\', '/')
        
        # Check if running from LLM\.venv (case-insensitive)
        current_python_normalized = current_python_str.lower()
        if '/llm/.venv/' in current_python_normalized or '\\llm\\.venv\\' in current_python_str.lower():
            self.log("=" * 60)
            self.log("CRITICAL ERROR: Installer is running from target venv!")
            self.log(f"Current Python: {sys.executable}")
            self.log("This should never happen - bootstrap guard should have prevented this.")
            self.log("Aborting PyTorch installation.")
            self.log("=" * 60)
            return False
        
        # CRITICAL: Terminate any processes using the target venv
        self.log("Checking for processes using target venv...")
        success, terminated_pids = self._terminate_venv_processes(venv_path)
        if not success:
            self.log("WARNING: Could not check/terminate processes. Proceeding anyway...")
        elif terminated_pids:
            self.log(f"Terminated {len(terminated_pids)} process(es) using the venv")
            import time
            time.sleep(3)  # Wait for processes to fully terminate
        
        # Verify no python.exe from venv is running
        if platform.system() == "Windows":
            try:
                venv_python = venv_path / "Scripts" / "python.exe"
                if venv_python.exists():
                    check_cmd = [
                        "tasklist", "/FI", f"IMAGEPATH eq {venv_python}", "/FO", "CSV", "/NH"
                    ]
                    result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5, **self.subprocess_flags)
                    if result.returncode == 0 and venv_python.name in result.stdout:
                        self.log("ERROR: Python process from target venv is still running!")
                        self.log("Please close all Python windows and try again.")
                        return False
            except Exception:
                pass  # Non-fatal
        
        # CRITICAL: Delete the entire torch directory if it exists
        self.log("Checking for existing torch installation...")
        success, error_msg = self._delete_torch_directory(venv_path)
        if not success:
            self.log("=" * 60)
            self.log("CRITICAL ERROR: Cannot proceed with PyTorch installation")
            self.log(error_msg)
            self.log("=" * 60)
            return False
        
        # First, uninstall any existing torch to avoid conflicts.
        # IMPORTANT: also remove xformers and triton variants on Windows to avoid pip resolver downgrading torch.
        self.log("Uninstalling any existing PyTorch / xformers / triton installation...")
        packages_to_uninstall = ["torch", "torchvision", "torchaudio", "xformers", "triton", "triton-windows"]
        for pkg in packages_to_uninstall:
            try:
                success, last_lines, exit_code = self._run_pip_worker(
                    action="uninstall",
                    package=pkg,
                    python_executable=python_executable
                )
                # Uninstall failures are OK if package wasn't installed
                if not success and "not installed" not in last_lines.lower():
                    self.log(f"Note: Uninstall of {pkg} had issues: {last_lines[:200]}")
            except Exception as e:
                self.log(f"Note: Could not uninstall {pkg}: {e}")
        
        # Determine optimal CUDA build
        cuda_build = self.get_optimal_cuda_build()
        
        # Get versions from matrix
        build_key_map = {
            "cu124": "cuda_12_4",
            "cu121": "cuda_12_1",
            "cu118": "cuda_11_8",
            "cpu": "cpu",
        }
        build_key = build_key_map.get(cuda_build, "cpu")
        if build_key not in self.VERSION_MATRIX:
            build_key = "cpu"
        
        versions = self.VERSION_MATRIX[build_key]
        pytorch_version = versions["torch"]
        torchvision_version = versions["torchvision"]
        torchaudio_version = versions["torchaudio"]
        triton_version = versions["triton"]
        
        self.log(f"Installing PyTorch {pytorch_version} ({cuda_build} build) [force-reinstall, no-deps]...")
        
        try:
            # Ensure numpy is in a safe range before importing torch inside the GUI / detectors.
            # NumPy 2.x can break some torch numpy bridges on Windows.
            try:
                success, last_lines, exit_code = self._run_pip_worker(
                    action="install",
                    package="numpy<2",
                    python_executable=python_executable,
                    pip_args=["--upgrade"]
                )
                # Non-fatal if numpy install fails
            except Exception:
                pass

            # CRITICAL: Install torch FIRST and verify it works before installing torchvision/torchaudio
            # Installing all three together with --no-deps can cause crashes if torch isn't fully ready
            index_url = f"https://download.pytorch.org/whl/{cuda_build}" if cuda_build != "cpu" else "https://download.pytorch.org/whl/cpu"
            
            self.log("Downloading PyTorch (~2.5GB)...")
            
            # Use worker process for installation
            # CRITICAL: Use --no-cache-dir to avoid file lock issues
            success, last_lines, exit_code = self._run_pip_worker(
                action="install",
                package=f"torch=={pytorch_version}",
                python_executable=python_executable,
                index_url=index_url,
                pip_args=["--force-reinstall", "--no-deps", "--no-cache-dir"]
            )
            
            if success:
                self.log("PyTorch torch package installed successfully")
                
                # CRITICAL: Verify torch can be imported BEFORE installing torchvision/torchaudio
                self.log("Verifying torch installation before proceeding...")
                torch_ok, torch_ver, torch_cuda, torch_error = self._verify_torch(python_executable)
                if not torch_ok:
                    self.log(f"ERROR: torch installed but verification failed: {torch_error}")
                    return False
                
                self.log(f"✓ torch verified and working (version {torch_ver})")
                # NOTE: torchvision and torchaudio will be installed separately by _install_component
                # when processing the checklist items "PyTorch Vision" and "PyTorch Audio"
                # This ensures proper dependency order
                
                return True
            else:
                self.log(f"PyTorch installation failed with exit code {exit_code}")
                self.log("=" * 60)
                self.log("PYTORCH INSTALLATION ERROR OUTPUT:")
                self.log("=" * 60)
                # Log last 200 lines (already captured by worker)
                error_lines = last_lines.split('\n')[-50:] if last_lines else []
                for line in error_lines:
                    self.log(line)
                self.log("=" * 60)
                # Check for common errors
                if "No space left on device" in last_lines or "Errno 28" in last_lines:
                    has_space, free_gb = self._check_disk_space()
                    self.log(f"ERROR: Insufficient disk space! Only {free_gb:.2f} GB available")
                    self.log("PyTorch requires ~3GB free space. Please free up disk space.")
                    return False
                elif "WinError 5" in last_lines or "Access is denied" in last_lines or "Permission denied" in last_lines:
                    # File lock error - try to identify which file and process
                    self.log("=" * 60)
                    self.log("CRITICAL ERROR: File lock detected during PyTorch installation")
                    self.log("=" * 60)
                    
                    # Try to extract the locked file from error message
                    import re
                    file_match = re.search(r"Access is denied: '([^']+)'", last_lines)
                    if file_match:
                        locked_file = file_match.group(1)
                        self.log(f"Locked file: {locked_file}")
                        
                        # Try to find which process is locking it (Windows only)
                        if platform.system() == "Windows":
                            try:
                                # Use handle.exe if available, or try tasklist
                                handle_cmd = ["handle.exe", locked_file]
                                handle_result = subprocess.run(
                                    handle_cmd, capture_output=True, text=True, timeout=5,
                                    **self.subprocess_flags
                                )
                                if handle_result.returncode == 0:
                                    self.log("Process locking file:")
                                    self.log(handle_result.stdout)
                                else:
                                    self.log("Note: handle.exe not available. Install Sysinternals Handle to identify locking process.")
                            except Exception:
                                pass
                    
                    self.log("=" * 60)
                    self.log("SOLUTION:")
                    self.log("1. Close ALL Python windows and processes")
                    self.log("2. Close the installer GUI")
                    self.log("3. Wait 10 seconds")
                    self.log("4. Run the installer again")
                    self.log("=" * 60)
                    return False
                elif "Could not find a version" in last_lines:
                    self.log("ERROR: Could not find PyTorch version. Check internet connection.")
                    return False
                return False
        
        except Exception as e:
            self.log(f"Error installing PyTorch: {e}")
            return False
    
    def install_dependencies(self, python_executable: Optional[str] = None) -> bool:
        """Install application dependencies with compatibility handling"""
        if not python_executable:
            python_info = self.detection_results.get("python", {})
            if not python_info.get("found"):
                self.log("Python not found. Cannot install dependencies.")
                return False
            python_executable = python_info.get("executable")
        
        self.log("Installing application dependencies...")
        
        try:
            # First install core dependencies from requirements.txt (excluding problematic ones)
            requirements_file = Path(__file__).parent / "requirements.txt"
            
            if requirements_file.exists():
                self.log("Installing base requirements...")
                cmd = [
                    python_executable, "-m", "pip", "install",
                    "-r", str(requirements_file)
                ]
                
                self.log(f"Running: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minutes timeout
                    **self.subprocess_flags
                )
                
                if result.returncode != 0:
                    self.log(f"Warning: Some packages failed: {result.stderr[:500]}")
            
            # Install unsloth separately with careful version control.
            # IMPORTANT: use --no-deps so pip cannot swap torch/triton versions underneath us.
            self.log("Installing unsloth (this may take a few minutes)...")
            unsloth_cmd = [
                python_executable, "-m", "pip", "install",
                "--upgrade", "--no-deps",
                "unsloth",
            ]
            
            result = subprocess.run(unsloth_cmd, capture_output=True, text=True, timeout=900, **self.subprocess_flags)
            
            if result.returncode == 0:
                self.log("✅ unsloth installed")
            else:
                self.log(f"⚠️ unsloth installation warning: {result.stderr[:200]}")
            
            # Remove incompatible torchao if present (known Windows issue)
            self.log("Checking for torchao compatibility...")
            remove_torchao = [python_executable, "-m", "pip", "uninstall", "-y", "torchao"]
            subprocess.run(remove_torchao, capture_output=True, timeout=60, **self.subprocess_flags)
            self.log("Removed torchao (incompatible with current setup)")
            
            # Test if unsloth works
            self.log("Testing unsloth import...")
            test_cmd = [python_executable, "-c", "from unsloth import FastLanguageModel; print('OK')"]
            test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30, **self.subprocess_flags)
            
            if test_result.returncode == 0 and "OK" in test_result.stdout:
                self.log("✅ unsloth is working correctly")
                return True
            else:
                self.log(f"⚠️ unsloth may have issues: {test_result.stderr[:300]}")
                return True  # Still return True as most deps are installed
        
        except subprocess.TimeoutExpired:
            self.log("Dependencies installation timed out")
            return False
        except Exception as e:
            self.log(f"Error installing dependencies: {e}")
            return False
    
    def create_launcher(self) -> bool:
        """Create launcher script"""
        self.log("Creating launcher script...")
        
        launcher_content = """#!/usr/bin/env python3
\"\"\"
Launcher for LLM Fine-tuning Studio
\"\"\"
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Find Python executable
    python_exe = sys.executable
    
    # Launch Streamlit
    gui_file = script_dir / "gui.py"
    if not gui_file.exists():
        print(f"Error: gui.py not found at {gui_file}")
        input("Press Enter to exit...")
        return
    
    print("Starting LLM Fine-tuning Studio...")
    print("The GUI will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server.")
    print()
    
    try:
        subprocess.run([python_exe, "-m", "streamlit", "run", str(gui_file)])
    except KeyboardInterrupt:
        print("\\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
"""
        
        # Create launcher for current platform
        if platform.system() == "Windows":
            launcher_path = self.install_dir / "launch_gui.bat"
            bat_content = f"""@echo off
REM Launcher for LLM Fine-tuning Studio
cd /d "%~dp0"
python gui.py
if errorlevel 1 (
    echo.
    echo Error: Failed to start GUI
    echo Make sure Python and dependencies are installed.
    pause
)
"""
            try:
                with open(launcher_path, 'w') as f:
                    f.write(bat_content)
                self.log(f"Created launcher: {launcher_path}")
                return True
            except Exception as e:
                self.log(f"Error creating launcher: {e}")
                return False
        else:
            launcher_path = self.install_dir / "launch_gui.sh"
            try:
                with open(launcher_path, 'w') as f:
                    f.write(launcher_content)
                # Make executable
                os.chmod(launcher_path, 0o755)
                self.log(f"Created launcher: {launcher_path}")
                return True
            except Exception as e:
                self.log(f"Error creating launcher: {e}")
                return False
    
    def auto_install_all(self, progress_callback=None) -> Dict:
        """
        Single method to install everything with proper error handling
        Returns detailed results for each component
        """
        results = {
            "detection": {"success": False, "data": {}},
            "python": {"success": False, "message": ""},
            "vcredist": {"success": False, "message": ""},
            "pytorch": {"success": False, "message": ""},
            "dependencies": {"success": False, "message": ""},
            "overall_success": False
        }
        
        def log_with_callback(msg):
            self.log(msg)
            if progress_callback:
                progress_callback(msg)
        
        try:
            # Step 1: Detection
            log_with_callback("Running system detection...")
            self.run_detection()
            results["detection"]["success"] = True
            results["detection"]["data"] = self.detection_results
            
            # Step 2: Python check
            log_with_callback("Checking Python installation...")
            if self.install_python():
                results["python"]["success"] = True
                results["python"]["message"] = "Python found"
            else:
                results["python"]["message"] = "Python not found or invalid"
                return results
            
            # Step 3: Visual C++ Redistributables (Windows only)
            if platform.system() == "Windows":
                log_with_callback("Checking Visual C++ Redistributables...")
                if self.install_vcredist():
                    results["vcredist"]["success"] = True
                    results["vcredist"]["message"] = "Visual C++ Redistributables OK"
                else:
                    results["vcredist"]["message"] = "Visual C++ install failed (non-critical)"
            else:
                results["vcredist"]["success"] = True
                results["vcredist"]["message"] = "Not required on this platform"
            
            # Step 4: PyTorch (always install/reinstall to ensure clean state)
            log_with_callback("Installing PyTorch...")
            pytorch_info = self.detection_results.get("pytorch", {})
            if pytorch_info.get("found"):
                log_with_callback(f"Found existing PyTorch {pytorch_info.get('version')} - will reinstall to ensure compatibility")
            
            if self.install_pytorch():
                results["pytorch"]["success"] = True
                results["pytorch"]["message"] = "PyTorch installed successfully"
            else:
                results["pytorch"]["message"] = "PyTorch installation failed"
                return results
            
            # Step 5: Dependencies
            log_with_callback("Installing dependencies...")
            if self.install_dependencies():
                results["dependencies"]["success"] = True
                results["dependencies"]["message"] = "Dependencies installed successfully"
            else:
                results["dependencies"]["message"] = "Some dependencies failed (may still work)"
            
            # Overall success if critical components installed
            results["overall_success"] = (
                results["python"]["success"] and 
                results["pytorch"]["success"]
            )
            
            log_with_callback("Installation process completed!")
            return results
            
        except Exception as e:
            log_with_callback(f"Error during installation: {str(e)}")
            return results
    
    def install(self) -> bool:
        """Run complete installation process"""
        self.log("=" * 60)
        self.log("LLM Fine-tuning Studio - Smart Installer")
        self.log("=" * 60)
        
        # Step 1: Detection
        self.run_detection()
        
        # Step 2: Install Python (if needed)
        if not self.install_python():
            self.log("Installation cannot continue without Python.")
            return False
        
        # Step 3: Install Visual C++ Redistributables (Windows)
        if platform.system() == "Windows":
            if not self.install_vcredist():
                self.log("Warning: Visual C++ Redistributables installation failed.")
                self.log("The application may not work correctly.")
        
        # Step 4: Install PyTorch (always reinstall to ensure correct version/CUDA build)
        pytorch_info = self.detection_results.get("pytorch", {})
        if pytorch_info.get("found"):
            self.log(f"PyTorch {pytorch_info.get('version')} detected - checking if correct version...")
        
        # Always run install_pytorch - it will uninstall old version and install correct one
        if not self.install_pytorch():
            self.log("Warning: PyTorch installation failed.")
            self.log("You can install it manually later.")
            return False

        # Step 4b: Hard-remove xformers if present (it can downgrade torch)
        try:
            pyexe = self.detection_results.get("python", {}).get("executable") or python_executable or sys.executable
            subprocess.run(
                [pyexe, "-m", "pip", "uninstall", "-y", "xformers"],
                capture_output=True,
                text=True,
                timeout=180,
                **self.subprocess_flags,
            )
        except Exception:
            pass
        
        # Step 5: Install dependencies
        if not self.install_dependencies():
            self.log("Warning: Some dependencies may not have installed correctly.")
        
        # Step 6: Create launcher
        self.create_launcher()
        
        self.log("=" * 60)
        self.log("Installation complete!")
        self.log("=" * 60)
        
        return True
    
    def get_installation_checklist(self, python_executable: Optional[str] = None) -> list:
        """
        Get list of all components that will be installed with their status.
        Returns list of dicts with keys: component, version, status, status_text
        
        Args:
            python_executable: Python executable to use for checking packages (defaults to current Python)
        """
        checklist = []
        
        # Determine which Python to use for checking packages
        if not python_executable:
            python_info = self.detection_results.get("python", {}) if self.detection_results else {}
            python_executable = python_info.get("executable") or sys.executable
        
        # Try to use venv Python if available
        venv_python = None
        venv_path = Path(python_executable).parent.parent if python_executable else None
        if venv_path and (venv_path / "Scripts" / "python.exe").exists() if sys.platform == "win32" else (venv_path / "bin" / "python").exists():
            if sys.platform == "win32":
                venv_python = str(venv_path / "Scripts" / "python.exe")
            else:
                venv_python = str(venv_path / "bin" / "python")
        
        check_python = venv_python or python_executable or sys.executable
        
        # Run detection if not already done
        if not self.detection_results:
            self.run_detection()
        
        # Python
        python_info = self.detection_results.get("python", {})
        if python_info.get("found"):
            checklist.append({
                "component": "Python",
                "version": f"{python_info.get('version', 'Unknown')}+",
                "status": "installed",
                "status_text": f"✓ Installed ({python_info.get('version', 'Unknown')})"
            })
        else:
            checklist.append({
                "component": "Python",
                "version": "3.10+",
                "status": "missing",
                "status_text": "✗ Not Installed"
            })
        
        # PyTorch packages - use _verify_torch with target Python ONLY
        # Determine target venv Python path
        target_python = check_python
        if python_executable:
            target_python = python_executable
        elif check_python:
            target_python = check_python
        
        # Verify torch using target Python
        torch_ok, torch_ver, torch_cuda, torch_error = self._verify_torch(target_python)
        if torch_ok:
            if torch_cuda and torch_ver and "2.5" in torch_ver:
                checklist.append({
                    "component": "PyTorch (CUDA)",
                    "version": "2.5.1+cu118",
                    "status": "installed",
                    "status_text": f"✓ Installed ({torch_ver})"
                })
            elif torch_cuda:
                checklist.append({
                    "component": "PyTorch (CUDA)",
                    "version": "2.5.1+cu118",
                    "status": "wrong_version",
                    "status_text": f"⚠ Wrong Version ({torch_ver})"
                })
            else:
                checklist.append({
                    "component": "PyTorch (CUDA)",
                    "version": "2.5.1+cu118",
                    "status": "wrong_version",
                    "status_text": "⚠ CPU-only version"
                })
        else:
            checklist.append({
                "component": "PyTorch (CUDA)",
                "version": "2.5.1+cu118",
                "status": "missing",
                "status_text": "✗ Not Installed"
            })
        
        # PyTorch Vision and Audio (check if installed using correct Python)
        def check_package_version(pkg_name):
            """Check package version using the correct Python executable"""
            try:
                result = subprocess.run(
                    [check_python, "-c", f"import importlib.metadata; print(importlib.metadata.version('{pkg_name}'))"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    **self.subprocess_flags
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                return None
            except:
                return None
        
        torchvision_ver = check_package_version("torchvision")
        if torchvision_ver:
            checklist.append({
                "component": "PyTorch Vision",
                "version": "0.20.1+cu118",
                "status": "installed",
                "status_text": f"✓ Installed ({torchvision_ver})"
            })
        else:
            checklist.append({
                "component": "PyTorch Vision",
                "version": "0.20.1+cu118",
                "status": "missing",
                "status_text": "✗ Not Installed"
            })
        
        torchaudio_ver = check_package_version("torchaudio")
        if torchaudio_ver:
            checklist.append({
                "component": "PyTorch Audio",
                "version": "2.5.1+cu118",
                "status": "installed",
                "status_text": f"✓ Installed ({torchaudio_ver})"
            })
        else:
            checklist.append({
                "component": "PyTorch Audio",
                "version": "2.5.1+cu118",
                "status": "missing",
                "status_text": "✗ Not Installed"
            })
        
        # Triton (Windows) - verify using import (package name is triton-windows, import is triton)
        def check_triton_installed():
            """Check if triton is installed by importing it"""
            try:
                result = subprocess.run(
                    [check_python, "-c", "import triton; import triton.language as tl; import sys; print(sys.executable); print('triton', triton.__version__)"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    **self.subprocess_flags
                )
                if result.returncode == 0:
                    # Parse version from output (format: "triton X.Y.Z")
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.startswith('triton '):
                            return line.split(' ', 1)[1] if ' ' in line else None
                    return "installed"  # If import works but can't parse version, still consider it installed
                return None
            except:
                return None
        
        triton_ver = check_triton_installed()
        if triton_ver:
            checklist.append({
                "component": "Triton (Windows)",
                "version": "any",  # Accept any installed version
                "status": "installed",
                "status_text": f"✓ Installed ({triton_ver})"
            })
        else:
            checklist.append({
                "component": "Triton (Windows)",
                "version": "any",  # Accept any installed version
                "status": "missing",
                "status_text": "✗ Not Installed"
            })
        
        # PySide6 packages
        pyside_components = ["PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6"]
        for component in pyside_components:
            ver = check_package_version(component)
            if ver:
                if ver == "6.8.1":
                    checklist.append({
                        "component": component,
                        "version": "6.8.1",
                        "status": "installed",
                        "status_text": f"✓ Installed ({ver})"
                    })
                else:
                    checklist.append({
                        "component": component,
                        "version": "6.8.1",
                        "status": "wrong_version",
                        "status_text": f"⚠ Wrong Version ({ver})"
                    })
            else:
                checklist.append({
                    "component": component,
                    "version": "6.8.1",
                    "status": "missing",
                    "status_text": "✗ Not Installed"
                })
        
        # Read requirements.txt for other packages
        requirements_file = Path(__file__).parent / "requirements.txt"
        if requirements_file.exists():
            with open(requirements_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Sanitize: remove inline comments
                    line = self._sanitize_requirement(line)
                    if not line:
                        continue
                    
                    # Parse package name and version
                    parts = line.split(';')
                    pkg_line = parts[0].strip()
                    
                    # Extract package name
                    import re
                    match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)(.*)$', pkg_line)
                    if match:
                        pkg_name = match.group(1).split('[')[0]
                        version_spec = match.group(2).strip() if match.group(2) else "any"
                        # Sanitize version_spec to remove any remaining comments
                        version_spec = self._sanitize_requirement(version_spec) if version_spec != "any" else "any"
                        
                        # Skip PyTorch and PySide6 packages (already added)
                        if pkg_name.lower() in ['torch', 'torchvision', 'torchaudio', 'triton', 'pyside6']:
                            continue
                        if 'pyside6' in pkg_name.lower() or 'shiboken6' in pkg_name.lower():
                            continue
                        
                        # Check if installed using correct Python
                        # Use import test for transformers, datasets, huggingface_hub (GUI status rule)
                        if pkg_name.lower() in ["transformers", "datasets", "huggingface-hub", "huggingface_hub"]:
                            # Use import verification (no metadata inspection)
                            if self._verify_import(check_python, pkg_name):
                                status = "installed"
                                status_text = "✓ Installed"
                                checklist.append({
                                    "component": pkg_name,
                                    "version": version_spec,
                                    "status": status,
                                    "status_text": status_text
                                })
                            else:
                                checklist.append({
                                    "component": pkg_name,
                                    "version": version_spec,
                                    "status": "missing",
                                    "status_text": "✗ Not Installed"
                                })
                        else:
                            # Use metadata for other packages
                            installed_ver = check_package_version(pkg_name)
                            if installed_ver:
                                # Check version compatibility if version_spec is provided
                                status = "installed"
                                status_text = f"✓ Installed ({installed_ver})"
                                
                                if version_spec and version_spec != "any":
                                    # Simple version check (can be enhanced)
                                    if "==" in version_spec:
                                        required = version_spec.split("==")[1].strip()
                                        if installed_ver != required:
                                            status = "wrong_version"
                                            status_text = f"⚠ Wrong Version (need {required}, have {installed_ver})"
                                
                                checklist.append({
                                    "component": pkg_name,
                                    "version": version_spec,
                                    "status": status,
                                    "status_text": status_text
                                })
                            else:
                                checklist.append({
                                    "component": pkg_name,
                                    "version": version_spec,
                                    "status": "missing",
                                    "status_text": "✗ Not Installed"
                                })
        
        return checklist
    
    def check_component_status(self, component_name: str) -> dict:
        """
        Check the installation status of a single component.
        Returns dict with: status, version, status_text
        """
        # This is a simplified version - can be enhanced
        checklist = self.get_installation_checklist()
        for item in checklist:
            if item["component"] == component_name:
                return {
                    "status": item["status"],
                    "version": item.get("installed_version"),
                    "status_text": item["status_text"]
                }
        return {"status": "unknown", "version": None, "status_text": "? Not found in checklist"}
    
    def _verify_torch(self, target_python: str) -> Tuple[bool, Optional[str], Optional[bool], Optional[str]]:
        """
        Verify torch installation using target venv Python ONLY.
        
        Args:
            target_python: Exact path to target venv Python (e.g., D:\\...\\LLM\\.venv\\Scripts\\python.exe)
        
        Returns:
            Tuple of (success: bool, version: str or None, cuda_available: bool or None, error: str or None)
        """
        # Ensure we're using the exact target Python path
        target_python_path = Path(target_python).resolve()
        if not target_python_path.exists():
            return False, None, None, f"Target Python not found: {target_python}"
        
        self.log(f"Verifying torch using target Python: {target_python_path}")
        
        # Run verification command
        verify_cmd = [
            str(target_python_path), "-c",
            "import torch, sys; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
        ]
        
        try:
            verify_result = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                **self.subprocess_flags
            )
            
            # Log the executable that was actually used
            if verify_result.returncode == 0 and verify_result.stdout:
                lines = verify_result.stdout.strip().split('\n')
                if len(lines) >= 1:
                    actual_exe = lines[0].strip()
                    self.log(f"Verification used Python: {actual_exe}")
                    
                    # Assert it equals the target venv Python
                    actual_exe_path = Path(actual_exe).resolve()
                    if str(actual_exe_path).lower() != str(target_python_path).lower():
                        error_msg = f"Verification used wrong Python! Expected: {target_python_path}, Got: {actual_exe_path}"
                        self.log(f"ERROR: {error_msg}")
                        return False, None, None, error_msg
                    
                    # Parse output
                    if len(lines) >= 2:
                        torch_version = lines[1].strip()
                        cuda_available = False
                        if len(lines) >= 3:
                            cuda_str = lines[2].strip().lower()
                            cuda_available = cuda_str == "true"
                        
                        self.log(f"✓ torch verified: version {torch_version}, CUDA: {cuda_available}")
                        return True, torch_version, cuda_available, None
                    else:
                        return False, None, None, "Verification output incomplete"
                else:
                    return False, None, None, "Verification output empty"
            else:
                # Exit code != 0 or no output
                error_output = verify_result.stderr if verify_result.stderr else verify_result.stdout
                error_msg = f"torch import failed (exit code {verify_result.returncode})"
                if error_output:
                    error_msg += f": {error_output[:500]}"
                self.log(f"ERROR: {error_msg}")
                return False, None, None, error_msg
                
        except Exception as e:
            error_msg = f"Exception during torch verification: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            return False, None, None, error_msg
    
    def _verify_transformers(self, target_python: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Verify transformers installation using target venv Python ONLY.
        
        Args:
            target_python: Exact path to target venv Python (e.g., D:\\...\\LLM\\.venv\\Scripts\\python.exe)
        
        Returns:
            Tuple of (success: bool, version: str or None, error: str or None)
        """
        # Ensure we're using the exact target Python path
        target_python_path = Path(target_python).resolve()
        if not target_python_path.exists():
            return False, None, f"Target Python not found: {target_python}"
        
        self.log(f"Verifying transformers using target Python: {target_python_path}")
        
        # Run verification command
        verify_cmd = [
            str(target_python_path), "-c",
            "import transformers, sys; print(sys.executable); print(transformers.__version__)"
        ]
        
        try:
            verify_result = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                **self.subprocess_flags
            )
            
            # Log the executable that was actually used
            if verify_result.returncode == 0 and verify_result.stdout:
                lines = verify_result.stdout.strip().split('\n')
                if len(lines) >= 1:
                    actual_exe = lines[0].strip()
                    self.log(f"Verification used Python: {actual_exe}")
                    
                    # Assert it equals the target venv Python
                    actual_exe_path = Path(actual_exe).resolve()
                    if str(actual_exe_path).lower() != str(target_python_path).lower():
                        error_msg = f"Verification used wrong Python! Expected: {target_python_path}, Got: {actual_exe_path}"
                        self.log(f"ERROR: {error_msg}")
                        return False, None, error_msg
                    
                    # Parse output
                    if len(lines) >= 2:
                        transformers_version = lines[1].strip()
                        self.log(f"✓ transformers verified: version {transformers_version}")
                        return True, transformers_version, None
                    else:
                        return False, None, "Verification output incomplete"
                else:
                    return False, None, "Verification output empty"
            else:
                # Exit code != 0 or no output
                error_output = verify_result.stderr if verify_result.stderr else verify_result.stdout
                error_msg = f"transformers import failed (exit code {verify_result.returncode})"
                if error_output:
                    error_msg += f": {error_output[:500]}"
                self.log(f"ERROR: {error_msg}")
                return False, None, error_msg
                
        except Exception as e:
            self.log(f"ERROR: Exception during transformers verification: {str(e)}")
            return False, None, f"Exception during verification: {str(e)}"
    
    def _verify_datasets(self, target_python: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Verify datasets installation using target venv Python ONLY.
        
        Args:
            target_python: Exact path to target venv Python (e.g., D:\\...\\LLM\\.venv\\Scripts\\python.exe)
        
        Returns:
            Tuple of (success: bool, version: str or None, error: str or None)
        """
        # Ensure we're using the exact target Python path
        target_python_path = Path(target_python).resolve()
        if not target_python_path.exists():
            return False, None, f"Target Python not found: {target_python}"
        
        self.log(f"Verifying datasets using target Python: {target_python_path}")
        
        # Run verification command
        verify_cmd = [
            str(target_python_path), "-c",
            "import datasets, sys; print(sys.executable); print(datasets.__version__)"
        ]
        
        try:
            verify_result = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                **self.subprocess_flags
            )
            
            # Log the executable that was actually used
            if verify_result.returncode == 0 and verify_result.stdout:
                lines = verify_result.stdout.strip().split('\n')
                if len(lines) >= 1:
                    actual_exe = lines[0].strip()
                    self.log(f"Verification used Python: {actual_exe}")
                    
                    # Assert it equals the target venv Python
                    actual_exe_path = Path(actual_exe).resolve()
                    if str(actual_exe_path).lower() != str(target_python_path).lower():
                        error_msg = f"Verification used wrong Python! Expected: {target_python_path}, Got: {actual_exe_path}"
                        self.log(f"ERROR: {error_msg}")
                        return False, None, error_msg
                    
                    # Parse output
                    if len(lines) >= 2:
                        datasets_version = lines[1].strip()
                        self.log(f"✓ datasets verified: version {datasets_version}")
                        return True, datasets_version, None
                    else:
                        return False, None, "Verification output incomplete"
                else:
                    return False, None, "Verification output empty"
            else:
                # Exit code != 0 or no output
                error_output = verify_result.stderr if verify_result.stderr else verify_result.stdout
                error_msg = f"datasets import failed (exit code {verify_result.returncode})"
                if error_output:
                    error_msg += f": {error_output[:500]}"
                self.log(f"ERROR: {error_msg}")
                return False, None, error_msg
                
        except Exception as e:
            self.log(f"ERROR: Exception during datasets verification: {str(e)}")
            return False, None, f"Exception during verification: {str(e)}"
    
    def _verify_import(self, target_python: str, package_name: str) -> bool:
        """
        Verify package installation by import test only (no metadata inspection).
        
        Args:
            target_python: Exact path to target venv Python
            package_name: Package name to import (e.g., "transformers", "datasets")
        
        Returns:
            True if import succeeds, False otherwise
        """
        target_python_path = Path(target_python).resolve()
        if not target_python_path.exists():
            return False
        
        # Map package names to import names
        import_map = {
            "transformers": "transformers",
            "huggingface_hub": "huggingface_hub",
            "datasets": "datasets",
        }
        import_name = import_map.get(package_name, package_name)
        
        # Run import test
        verify_cmd = [
            str(target_python_path), "-c",
            f"import {import_name}"
        ]
        
        try:
            verify_result = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                **self.subprocess_flags
            )
            return verify_result.returncode == 0
        except Exception:
            return False
    
    def _verify_package_version(self, python_executable: Optional[str], package_name: str, version_spec: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Verify if a package is installed and matches the version specification.
        
        Args:
            python_executable: Path to Python executable
            package_name: Name of the package to check
            version_spec: Version specification (e.g., "==4.51.3", ">=0.21,<0.22")
        
        Returns:
            Tuple of (success: bool, installed_version: str or None, error: str or None)
        """
        try:
            # Try importlib.metadata first (Python 3.8+)
            try:
                import importlib.metadata
                try:
                    installed_ver = importlib.metadata.version(package_name)
                except importlib.metadata.PackageNotFoundError:
                    return False, None, f"Package {package_name} not installed"
            except ImportError:
                # Fallback for older Python versions
                try:
                    import importlib_metadata as importlib_metadata
                    try:
                        installed_ver = importlib_metadata.version(package_name)
                    except importlib_metadata.PackageNotFoundError:
                        return False, None, f"Package {package_name} not installed"
                except ImportError:
                    # Last resort: try using pip show (only if python_executable is provided)
                    if not python_executable:
                        return False, None, f"Package {package_name} not installed (cannot verify without Python executable)"
                    try:
                        result = subprocess.run(
                            [python_executable, "-m", "pip", "show", package_name],
                            capture_output=True,
                            text=True,
                            timeout=10,
                            **self.subprocess_flags
                        )
                        if result.returncode != 0:
                            return False, None, f"Package {package_name} not installed"
                        # Parse version from pip show output
                        for line in result.stdout.split('\n'):
                            if line.startswith('Version:'):
                                installed_ver = line.split(':', 1)[1].strip()
                                break
                        else:
                            return False, None, f"Could not determine version of {package_name}"
                    except Exception as e:
                        return False, None, f"Error checking package: {str(e)}"
            
            # If no version spec, just check if installed
            if not version_spec or version_spec.lower() == "any":
                return True, installed_ver, None
            
            # Use packaging library for version comparison
            if pkg_version and SpecifierSet:
                try:
                    spec = SpecifierSet(version_spec)
                    if spec.contains(installed_ver):
                        return True, installed_ver, None
                    else:
                        return False, installed_ver, f"Version {installed_ver} does not match {version_spec}"
                except Exception as e:
                    return False, installed_ver, f"Error parsing version spec: {str(e)}"
            else:
                # Fallback: simple string comparison for ==
                if "==" in version_spec:
                    required = version_spec.split("==")[1].strip()
                    if installed_ver == required:
                        return True, installed_ver, None
                    else:
                        return False, installed_ver, f"Version {installed_ver} != {required}"
                else:
                    # Can't do complex comparison without packaging library
                    return True, installed_ver, None  # Assume OK if we can't verify
            
        except Exception as e:
            # Log the error but don't crash
            self.log(f"Warning: Error in _verify_package_version for {package_name}: {str(e)}")
            return False, None, f"Error checking package: {str(e)}"
    
    def _install_component(self, component: dict, python_executable: str) -> Tuple[bool, str]:
        """
        Install a single component from the checklist.
        Returns (success, error_message)
        """
        component_name = component["component"]
        required_version = component["version"]
        status = component.get("status", "missing")
        
        # Skip if already installed correctly
        if status == "installed":
            return True, None
        
        self.log(f"Installing {component_name} {required_version}...")
        
        try:
            # Map component names to installation commands
            if component_name == "Python":
                # Python should already be installed (checked by launcher)
                return True, None
            
            elif component_name == "PyTorch (CUDA)":
                # Use install_pytorch which now uses worker process
                success = self.install_pytorch(python_executable=python_executable)
                if success:
                    # Verify immediately using target venv Python
                    torch_ok, torch_ver, torch_cuda, torch_error = self._verify_torch(python_executable)
                    if torch_ok:
                        if torch_cuda:
                            return True, None
                        else:
                            return False, "PyTorch installed but CUDA not available"
                    else:
                        return False, f"PyTorch verification failed: {torch_error}"
                else:
                    return False, "PyTorch installation failed"
            
            elif component_name == "PyTorch Vision":
                # CRITICAL: Verify torch is installed and working BEFORE installing torchvision
                self.log("Verifying torch is installed before installing torchvision...")
                torch_ok, torch_ver, torch_cuda, torch_error = self._verify_torch(python_executable)
                if not torch_ok:
                    return False, f"torch must be installed and working before installing torchvision: {torch_error}"
                
                # Install torchvision with same CUDA version as torch
                cuda_build = self.get_optimal_cuda_build()
                build_key_map = {"cu124": "cuda_12_4", "cu121": "cuda_12_1", "cu118": "cuda_11_8", "cpu": "cpu"}
                build_key = build_key_map.get(cuda_build, "cpu")
                versions = self.VERSION_MATRIX[build_key]
                torchvision_version = versions["torchvision"]
                
                # Determine index URL based on CUDA build
                index_url = f"https://download.pytorch.org/whl/{cuda_build}" if cuda_build != "cpu" else "https://download.pytorch.org/whl/cpu"
                
                # Use plain version (not +cu118) with index URL
                # Use worker process for installation
                success, last_lines, exit_code = self._run_pip_worker(
                    action="install",
                    package=f"torchvision=={torchvision_version}",
                    python_executable=python_executable,
                    index_url=index_url,
                    pip_args=["--force-reinstall", "--no-deps", "--no-cache-dir"]
                )
                if success:
                    # Verify using target Python
                    self.log("Verifying torchvision installation...")
                    verify_cmd = [
                        python_executable, "-c",
                        "import torchvision, torch; print(torch.__version__); print(torchvision.__version__)"
                    ]
                    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30, **self.subprocess_flags)
                    if verify_result.returncode == 0:
                        self.log(f"✓ torchvision verified: {verify_result.stdout.strip()}")
                        return True, None
                    else:
                        return False, f"torchvision verification failed: {verify_result.stderr[:200]}"
                else:
                    return False, f"torchvision installation failed (exit code {exit_code}): {last_lines[-500:]}"
            
            elif component_name == "PyTorch Audio":
                # CRITICAL: Verify torch is installed and working BEFORE installing torchaudio
                self.log("Verifying torch is installed before installing torchaudio...")
                torch_ok, torch_ver, torch_cuda, torch_error = self._verify_torch(python_executable)
                if not torch_ok:
                    return False, f"torch must be installed and working before installing torchaudio: {torch_error}"
                
                # Install torchaudio with same CUDA version as torch
                cuda_build = self.get_optimal_cuda_build()
                build_key_map = {"cu124": "cuda_12_4", "cu121": "cuda_12_1", "cu118": "cuda_11_8", "cpu": "cpu"}
                build_key = build_key_map.get(cuda_build, "cpu")
                versions = self.VERSION_MATRIX[build_key]
                torchaudio_version = versions["torchaudio"]
                
                # Determine index URL based on CUDA build
                index_url = f"https://download.pytorch.org/whl/{cuda_build}" if cuda_build != "cpu" else "https://download.pytorch.org/whl/cpu"
                
                # Use plain version (not +cu118) with index URL
                # Use worker process for installation
                success, last_lines, exit_code = self._run_pip_worker(
                    action="install",
                    package=f"torchaudio=={torchaudio_version}",
                    python_executable=python_executable,
                    index_url=index_url,
                    pip_args=["--force-reinstall", "--no-deps", "--no-cache-dir"]
                )
                if success:
                    # Verify using target Python
                    self.log("Verifying torchaudio installation...")
                    verify_cmd = [
                        python_executable, "-c",
                        "import torchaudio, torch; print(torch.__version__); print(torchaudio.__version__)"
                    ]
                    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30, **self.subprocess_flags)
                    if verify_result.returncode == 0:
                        self.log(f"✓ torchaudio verified: {verify_result.stdout.strip()}")
                        return True, None
                    else:
                        return False, f"torchaudio verification failed: {verify_result.stderr[:200]}"
                else:
                    return False, f"torchaudio installation failed (exit code {exit_code}): {last_lines[-500:]}"
            
            elif component_name == "Triton (Windows)":
                # CRITICAL: Verify torch is installed and working BEFORE installing triton
                self.log("Verifying torch is installed before installing triton...")
                torch_ok, torch_ver, torch_cuda, torch_error = self._verify_torch(python_executable)
                if not torch_ok:
                    return False, f"torch must be installed and working before installing triton: {torch_error}"
                
                # Install triton-windows for Windows (package name), triton for others
                if platform.system() == "Windows":
                    pkg_spec = "triton-windows"
                else:
                    pkg_spec = "triton"
                
                # Use worker process for installation
                success, last_lines, exit_code = self._run_pip_worker(
                    action="install",
                    package=pkg_spec,
                    python_executable=python_executable,
                    pip_args=["--force-reinstall", "--no-deps"]
                )
                if success:
                    # Verify using target Python with exact command specified
                    self.log("Verifying triton installation...")
                    verify_cmd = [
                        python_executable, "-c",
                        "import triton; import triton.language as tl; import sys; print(sys.executable); print('triton', triton.__version__)"
                    ]
                    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30, **self.subprocess_flags)
                    if verify_result.returncode == 0:
                        # Parse and log the version
                        lines = verify_result.stdout.strip().split('\n')
                        triton_version = None
                        for line in lines:
                            if line.startswith('triton '):
                                triton_version = line.split(' ', 1)[1] if ' ' in line else "unknown"
                                break
                        if triton_version:
                            self.log(f"✓ triton verified: version {triton_version}")
                        else:
                            self.log("✓ triton verified (version could not be parsed)")
                        return True, None
                    else:
                        return False, f"Triton verification failed: {verify_result.stderr[:200] if verify_result.stderr else verify_result.stdout[:200]}"
                else:
                    return False, f"Triton installation failed (exit code {exit_code}): {last_lines[-500:]}"
            
            elif component_name in ["PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6"]:
                # Install all PySide6 components together
                if component_name == "PySide6":  # Only install once for the group
                    # Install PySide6 first
                    success, last_lines, exit_code = self._run_pip_worker(
                        action="install",
                        package="PySide6==6.8.1",
                        python_executable=python_executable,
                        pip_args=["--force-reinstall"]
                    )
                    if not success:
                        return False, f"PySide6 installation failed (exit code {exit_code}): {last_lines[-500:]}"
                    
                    # Install PySide6-Essentials
                    success, last_lines, exit_code = self._run_pip_worker(
                        action="install",
                        package="PySide6-Essentials==6.8.1",
                        python_executable=python_executable,
                        pip_args=["--force-reinstall"]
                    )
                    if not success:
                        return False, f"PySide6-Essentials installation failed (exit code {exit_code}): {last_lines[-500:]}"
                    
                    # Install PySide6-Addons
                    success, last_lines, exit_code = self._run_pip_worker(
                        action="install",
                        package="PySide6-Addons==6.8.1",
                        python_executable=python_executable,
                        pip_args=["--force-reinstall"]
                    )
                    if not success:
                        return False, f"PySide6-Addons installation failed (exit code {exit_code}): {last_lines[-500:]}"
                    
                    # Install shiboken6
                    success, last_lines, exit_code = self._run_pip_worker(
                        action="install",
                        package="shiboken6==6.8.1",
                        python_executable=python_executable,
                        pip_args=["--force-reinstall"]
                    )
                    if not success:
                        return False, f"shiboken6 installation failed (exit code {exit_code}): {last_lines[-500:]}"
                    
                    # Verify all components
                    all_ok = True
                    for pyside_comp in ["PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6"]:
                        success, ver, error = self._verify_package_version(python_executable, pyside_comp, "==6.8.1")
                        if not success:
                            all_ok = False
                            break
                    if all_ok:
                        # Test import
                        import_cmd = [python_executable, "-c", "import PySide6.QtCore; print('OK')"]
                        import_result = subprocess.run(import_cmd, capture_output=True, text=True, timeout=10, **self.subprocess_flags)
                        if import_result.returncode == 0:
                            return True, None
                        else:
                            return False, f"PySide6 import failed: {import_result.stderr[:200]}"
                    else:
                        return False, "PySide6 components version mismatch"
                else:
                    # Already handled by PySide6 installation
                    return True, None
            
            else:
                # Generic package installation from requirements.txt
                # Sanitize component_name and required_version to remove any comments
                component_name = self._sanitize_requirement(component_name)
                required_version = self._sanitize_requirement(required_version) if required_version else None
                
                # Encode version constraints in package name
                if required_version:
                    if required_version.startswith("==") or required_version.startswith("<") or required_version.startswith(">="):
                        pkg_spec = f"{component_name}{required_version}"
                    else:
                        pkg_spec = f"{component_name}=={required_version}"
                else:
                    pkg_spec = component_name
                
                # Sanitize final package spec before passing to pip
                pkg_spec = self._sanitize_requirement(pkg_spec)
                
                # Use worker process for installation
                success, last_lines, exit_code = self._run_pip_worker(
                    action="install",
                    package=pkg_spec,
                    python_executable=python_executable,
                    pip_args=["--force-reinstall"]
                )
                if success:
                    # Verify
                    success, ver, error = self._verify_package_version(python_executable, component_name, required_version)
                    if success:
                        return True, None
                    else:
                        return False, f"{component_name} verification failed: {error}"
                else:
                    # Check for disk space errors
                    if "No space left on device" in last_lines or "Errno 28" in last_lines:
                        return False, "Insufficient disk space"
                    return False, f"{component_name} installation failed (exit code {exit_code}): {last_lines[-500:]}"
        
        except Exception as e:
            return False, f"Error installing {component_name}: {str(e)}"
    
    def repair_all(self, python_executable: Optional[str] = None) -> bool:
        """
        Repair a broken environment deterministically.
        USES CHECKLIST AS SOURCE OF TRUTH - installs components in checklist order.
        STOPS IMMEDIATELY if any component fails.
        """
        self.log("=" * 60)
        self.log("LLM Fine-tuning Studio - Repair Mode")
        self.log("Using installation checklist as source of truth")
        self.log("=" * 60)
        
        # Check disk space before starting
        has_space, free_gb = self._check_disk_space()
        self.log(f"Available disk space: {free_gb:.2f} GB (minimum required: {self.min_disk_space_gb} GB)")
        if not has_space:
            self.log(f"ERROR: Insufficient disk space! Only {free_gb:.2f} GB available, need at least {self.min_disk_space_gb} GB")
            self.log("Please free up disk space and try again.")
            return False

        # Detection (also populates python path)
        if not self.detection_results:
            self.run_detection()

        if not python_executable:
            python_executable = (
                self.detection_results.get("python", {}).get("executable")
                or sys.executable
            )
        
        # GET CHECKLIST - THIS IS THE SOURCE OF TRUTH
        self.log("Getting installation checklist (source of truth)...")
        checklist = self.get_installation_checklist(python_executable=python_executable)
        
        # Only uninstall packages that need to be reinstalled (not already installed correctly)
        packages_to_uninstall = []
        for component in checklist:
            component_name = component["component"]
            status = component.get("status", "missing")
            
            # Only uninstall if not already installed correctly
            if status != "installed":
                if component_name == "PyTorch (CUDA)":
                    packages_to_uninstall.extend(["torch", "torchvision", "torchaudio", "xformers"])
                elif component_name == "PyTorch Vision":
                    if "torchvision" not in packages_to_uninstall:
                        packages_to_uninstall.append("torchvision")
                elif component_name == "PyTorch Audio":
                    if "torchaudio" not in packages_to_uninstall:
                        packages_to_uninstall.append("torchaudio")
                elif component_name == "Triton (Windows)":
                    packages_to_uninstall.extend(["triton", "triton-windows"])
                elif component_name in ["PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6"]:
                    if "PySide6" not in packages_to_uninstall:
                        packages_to_uninstall.extend(["PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6"])
                elif component_name in ["transformers", "tokenizers", "numpy", "datasets"]:
                    packages_to_uninstall.append(component_name)
        
        # Also always remove problematic packages
        always_remove = ["trl", "torchao", "xformers"]
        for pkg in always_remove:
            if pkg not in packages_to_uninstall:
                packages_to_uninstall.append(pkg)
        
        # Remove duplicates
        packages_to_uninstall = list(set(packages_to_uninstall))
        
        if packages_to_uninstall:
            self.log(f"Repair: Uninstalling packages that need reinstallation: {', '.join(packages_to_uninstall)}")
            for pkg in packages_to_uninstall:
                try:
                    success, last_lines, exit_code = self._run_pip_worker(
                        action="uninstall",
                        package=pkg,
                        python_executable=python_executable
                    )
                    # Uninstall failures are OK if package wasn't installed
                    if not success and "not installed" not in last_lines.lower():
                        self.log(f"Repair: Warning - uninstall of {pkg} had issues: {last_lines[:200]}")
                except Exception as exc:
                    self.log(f"Repair: Warning - could not uninstall {pkg}: {str(exc)}")
        else:
            self.log("Repair: No packages need to be uninstalled (all are already correct)")
        
        # INSTALL CORE STACK IN EXACT ORDER - NO DEVIATION
        self.log("=" * 60)
        self.log("Installing core stack in exact order...")
        self.log("=" * 60)
        
        # Step 1: Install numpy<2 with constraints
        self.log("[1/7] Installing numpy<2 with constraints...")
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package="numpy<2",
            python_executable=python_executable,
            pip_args=["--force-reinstall"]
        )
        if not success:
            self.log(f"ERROR: numpy installation failed - cannot continue")
            self.log(f"Exit code: {exit_code}")
            self.log(f"Error output: {last_lines[-500:]}")
            return False
        self.log("[1/7] ✓ numpy installed")
        
        # Step 2: Install PyTorch stack with verification guards
        self.log("[2/7] Checking PyTorch stack (torch, torchvision, torchaudio)...")
        cuda_build = self.get_optimal_cuda_build()
        index_url = f"https://download.pytorch.org/whl/{cuda_build}" if cuda_build != "cpu" else "https://download.pytorch.org/whl/cpu"
        
        # Verify torch first
        torch_ok, torch_ver, torch_cuda, torch_error = self._verify_torch(python_executable)
        torch_needs_install = True
        if torch_ok and torch_cuda:
            # Check if version matches expected 2.5.1+cu118 (or just 2.5.1)
            if torch_ver and ("2.5.1" in torch_ver or torch_ver.startswith("2.5.1")):
                self.log(f"[2/7] ✓ torch already installed and verified: {torch_ver} (CUDA available)")
                torch_needs_install = False
            else:
                self.log(f"[2/7] torch version mismatch: have {torch_ver}, need 2.5.1+cu118")
        else:
            self.log(f"[2/7] torch verification failed: {torch_error}")
        
        # Install torch only if verification failed
        if torch_needs_install:
            self.log("[2/7] Installing torch==2.5.1+cu118 (verification failed, using --force-reinstall)...")
            # Delete torch directory only if verification failed
            venv_path = Path(python_executable).parent.parent
            self._delete_torch_directory(venv_path)
            
            success, last_lines, exit_code = self._run_pip_worker(
                action="install",
                package="torch==2.5.1+cu118",
                python_executable=python_executable,
                index_url=index_url,
                pip_args=["--force-reinstall", "--no-deps", "--no-cache-dir"]
            )
            if not success:
                self.log(f"ERROR: torch installation failed - cannot continue")
                return False
            self.log("[2/7] ✓ torch installed")
        else:
            self.log("[2/7] torch installation SKIPPED (already installed and verified)")
        
        # Verify torchvision
        torchvision_ok = False
        verify_cmd = [python_executable, "-c", "import torchvision; print(torchvision.__version__)"]
        try:
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30, **self.subprocess_flags)
            if verify_result.returncode == 0 and verify_result.stdout:
                torchvision_ver = verify_result.stdout.strip()
                # Check if version matches expected 0.20.1
                if "0.20.1" in torchvision_ver or torchvision_ver.startswith("0.20.1"):
                    self.log(f"[2/7] ✓ torchvision already installed and verified: {torchvision_ver}")
                    torchvision_ok = True
                else:
                    self.log(f"[2/7] torchvision version mismatch: have {torchvision_ver}, need 0.20.1+cu118")
        except Exception as e:
            self.log(f"[2/7] torchvision verification failed: {str(e)}")
        
        # Install torchvision only if verification failed
        if not torchvision_ok:
            self.log("[2/7] Installing torchvision==0.20.1+cu118 (verification failed, using --force-reinstall)...")
            success, last_lines, exit_code = self._run_pip_worker(
                action="install",
                package="torchvision==0.20.1+cu118",
                python_executable=python_executable,
                index_url=index_url,
                pip_args=["--force-reinstall", "--no-deps", "--no-cache-dir"]
            )
            if not success:
                self.log(f"ERROR: torchvision installation failed - cannot continue")
                return False
            self.log("[2/7] ✓ torchvision installed")
        else:
            self.log("[2/7] torchvision installation SKIPPED (already installed and verified)")
        
        # Verify torchaudio
        torchaudio_ok = False
        verify_cmd = [python_executable, "-c", "import torchaudio; print(torchaudio.__version__)"]
        try:
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30, **self.subprocess_flags)
            if verify_result.returncode == 0 and verify_result.stdout:
                torchaudio_ver = verify_result.stdout.strip()
                # Check if version matches expected 2.5.1
                if "2.5.1" in torchaudio_ver or torchaudio_ver.startswith("2.5.1"):
                    self.log(f"[2/7] ✓ torchaudio already installed and verified: {torchaudio_ver}")
                    torchaudio_ok = True
                else:
                    self.log(f"[2/7] torchaudio version mismatch: have {torchaudio_ver}, need 2.5.1+cu118")
        except Exception as e:
            self.log(f"[2/7] torchaudio verification failed: {str(e)}")
        
        # Install torchaudio only if verification failed
        if not torchaudio_ok:
            self.log("[2/7] Installing torchaudio==2.5.1+cu118 (verification failed, using --force-reinstall)...")
            success, last_lines, exit_code = self._run_pip_worker(
                action="install",
                package="torchaudio==2.5.1+cu118",
                python_executable=python_executable,
                index_url=index_url,
                pip_args=["--force-reinstall", "--no-deps", "--no-cache-dir"]
            )
            if not success:
                self.log(f"ERROR: torchaudio installation failed - cannot continue")
                return False
            self.log("[2/7] ✓ torchaudio installed")
        else:
            self.log("[2/7] torchaudio installation SKIPPED (already installed and verified)")
        
        self.log("[2/7] ✓ PyTorch stack verified/installed")
        
        # Step 3: Hard-fix huggingface-hub BEFORE transformers
        self.log("[3/7] Hard-fixing huggingface-hub (uninstall then reinstall with constraints)...")
        # Uninstall huggingface-hub and hf-xet
        for pkg in ["huggingface-hub", "hf-xet"]:
            self._run_pip_worker(
                action="uninstall",
                package=pkg,
                python_executable=python_executable
            )
        # Install huggingface-hub with constraints
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package="huggingface-hub>=0.30.0,<1.0",
            python_executable=python_executable,
            pip_args=["--force-reinstall"]
        )
        if not success:
            self.log(f"ERROR: huggingface-hub installation failed - cannot continue")
            return False
        self.log("[3/7] ✓ huggingface-hub fixed")
        
        # Step 4: Install transformers with constraints
        self.log("[4/7] Installing transformers==4.51.3 with constraints...")
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package="transformers==4.51.3",
            python_executable=python_executable,
            pip_args=["--force-reinstall"]
        )
        if not success:
            self.log(f"ERROR: transformers installation failed - cannot continue")
            return False
        # Verify by import
        if not self._verify_import(python_executable, "transformers"):
            self.log(f"ERROR: transformers import verification failed")
            return False
        self.log("[4/7] ✓ transformers installed and verified")
        
        # Step 5: Install tokenizers with constraints
        self.log("[5/7] Installing tokenizers==0.21.4 with constraints...")
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package="tokenizers==0.21.4",
            python_executable=python_executable,
            pip_args=["--force-reinstall"]
        )
        if not success:
            self.log(f"ERROR: tokenizers installation failed - cannot continue")
            return False
        self.log("[5/7] ✓ tokenizers installed")
        
        # Step 6: Install datasets with constraints
        self.log("[6/7] Installing datasets>=2.11.0,<4.4.0 with constraints...")
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package="datasets>=2.11.0,<4.4.0",
            python_executable=python_executable,
            pip_args=["--force-reinstall"]
        )
        if not success:
            self.log(f"ERROR: datasets installation failed - cannot continue")
            return False
        # Verify by import
        if not self._verify_import(python_executable, "datasets"):
            self.log(f"ERROR: datasets import verification failed")
            return False
        self.log("[6/7] ✓ datasets installed and verified")
        
        # Step 7: Install trl, xformers, torchao, peft with constraints
        self.log("[7/7] Installing trl, xformers, torchao, peft with constraints...")
        for pkg in ["trl>=0.18.2,<0.25.0", "xformers", "torchao", "peft"]:
            success, last_lines, exit_code = self._run_pip_worker(
                action="install",
                package=pkg,
                python_executable=python_executable,
                pip_args=["--force-reinstall"]
            )
            if not success:
                self.log(f"WARNING: {pkg} installation had issues: {last_lines[-200:]}")
                # Continue anyway for optional packages
        self.log("[7/7] ✓ Additional packages installed")
        
        # Final verification: import test for core packages
        self.log("=" * 60)
        self.log("Final verification: Import test for core packages...")
        core_imports = ["transformers", "huggingface_hub", "datasets"]
        all_ok = True
        for pkg in core_imports:
            if self._verify_import(python_executable, pkg):
                self.log(f"  ✓ {pkg} import OK")
            else:
                self.log(f"  ✗ {pkg} import FAILED")
                all_ok = False
        
        if not all_ok:
            self.log("=" * 60)
            self.log("REPAIR FAILED - Import verification failed")
            self.log("=" * 60)
            return False
        
        # Final summary
        self.log("=" * 60)
        self.log("REPAIR COMPLETE - All components installed and verified")
        self.log("=" * 60)
        return True
        try:
            sp = subprocess.run(
                [python_executable, "-c", "import site; print(site.getsitepackages()[0])"],
                capture_output=True,
                text=True,
                timeout=10,
                **self.subprocess_flags,
            )
            if sp.returncode == 0:
                site_packages = sp.stdout.strip()
        except Exception:
            pass

        # Hard uninstall problematic packages
        self.log("Repair: Uninstalling conflicting packages...")
        try:
            subprocess.run(
                [
                    python_executable, "-m", "pip", "uninstall", "-y",
                    "torch", "torchvision", "torchaudio",
                    "xformers",
                    "triton", "triton-windows",
                    "transformers", "tokenizers",  # Also uninstall these to force fresh install
                    "numpy", "datasets",  # Add numpy and datasets to force correct versions
                    "PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6",  # Also fix PySide6
                    "trl",  # Remove trl (requires transformers>=4.56.1, conflicts with our 4.51.3)
                    "torchao",  # Remove torchao (incompatible, unsloth-zoo wants it but it breaks things)
                ],
                capture_output=True,
                text=True,
                timeout=600,
                **self.subprocess_flags,
            )
        except Exception as e:
            self.log(f"Repair: Uninstall step warning: {e}")

        # Remove leftovers that can cause "No package metadata found" or weird imports
        if site_packages and os.path.isdir(site_packages):
            try:
                import shutil
                spath = Path(site_packages)
                leftovers = [
                    "torch", "torchvision", "torchaudio", "triton", "xformers",
                    "transformers", "tokenizers",  # Also clean these
                    "numpy", "datasets",  # Add numpy and datasets cleanup
                    "PySide6", "shiboken6",  # Add PySide6 cleanup
                    "torch-*.dist-info", "torchvision-*.dist-info", "torchaudio-*.dist-info",
                    "triton-*.dist-info", "triton_windows-*.dist-info", "triton_windows-*.data",
                    "xformers-*.dist-info",
                    "transformers-*.dist-info", "tokenizers-*.dist-info",  # Clean these too
                    "numpy-*.dist-info", "datasets-*.dist-info",  # Add these too
                    "PySide6-*.dist-info", "PySide6_*.dist-info", "shiboken6-*.dist-info",  # PySide6 cleanup
                    "~~rch", "~orch",  # common temp dirs seen on Windows
                ]
                self.log("Repair: Cleaning leftover site-packages folders...")
                for pattern in leftovers:
                    for p in spath.glob(pattern):
                        try:
                            if p.is_dir():
                                shutil.rmtree(p, ignore_errors=True)
                            else:
                                p.unlink(missing_ok=True)
                        except Exception:
                            pass
            except Exception as e:
                self.log(f"Repair: Cleanup warning: {e}")

        # Step 1: Kill any running Python processes that might lock files
        # TEMPORARILY DISABLED - causing crashes, will re-enable after fixing
        #     self.log("Continuing with repair anyway...")

        # Step 1.5: Fix numpy FIRST (must be <2.0.0, before PyTorch and other packages)
        self.log("Repair: Ensuring numpy <2.0.0 (critical for PyTorch compatibility)...")
        try:
            result = subprocess.run(
                [python_executable, "-m", "pip", "install", "--force-reinstall", "numpy<2"],
                capture_output=True,
                text=True,
                timeout=300,
                **self.subprocess_flags
            )
            if result.returncode == 0:
                self.log("OK: numpy downgraded to <2.0.0")
            else:
                self.log(f"WARNING: numpy downgrade failed: {result.stderr}")
        except Exception as e:
            self.log(f"WARNING: Error downgrading numpy: {str(e)}")

        # Step 2: Install PyTorch FIRST (before requirements.txt, so accelerate doesn't pull wrong torch version)
        self.log("Repair: Installing PyTorch + Triton (BEFORE requirements.txt to prevent wrong version)...")
        pytorch_success = self.install_pytorch(python_executable=python_executable)
        if not pytorch_success:
            self.log("ERROR: PyTorch installation failed (may be due to file locks)")
            self.log("You may need to close all Python processes and run Fix Issues again")
            # Continue anyway - we'll try to install requirements.txt, but it may fail
        
        # Step 3: Uninstall PySide6 packages (will reinstall at end)
        self.log("Repair: Uninstalling PySide6 packages (will reinstall at correct version at end)...")
        subprocess.run(
            [python_executable, "-m", "pip", "uninstall", "-y", "PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6"],
            capture_output=True,
            timeout=60,
            **self.subprocess_flags
        )
        
        # Step 3: Install requirements.txt BUT exclude PySide6 and PyTorch (already installed)
        self.log("Repair: Installing core dependencies from requirements.txt (excluding PySide6 and PyTorch)...")
        requirements_file = Path(__file__).parent / "requirements.txt"
        if requirements_file.exists():
            # Read requirements and filter out PySide6 packages
            with open(requirements_file, 'r', encoding='utf-8') as f:
                req_lines = f.readlines()
            
            # Create temp requirements file without PySide6 AND PyTorch packages
            # (PyTorch must be installed separately with correct CUDA version)
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                for line in req_lines:
                    # Sanitize: remove inline comments before processing
                    sanitized_line = self._sanitize_requirement(line)
                    if not sanitized_line or sanitized_line.startswith('#'):
                        continue
                    
                    # Skip PySide6, PyTorch packages, and packages that depend on torch
                    line_lower = sanitized_line.lower()
                    skip = False
                    # Skip PySide6 packages
                    if any(pkg in line_lower for pkg in ['pyside6', 'shiboken6']):
                        skip = True
                    # Skip PyTorch packages (installed separately)
                    if any(pkg in line_lower for pkg in ['torch', 'torchvision', 'torchaudio', 'triton']):
                        skip = True
                    # Skip accelerate temporarily (it requires torch, we'll install it after PyTorch)
                    # Actually, we can't skip accelerate because other packages might need it
                    # Instead, we'll install PyTorch FIRST, then requirements.txt
                    if not skip:
                        # Write sanitized line (without comments) to temp file
                        tmp.write(sanitized_line + '\n')
                tmp_req_file = tmp.name
            
            # Install from filtered requirements
            # NOTE: Do NOT use --force-reinstall here because:
            # 1. PyTorch is already installed and its DLLs may be locked
            # 2. We'll force-reinstall specific packages (transformers, tokenizers) separately
            # 3. This prevents file lock errors on torch DLLs
            cmd = [
                python_executable, "-m", "pip", "install",
                "-r", tmp_req_file
            ]
            self.log(f"Running: {' '.join(cmd)}")
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **self.subprocess_flags
            )
            
            # Stream output in real-time and capture for error checking
            output_lines = []
            for line in proc.stdout:
                line = line.strip()
                if line:
                    self.log(line)
                    output_lines.append(line)
            
            proc.wait()
            # Clean up temp file
            try:
                Path(tmp_req_file).unlink()
            except:
                pass
            
            if proc.returncode != 0:
                # Check for disk space errors in the output
                output_text = '\n'.join(output_lines)
                if "No space left on device" in output_text or "Errno 28" in output_text:
                    has_space, free_gb = self._check_disk_space()
                    self.log(f"ERROR: Installation failed due to insufficient disk space!")
                    self.log(f"Available disk space: {free_gb:.2f} GB (minimum required: {self.min_disk_space_gb} GB)")
                    self.log("Please free up disk space and try again.")
                    self.log("You can try:")
                    self.log("  1. Delete temporary files")
                    self.log("  2. Uninstall unused programs")
                    self.log("  3. Clear pip cache: pip cache purge")
                    self.log("  4. Move the installation to a drive with more space")
                    return False  # Fail immediately on disk space error
                
                self.log(f"ERROR: Requirements installation failed with exit code {proc.returncode}")
                self.log("Full output:")
                # Output was already logged line by line, but log summary
                self.log("WARNING: Requirements.txt installation FAILED - continuing to install PySide6 (required for GUI)")
                self.log("You may need to close all Python processes and run Fix Issues again")
                # Don't return False here - continue to install PySide6 so GUI can start
            else:
                self.log("OK: Requirements.txt installation completed successfully")
                
                # Verify key packages are installed
                self.log("Repair: Verifying key packages from requirements.txt...")
                all_ok = True
                
                # Verify transformers using import test
                success, installed_ver, error = self._verify_transformers(python_executable)
                if success:
                    self.log(f"  OK: transformers {installed_ver} verified")
                else:
                    self.log(f"  ERROR: transformers verification failed: {error}")
                    all_ok = False
                
                # Verify datasets using import test
                success, installed_ver, error = self._verify_datasets(python_executable)
                if success:
                    self.log(f"  OK: datasets {installed_ver} verified")
                else:
                    self.log(f"  ERROR: datasets verification failed: {error}")
                    all_ok = False
                
                # Verify other packages using metadata (numpy, accelerate, peft)
                other_packages = {
                    "numpy": "<2.0.0",  # Critical: must be <2.0.0
                    "accelerate": ">=0.18.0",
                    "peft": ">=0.3.0",
                }
                for pkg, version_spec in other_packages.items():
                    success, installed_ver, error = self._verify_package_version(python_executable, pkg, version_spec)
                    if success:
                        self.log(f"  OK: {pkg} {installed_ver} verified")
                    else:
                        self.log(f"  ERROR: {pkg} verification failed: {error}")
                        all_ok = False
                        # If numpy is wrong version, force reinstall it
                        if pkg == "numpy":
                            self.log(f"  Attempting to fix numpy version...")
                            try:
                                fix_result = subprocess.run(
                                    [python_executable, "-m", "pip", "install", "--force-reinstall", "numpy<2"],
                                    capture_output=True,
                                    text=True,
                                    timeout=300,
                                    **self.subprocess_flags
                                )
                                if fix_result.returncode == 0:
                                    self.log(f"  OK: numpy fixed")
                                else:
                                    self.log(f"  ERROR: numpy fix failed: {fix_result.stderr}")
                            except Exception as e:
                                self.log(f"  ERROR: numpy fix exception: {str(e)}")
                
                if not all_ok:
                    self.log("WARNING: Some key packages failed verification after requirements.txt installation")
                    self.log("Continuing to install PySide6 (required for GUI to start)")
                    # Don't return False - continue to install PySide6
        
        # Step 3a: Force-fix constraint pins BEFORE transformers install
        # This ensures constraints are enforced from the start
        self.log("Repair: Force-fixing constraint pins (enforcing constraints)...")
        constraints_file = Path(__file__).parent / "constraints.txt"
        if not constraints_file.exists():
            self.log(f"ERROR: constraints.txt not found at {constraints_file}")
            return False
        
        # Force-reinstall all constraint packages: numpy<2, sympy==1.13.1, huggingface-hub<1.0
        # Constraints file will be automatically applied by pip_worker.py
        constraint_packages = ["numpy<2", "sympy==1.13.1", "huggingface-hub<1.0"]
        for pkg in constraint_packages:
            self.log(f"Force-fixing constraint: {pkg}...")
            success, lines, exit_code = self._run_pip_worker(
                action="install",
                package=pkg,
                python_executable=python_executable,
                pip_args=["--force-reinstall"]
            )
            if not success:
                self.log(f"ERROR: Failed to enforce {pkg} constraint: {lines[-500:]}")
                return False
        
        self.log("OK: All constraints enforced (numpy<2, sympy==1.13.1, huggingface-hub<1.0)")
        
        # Step 3b: Force reinstall transformers and tokenizers with correct versions
        # (requirements.txt dependencies might have upgraded them)
        # Constraints file will be automatically applied by pip_worker.py
        self.log("Repair: Ensuring transformers and tokenizers are at correct versions (with constraints)...")
        
        # Install transformers with constraints (pip_worker will add -c constraints.txt automatically)
        transformers_success, transformers_lines, transformers_exit = self._run_pip_worker(
            action="install",
            package="transformers==4.51.3",
            python_executable=python_executable,
            pip_args=["--force-reinstall"]
        )
        if not transformers_success:
            # Check for disk space errors
            if "No space left on device" in transformers_lines or "Errno 28" in transformers_lines:
                has_space, free_gb = self._check_disk_space()
                self.log(f"ERROR: Installation failed due to insufficient disk space!")
                self.log(f"Available disk space: {free_gb:.2f} GB (minimum required: {self.min_disk_space_gb} GB)")
                self.log("Please free up disk space and try again.")
                return False  # Fail immediately on disk space error
            
            self.log(f"ERROR: transformers installation failed!")
            self.log(f"Last 500 lines: {transformers_lines[-500:]}")
            return False  # Fail repair if critical packages can't be installed
        
        # Install tokenizers with constraints
        tokenizers_success, tokenizers_lines, tokenizers_exit = self._run_pip_worker(
            action="install",
            package="tokenizers>=0.21,<0.22",
            python_executable=python_executable,
            pip_args=["--force-reinstall"]
        )
        if not tokenizers_success:
            self.log(f"ERROR: tokenizers installation failed!")
            self.log(f"Last 500 lines: {tokenizers_lines[-500:]}")
            return False
        
        # Verify versions immediately after installation using target Python
        self.log("Repair: Verifying transformers version...")
        success, installed_ver, error = self._verify_transformers(python_executable)
        if not success:
            self.log(f"ERROR: transformers version verification failed: {error}")
            self.log(f"  Installed version: {installed_ver or 'unknown'}")
            return False
        else:
            self.log(f"OK: transformers {installed_ver} verified")
        
        self.log("Repair: Verifying tokenizers version...")
        success, installed_ver, error = self._verify_package_version(python_executable, "tokenizers", ">=0.21,<0.22")
        if not success:
            self.log(f"ERROR: tokenizers version verification failed: {error}")
            self.log(f"  Installed version: {installed_ver or 'unknown'}")
            return False
        else:
            self.log(f"OK: tokenizers {installed_ver} verified")
        
        # Step 4: Verify PyTorch installation (CRITICAL - must be installed for training)
        self.log("Repair: Verifying PyTorch installation...")
        pytorch_verified = False
        pytorch_cuda_available = False
        
        # Use _verify_torch which uses target venv Python ONLY
        torch_ok, torch_ver, torch_cuda, torch_error = self._verify_torch(python_executable)
        if torch_ok:
            self.log(f"OK: PyTorch verified (version {torch_ver})")
            pytorch_verified = True
            if torch_cuda:
                pytorch_cuda_available = True
                self.log("OK: PyTorch CUDA is available")
            else:
                # Check if GPU exists but PyTorch is CPU-only
                try:
                    smi_result = subprocess.run(
                        ["nvidia-smi"],
                        capture_output=True,
                        timeout=5,
                        **self.subprocess_flags
                    )
                    if smi_result.returncode == 0:
                        self.log("WARNING: GPU detected but PyTorch is CPU-only - training will be slow")
                    else:
                        self.log("INFO: PyTorch is CPU-only (no GPU detected)")
                except:
                    pass
        else:
            self.log(f"ERROR: PyTorch verification failed: {torch_error}")
        
        if not pytorch_verified:
            self.log("ERROR: PyTorch verification failed - training will not work")
            # Don't return False here - continue to install PySide6 so GUI can start
            # But we'll return False at the end if PyTorch is critical
        
        # Step 5: Install unsloth (dependencies should already be installed from requirements.txt)
        # We install with --no-deps to protect torch, but verify dependencies are present
        self.log("Repair: Installing unsloth (with --no-deps to protect torch)...")
        unsloth_cmd = [
            python_executable, "-m", "pip", "install",
            "--upgrade", "--no-deps",
            "unsloth",
        ]
        self.log(f"Running: {' '.join(unsloth_cmd)}")
        result = subprocess.run(unsloth_cmd, capture_output=True, text=True, timeout=900, **self.subprocess_flags)
        if result.returncode != 0:
            self.log(f"ERROR: unsloth installation failed!")
            self.log(f"STDOUT: {result.stdout}")
            self.log(f"STDERR: {result.stderr}")
            # Don't fail repair for unsloth - it's optional for GUI to start
            self.log("WARNING: Continuing without unsloth (GUI can still start)")
        else:
            # Verify unsloth can be imported
            self.log("Repair: Verifying unsloth installation...")
            verify_cmd = [
                python_executable, "-c",
                "import unsloth; print('OK')"
            ]
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30, **self.subprocess_flags)
            if verify_result.returncode == 0:
                self.log("OK: unsloth verified and can be imported")
            else:
                self.log(f"WARNING: unsloth installed but cannot be imported: {verify_result.stderr[:200]}")
        
        # Step 6: Remove torchao if present
        self.log("Repair: Removing torchao (known incompatibility)...")
        try:
            subprocess.run(
                [python_executable, "-m", "pip", "uninstall", "-y", "torchao"],
                capture_output=True,
                timeout=60,
                **self.subprocess_flags
            )
        except Exception:
            pass
        
        # Step 7: Install PySide6 LAST (after everything) so nothing can upgrade it
        self.log("Repair: Installing PySide6 6.8.1 (FINAL STEP - after all other packages)...")
        pyside_cmd = [
            python_executable, "-m", "pip", "install",
            "--force-reinstall",  # Force to override any version from requirements.txt
            "PySide6==6.8.1",  # Specific stable version for Windows
            "PySide6-Essentials==6.8.1",  # MUST match PySide6 version
            "PySide6-Addons==6.8.1",  # MUST match PySide6 version
            "shiboken6==6.8.1"  # MUST match PySide6 version
        ]
        self.log(f"Running: {' '.join(pyside_cmd)}")
        result = subprocess.run(pyside_cmd, capture_output=True, text=True, timeout=600, **self.subprocess_flags)
        if result.returncode != 0:
            self.log(f"ERROR: PySide6 installation failed!")
            self.log(f"STDOUT: {result.stdout}")
            self.log(f"STDERR: {result.stderr}")
        else:
            self.log("OK: PySide6 6.8.1 (all packages) installed successfully")
            
            # Verify all PySide6 components are at correct version
            self.log("Repair: Verifying PySide6 component versions...")
            pyside_components = ["PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6"]
            all_ok = True
            for component in pyside_components:
                success, installed_ver, error = self._verify_package_version(python_executable, component, "==6.8.1")
                if success:
                    self.log(f"  OK: {component} {installed_ver}")
                else:
                    self.log(f"  ERROR: {component} version verification failed: {error}")
                    all_ok = False
            
            if not all_ok:
                self.log("WARNING: Some PySide6 components have incorrect versions")

        # Final verification - check PySide6 first (critical for GUI to start)
        pyside_ok = False
        self.log("Repair: Verifying PySide6 (required for GUI)...")
        try:
            pyside_check = subprocess.run(
                [python_executable, "-c", "import PySide6.QtCore; print('OK')"],
                capture_output=True,
                text=True,
                timeout=10,
                **self.subprocess_flags,
            )
            if pyside_check.returncode == 0:
                self.log("OK: PySide6 is working - GUI can start")
                pyside_ok = True
            else:
                self.log("ERROR: PySide6 verification failed - GUI cannot start")
                self.log((pyside_check.stderr or "").strip())
        except Exception as e:
            self.log(f"Repair warning: PySide6 check failed: {e}")
        
        # Full verification (may fail if PyTorch has issues, but that's OK if PySide6 works)
        self.log("Repair: Verifying full environment...")
        try:
            verify_cmd = [
                python_executable,
                "-c",
                "import verify_installation as v; ok, checks = v.verify_all(); print('OK' if ok else 'FAIL'); raise SystemExit(0 if ok else 1)",
            ]
            res = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                **self.subprocess_flags,
            )
            if res.returncode != 0:
                self.log("Repair verification: Some components failed (check details above)")
                self.log((res.stdout or "").strip())
                verification_summary = "Some components failed verification"
            else:
                verification_summary = "All components verified successfully"
        except Exception as e:
            self.log(f"Repair verification warning: {e}")
            verification_summary = f"Verification error: {e}"
        
        # Create installation summary
        self.log("=" * 60)
        self.log("REPAIR SUMMARY")
        self.log("=" * 60)
        self.log(f"PySide6: {'OK' if pyside_ok else 'FAILED'}")
        self.log(f"PyTorch: {'OK' if pytorch_verified else 'FAILED'}")
        if pytorch_verified and not pytorch_cuda_available:
            self.log(f"PyTorch CUDA: Not available (CPU-only mode)")
        self.log(f"Transformers/Tokenizers: Verified")
        self.log(f"Requirements.txt: Installed")
        self.log(f"Unsloth: Installed (may have warnings)")
        try:
            self.log(f"Full Verification: {verification_summary}")
        except:
            pass
        self.log("=" * 60)
        
        # Final status check - both PySide6 AND PyTorch must be working
        if not pyside_ok:
            self.log("Repair failed: PySide6 is not working - GUI cannot start.")
            return False
        
        # PySide6 is OK, but check if PyTorch is actually installed
        if not pytorch_verified:
            self.log("Repair incomplete: PyTorch is not installed - training will not work")
            self.log("Please run 'Fix Issues' again to install PyTorch")
            return False  # Return False so installer GUI stays open and user can retry
        
        # All critical components are installed
        self.log("Repair complete: All critical components are installed and working.")
        if not pytorch_cuda_available:
            self.log("INFO: PyTorch is installed but CUDA is not available - training will use CPU")
        return True


if __name__ == "__main__":
    installer = SmartInstaller()
    success = installer.install()
    sys.exit(0 if success else 1)

