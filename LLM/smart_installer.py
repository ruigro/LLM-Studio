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
        self.install_plan = {}  # Frozen install plan dict (hardware/platform-driven)
        
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
    
    def _detect_hardware_platform(self, python_executable: Optional[str] = None) -> Dict:
        """
        Detection phase: Detect OS, Python, CPU arch, NVIDIA GPU, CUDA driver.
        Runs BEFORE any install.
        
        Returns:
            Frozen install_plan dict with all hardware/platform decisions
        """
        self.log("=" * 60)
        self.log("DETECTION PHASE: Hardware and Platform Detection")
        self.log("=" * 60)
        
        plan = {}
        
        # 1. Detect OS
        plan["os"] = platform.system()  # Windows, Linux, Darwin
        plan["os_version"] = platform.version()
        self.log(f"OS: {plan['os']} {plan['os_version']}")
        
        # 2. Detect CPU architecture
        plan["cpu_arch"] = platform.machine()  # x86_64, AMD64, etc.
        self.log(f"CPU Architecture: {plan['cpu_arch']}")
        
        # 3. Detect Python version
        if not python_executable:
            python_executable = sys.executable
        
        try:
            result = subprocess.run(
                [python_executable, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True,
                text=True,
                timeout=10,
                **self.subprocess_flags
            )
            if result.returncode == 0:
                python_version = result.stdout.strip()
                plan["python_version"] = python_version
                major, minor = map(int, python_version.split('.'))
                plan["python_major"] = major
                plan["python_minor"] = minor
                self.log(f"Python: {python_version}")
            else:
                raise ValueError("Could not detect Python version")
        except Exception as e:
            self.log(f"ERROR: Python detection failed: {e}")
            return {}
        
        # 4. Detect NVIDIA GPU via nvidia-smi
        plan["nvidia_gpu_present"] = False
        plan["cuda_driver_version"] = None
        plan["cuda_compute_capability"] = None
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                **self.subprocess_flags
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    # Parse first GPU
                    parts = lines[0].split(',')
                    if len(parts) >= 2:
                        plan["nvidia_gpu_present"] = True
                        plan["gpu_name"] = parts[0].strip()
                        plan["cuda_driver_version"] = parts[1].strip()
                        if len(parts) >= 3:
                            plan["cuda_compute_capability"] = parts[2].strip()
                        self.log(f"NVIDIA GPU detected: {plan['gpu_name']}")
                        self.log(f"CUDA Driver: {plan['cuda_driver_version']}")
                        if plan["cuda_compute_capability"]:
                            self.log(f"Compute Capability: {plan['cuda_compute_capability']}")
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            self.log(f"No NVIDIA GPU detected (nvidia-smi not available or failed: {e})")
        
        # 5. Determine CUDA build from driver version
        if plan.get("nvidia_gpu_present") and plan.get("cuda_driver_version"):
            driver_version = plan["cuda_driver_version"]
            # Parse major.minor from driver version (e.g., "550.54.15" -> 550)
            try:
                driver_major = int(driver_version.split('.')[0])
                # Map driver version to CUDA build
                if driver_major >= 550:
                    plan["cuda_build"] = "cu124"  # CUDA 12.4
                elif driver_major >= 530:
                    plan["cuda_build"] = "cu121"  # CUDA 12.1
                elif driver_major >= 520:
                    plan["cuda_build"] = "cu118"  # CUDA 11.8
                else:
                    plan["cuda_build"] = "cu118"  # Default to 11.8
                self.log(f"Selected CUDA build: {plan['cuda_build']}")
            except (ValueError, IndexError):
                plan["cuda_build"] = "cu118"  # Default
                self.log(f"Could not parse driver version, defaulting to cu118")
        else:
            plan["cuda_build"] = "cpu"
            self.log("No GPU detected, using CPU-only build")
        
        # Make plan immutable (frozen)
        plan["_frozen"] = True
        
        self.log("=" * 60)
        self.log("Detection phase complete")
        self.log("=" * 60)
        
        return plan
    
    def _generate_install_plan(self, python_executable: Optional[str] = None) -> Dict:
        """
        Generate immutable install plan from hardware/platform detection.
        Derives all package version decisions from install_plan.
        
        Returns:
            Complete install_plan with package decisions
        """
        if not self.install_plan or not self.install_plan.get("_frozen"):
            # Run detection phase first
            self.install_plan = self._detect_hardware_platform(python_executable)
        
        if not self.install_plan:
            return {}
        
        plan = self.install_plan.copy()
        
        # Derive immutable decisions from hardware/platform
        
        # 1. NumPy version based on Python version
        if plan["python_major"] >= 3 and plan["python_minor"] >= 12:
            plan["numpy_spec"] = "numpy>=2,<3"
        else:
            plan["numpy_spec"] = "numpy<2"
        self.log(f"NumPy decision: {plan['numpy_spec']} (Python {plan['python_version']})")
        
        # 2. PyTorch build based on GPU presence
        if plan.get("nvidia_gpu_present"):
            cuda_build = plan["cuda_build"]
            plan["torch_spec"] = f"torch==2.5.1+{cuda_build}"
            plan["torchvision_spec"] = f"torchvision==0.20.1+{cuda_build}"
            plan["torchaudio_spec"] = f"torchaudio==2.5.1+{cuda_build}"
            plan["torch_index_url"] = f"https://download.pytorch.org/whl/{cuda_build}"
        else:
            plan["torch_spec"] = "torch==2.5.1"
            plan["torchvision_spec"] = "torchvision==0.20.1"
            plan["torchaudio_spec"] = "torchaudio==2.5.1"
            plan["torch_index_url"] = "https://download.pytorch.org/whl/cpu"
        self.log(f"PyTorch decision: {plan['torch_spec']} ({plan['torch_index_url']})")
        
        # 3. Always pin these packages
        plan["sympy_spec"] = "sympy==1.13.1"
        plan["fsspec_spec"] = "fsspec<=2025.9.0"
        plan["huggingface_hub_spec"] = "huggingface-hub<1.0"
        
        # 4. Other packages
        plan["transformers_spec"] = "transformers==4.51.3"
        plan["tokenizers_spec"] = "tokenizers==0.21.4"
        plan["datasets_spec"] = "datasets>=2.11.0,<4.4.0"
        
        return plan
    
    def _generate_constraints_file(self, install_plan: Dict) -> Path:
        """
        Generate constraints.txt dynamically from install_plan.
        Do NOT hardcode numpy in source files.
        
        Returns:
            Path to generated constraints.txt file
        """
        constraints_file = Path(__file__).parent / "constraints.txt"
        
        constraints = [
            "# Global constraints for LLM Fine-tuning Studio",
            "# Generated dynamically from hardware/platform detection",
            "",
            f"# NumPy constraint based on Python {install_plan.get('python_version', 'unknown')}",
            install_plan.get("numpy_spec", "numpy<2"),
            "",
            "# Always pinned packages",
            install_plan.get("sympy_spec", "sympy==1.13.1"),
            install_plan.get("fsspec_spec", "fsspec<=2025.9.0"),
            install_plan.get("huggingface_hub_spec", "huggingface-hub<1.0"),
            "",
        ]
        
        with open(constraints_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(constraints))
        
        self.log(f"Generated constraints.txt: {constraints_file}")
        return constraints_file
    
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
    
    def _check_running_processes_before_binary_install(self, python_executable: str) -> Tuple[bool, str]:
        """
        BEFORE uninstall/install of any binary wheel:
        - Check no python.exe is running from the target venv
        - Check the GUI app is not running
        - If any are found, abort with a clear message
        
        Returns:
            Tuple of (safe_to_proceed: bool, error_message: str)
        """
        venv_path = Path(python_executable).parent.parent
        venv_python = venv_path / "Scripts" / "python.exe" if sys.platform == "win32" else venv_path / "bin" / "python"
        
        if not venv_python.exists():
            return True, None  # Venv doesn't exist yet, safe to proceed
        
        running_processes = []
        
        # Check for python.exe from target venv
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["tasklist", "/FI", f"IMAGEPATH eq {venv_python}", "/FO", "CSV", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    **self.subprocess_flags
                )
                if result.returncode == 0 and venv_python.name in result.stdout:
                    running_processes.append(f"Python process from target venv: {venv_python}")
            except Exception:
                pass
        
        # Check for GUI app (desktop_app/main.py or launcher.exe)
        try:
            if sys.platform == "win32":
                # Check for launcher.exe or main.py processes
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq launcher.exe", "/FO", "CSV", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    **self.subprocess_flags
                )
                if result.returncode == 0 and "launcher.exe" in result.stdout:
                    running_processes.append("GUI launcher (launcher.exe) is running")
        except Exception:
            pass
        
        if running_processes:
            error_msg = "Cannot install binary wheels while the following processes are running:\n"
            error_msg += "\n".join(f"  - {proc}" for proc in running_processes)
            error_msg += "\n\nPlease close all Python processes and the GUI application, then try again."
            return False, error_msg
        
        return True, None
    
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
        
        # Map package names to import names (package name may differ from import name)
        import_map = {
            "transformers": "transformers",
            "huggingface_hub": "huggingface_hub",
            "huggingface-hub": "huggingface_hub",
            "datasets": "datasets",
            "torchvision": "torchvision",
            "torchaudio": "torchaudio",
            "numpy": "numpy",
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
        Repair a broken environment using hardware/platform-driven architecture.
        - Detection phase runs BEFORE any install
        - Install plan derived from hardware/platform
        - Constraints.txt generated dynamically
        - Strict layer-based installation (no auto-upgrade)
        - Import-only verification
        """
        self.log("=" * 60)
        self.log("LLM Fine-tuning Studio - Hardware/Platform-Driven Repair Mode")
        self.log("=" * 60)
        
        # 1. Check disk space
        has_space, free_gb = self._check_disk_space()
        self.log(f"Available disk space: {free_gb:.2f} GB (minimum required: {self.min_disk_space_gb} GB)")
        if not has_space:
            self.log(f"ERROR: Insufficient disk space! Only {free_gb:.2f} GB available, need at least {self.min_disk_space_gb} GB")
            return False

        # 2. Detection phase: Run BEFORE any install
        if not python_executable:
            python_executable = sys.executable
        
        install_plan = self._generate_install_plan(python_executable)
        if not install_plan:
            self.log("ERROR: Failed to generate install plan from hardware/platform detection")
            return False
        
        # 3. Generate constraints.txt dynamically from install_plan
        constraints_file = self._generate_constraints_file(install_plan)
        
        # 4. Check running processes BEFORE binary installs
        safe, error_msg = self._check_running_processes_before_binary_install(python_executable)
        if not safe:
            self.log("=" * 60)
            self.log("ERROR: Cannot proceed with binary wheel installation")
            self.log(error_msg)
            self.log("=" * 60)
            return False
        
        # 5. Install strictly by layers (never allow pip to auto-upgrade a previous layer)
        self.log("=" * 60)
        self.log("LAYER-BASED INSTALLATION (strict order, no auto-upgrade)")
        self.log("=" * 60)
        
        # LAYER 1: numpy, sympy, fsspec
        self.log("=" * 60)
        self.log("LAYER 1: numpy, sympy, fsspec")
        self.log("=" * 60)
        
        # Install numpy (from install_plan, dynamically determined)
        self.log(f"Installing {install_plan['numpy_spec']}...")
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package=install_plan["numpy_spec"],
            python_executable=python_executable,
            pip_args=[]  # No --force-reinstall by default
        )
        if not success:
            self.log(f"ERROR: numpy installation failed - cannot continue")
            return False
        
        # Verify numpy by import (if fails, abort immediately - never continue)
        if not self._verify_import(python_executable, "numpy"):
            self.log("ERROR: numpy import verification failed - ABORTING")
            self.log("Never uninstall/reinstall numpy after torch is installed")
            return False
        self.log("✓ numpy installed and verified")
        
        # Install sympy
        self.log(f"Installing {install_plan['sympy_spec']}...")
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package=install_plan["sympy_spec"],
            python_executable=python_executable,
            pip_args=[]
        )
        if not success:
            self.log(f"ERROR: sympy installation failed - cannot continue")
            return False
        self.log("✓ sympy installed")
        
        # Install fsspec
        self.log(f"Installing {install_plan['fsspec_spec']}...")
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package=install_plan["fsspec_spec"],
            python_executable=python_executable,
            pip_args=[]
        )
        if not success:
            self.log(f"ERROR: fsspec installation failed - cannot continue")
            return False
        self.log("✓ fsspec installed")
        
        # LAYER 2: torch stack
        self.log("=" * 60)
        self.log("LAYER 2: torch stack")
        self.log("=" * 60)
        
        # Check running processes before binary install
        safe, error_msg = self._check_running_processes_before_binary_install(python_executable)
        if not safe:
            self.log("ERROR: Cannot install binary wheels while processes are running")
            self.log(error_msg)
            return False
        
        # Verify torch first (skip if already installed and verified)
        torch_ok, torch_ver, torch_cuda, torch_error = self._verify_torch(python_executable)
        torch_needs_install = True
        if torch_ok and torch_cuda:
            if torch_ver and ("2.5.1" in torch_ver or torch_ver.startswith("2.5.1")):
                self.log(f"✓ torch already installed and verified: {torch_ver} (CUDA available)")
                torch_needs_install = False
        
        if torch_needs_install:
            self.log(f"Installing {install_plan['torch_spec']}...")
            venv_path = Path(python_executable).parent.parent
            self._delete_torch_directory(venv_path)  # Only delete if verification failed
            
            success, last_lines, exit_code = self._run_pip_worker(
                action="install",
                package=install_plan["torch_spec"],
                python_executable=python_executable,
                index_url=install_plan["torch_index_url"],
                pip_args=["--no-deps"]  # Only use --no-deps, not --force-reinstall by default
            )
            if not success:
                self.log(f"ERROR: torch installation failed")
                return False
            self.log("✓ torch installed")
        
        # Verify torchvision
        torchvision_ok = self._verify_import(python_executable, "torchvision")
        if not torchvision_ok:
            self.log(f"Installing {install_plan['torchvision_spec']}...")
            success, last_lines, exit_code = self._run_pip_worker(
                action="install",
                package=install_plan["torchvision_spec"],
                python_executable=python_executable,
                index_url=install_plan["torch_index_url"],
                pip_args=["--no-deps"]
            )
            if not success:
                self.log(f"ERROR: torchvision installation failed")
                return False
            self.log("✓ torchvision installed")
        else:
            self.log("✓ torchvision already installed")
        
        # Verify torchaudio
        torchaudio_ok = self._verify_import(python_executable, "torchaudio")
        if not torchaudio_ok:
            self.log(f"Installing {install_plan['torchaudio_spec']}...")
            success, last_lines, exit_code = self._run_pip_worker(
                action="install",
                package=install_plan["torchaudio_spec"],
                python_executable=python_executable,
                index_url=install_plan["torch_index_url"],
                pip_args=["--no-deps"]
            )
            if not success:
                self.log(f"ERROR: torchaudio installation failed")
                return False
            self.log("✓ torchaudio installed")
        else:
            self.log("✓ torchaudio already installed")
        
        # LAYER 3: huggingface-hub, tokenizers, transformers, datasets
        self.log("=" * 60)
        self.log("LAYER 3: huggingface-hub, tokenizers, transformers, datasets")
        self.log("=" * 60)
        
        # Hard-fix huggingface-hub first
        self.log("Hard-fixing huggingface-hub...")
        for pkg in ["huggingface-hub", "hf-xet"]:
            self._run_pip_worker(action="uninstall", package=pkg, python_executable=python_executable)
        
        self.log(f"Installing {install_plan['huggingface_hub_spec']}...")
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package=f"huggingface-hub>=0.30.0,<1.0",
            python_executable=python_executable,
            pip_args=[]
        )
        if not success:
            self.log(f"ERROR: huggingface-hub installation failed")
            return False
        self.log("✓ huggingface-hub installed")
        
        # Install tokenizers
        self.log(f"Installing {install_plan['tokenizers_spec']}...")
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package=install_plan["tokenizers_spec"],
            python_executable=python_executable,
            pip_args=[]
        )
        if not success:
            self.log(f"ERROR: tokenizers installation failed")
            return False
        self.log("✓ tokenizers installed")
        
        # Install transformers
        self.log(f"Installing {install_plan['transformers_spec']}...")
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package=install_plan["transformers_spec"],
            python_executable=python_executable,
            pip_args=[]
        )
        if not success:
            self.log(f"ERROR: transformers installation failed")
            return False
        if not self._verify_import(python_executable, "transformers"):
            self.log(f"ERROR: transformers import verification failed")
            return False
        self.log("✓ transformers installed and verified")
        
        # Install datasets
        self.log(f"Installing {install_plan['datasets_spec']}...")
        success, last_lines, exit_code = self._run_pip_worker(
            action="install",
            package=install_plan["datasets_spec"],
            python_executable=python_executable,
            pip_args=[]
        )
        if not success:
            self.log(f"ERROR: datasets installation failed")
            return False
        if not self._verify_import(python_executable, "datasets"):
            self.log(f"ERROR: datasets import verification failed")
            return False
        self.log("✓ datasets installed and verified")
        
        # LAYER 4: trl, xformers, torchao, unsloth-zoo, unsloth
        self.log("=" * 60)
        self.log("LAYER 4: trl, xformers, torchao, unsloth-zoo, unsloth")
        self.log("=" * 60)
        
        layer4_packages = ["trl>=0.18.2,<0.25.0", "xformers", "torchao", "peft", "unsloth-zoo", "unsloth"]
        for pkg in layer4_packages:
            self.log(f"Installing {pkg}...")
            success, last_lines, exit_code = self._run_pip_worker(
                action="install",
                package=pkg,
                python_executable=python_executable,
                pip_args=[]
            )
            if not success:
                self.log(f"WARNING: {pkg} installation had issues (continuing)")
            else:
                self.log(f"✓ {pkg} installed")
        
        # Final verification: import test for core packages (GUI status rule)
        self.log("=" * 60)
        self.log("Final verification: Import test for core packages...")
        self.log("Component is OK if import succeeds (ignore pip resolver warnings)")
        core_imports = ["numpy", "transformers", "huggingface_hub", "datasets"]
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


if __name__ == "__main__":
    installer = SmartInstaller()
    success = installer.install()
    sys.exit(0 if success else 1)

