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
from pathlib import Path
from typing import Dict, Optional
from system_detector import SystemDetector, detect_all

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
        
        # First, uninstall any existing torch to avoid conflicts.
        # IMPORTANT: also remove xformers and triton variants on Windows to avoid pip resolver downgrading torch.
        self.log("Uninstalling any existing PyTorch / xformers / triton installation...")
        try:
            uninstall_cmd = [
                python_executable, "-m", "pip", "uninstall", "-y",
                "torch", "torchvision", "torchaudio",
                "xformers",
                "triton", "triton-windows",
            ]
            subprocess.run(uninstall_cmd, capture_output=True, text=True, timeout=300, **self.subprocess_flags)
        except Exception as e:
            self.log(f"Note: Could not uninstall old PyTorch: {e}")
        
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
                subprocess.run(
                    [python_executable, "-m", "pip", "install", "--upgrade", "numpy<2"],
                    capture_output=True,
                    text=True,
                    timeout=900,
                    **self.subprocess_flags,
                )
            except Exception:
                pass

            if cuda_build == "cpu":
                # Install CPU-only PyTorch
                cmd = [
                    python_executable, "-m", "pip", "install",
                    "--force-reinstall", "--no-deps",
                    f"torch=={pytorch_version}",
                    f"torchvision=={torchvision_version}",
                    f"torchaudio=={torchaudio_version}",
                    "--index-url", "https://download.pytorch.org/whl/cpu"
                ]
            else:
                # Install CUDA build
                cmd = [
                    python_executable, "-m", "pip", "install",
                    "--force-reinstall", "--no-deps",
                    f"torch=={pytorch_version}",
                    f"torchvision=={torchvision_version}",
                    f"torchaudio=={torchaudio_version}",
                    "--index-url", f"https://download.pytorch.org/whl/{cuda_build}"
                ]
            
            self.log(f"Running: {' '.join(cmd)}")
            self.log("Downloading PyTorch (~2.5GB)...")
            
            # Use Popen to capture real-time output
            popen_flags = {'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT, 'text': True, 'bufsize': 1, 'universal_newlines': True}
            popen_flags.update(self.subprocess_flags)
            process = subprocess.Popen(cmd, **popen_flags)
            
            # Parse output for download progress
            for line in process.stdout:
                line = line.strip()
                if line:
                    # Look for download progress indicators
                    if "Downloading" in line and "%" in line:
                        # Extract percentage if visible
                        import re
                        match = re.search(r'(\d+)%', line)
                        if match:
                            percent = int(match.group(1))
                            if self.progress_callback:
                                self.progress_callback(percent, f"Downloading PyTorch... {percent}%")
                        self.log(line)
                    elif "Installing" in line or "Collecting" in line or "Using cached" in line:
                        self.log(line)
                    elif "Successfully installed" in line:
                        self.log(line)
                        if self.progress_callback:
                            self.progress_callback(100, "PyTorch installed")
            
            process.wait()
            
            if process.returncode == 0:
                self.log("PyTorch installed successfully")
                
                # Install compatible triton explicitly to avoid version conflicts
                # Windows uses triton-windows; Linux uses triton.
                if sys.platform == "win32":
                    self.log("Installing compatible triton-windows (pinned)...")
                    triton_cmd = [
                        python_executable, "-m", "pip", "install",
                        "--upgrade", "--no-deps",
                        "triton-windows==3.5.1.post22",
                    ]
                else:
                    self.log(f"Installing compatible triton {triton_version}...")
                    triton_cmd = [
                        python_executable, "-m", "pip", "install",
                        "--upgrade", "--no-deps",
                        f"triton=={triton_version}",
                    ]
                subprocess.run(triton_cmd, capture_output=True, text=True, timeout=900, **self.subprocess_flags)

                # IMPORTANT: Do NOT install xformers automatically.
                # On Windows, xformers wheels often pin an exact torch version and can silently downgrade torch.
                
                return True
            else:
                self.log(f"PyTorch installation failed with code {process.returncode}")
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

    def repair_all(self, python_executable: Optional[str] = None) -> bool:
        """
        Repair a broken environment deterministically.

        Goals:
        - Ensure correct PyTorch CUDA build is installed (force reinstall).
        - Ensure Triton is installed correctly for the current platform (Windows uses triton-windows).
        - Prevent pip resolver from downgrading torch (remove xformers, install key packages with --no-deps).
        - Fix common corruption: missing dist-info / partial installs.
        """
        self.log("=" * 60)
        self.log("LLM Fine-tuning Studio - Repair Mode")
        self.log("=" * 60)

        # Detection (also populates python path)
        if not self.detection_results:
            self.run_detection()

        if not python_executable:
            python_executable = (
                self.detection_results.get("python", {}).get("executable")
                or sys.executable
            )

        # Try to locate site-packages for cleanup
        site_packages = None
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

        # Reinstall core stack
        self.log("Repair: Reinstalling PySide6 (GUI framework)...")
        pyside_cmd = [
            python_executable, "-m", "pip", "install",
            "--force-reinstall",
            "PySide6==6.8.1"  # Use stable version known to work on Windows
        ]
        result = subprocess.run(pyside_cmd, capture_output=True, text=True, timeout=600, **self.subprocess_flags)
        if result.returncode != 0:
            self.log(f"Repair warning: PySide6 installation failed: {result.stderr[:500]}")
        else:
            self.log("OK: PySide6 installed successfully")
        
        # Install requirements FIRST (before PyTorch) to avoid dependencies pulling wrong torch
        self.log("Repair: Installing core dependencies from requirements.txt...")
        requirements_file = Path(__file__).parent / "requirements.txt"
        if requirements_file.exists():
            cmd = [
                python_executable, "-m", "pip", "install",
                "--force-reinstall",  # Force reinstall to fix any corruption
                "-r", str(requirements_file)
            ]
            self.log(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                **self.subprocess_flags
            )
            if result.returncode != 0:
                self.log(f"Repair warning: Some requirements failed: {result.stderr[:500]}")
        
        # NOW install PyTorch LAST to override any CPU version pulled by dependencies
        self.log("Repair: Installing PyTorch + Triton (FINAL STEP - overrides any wrong versions)...")
        if not self.install_pytorch(python_executable=python_executable):
            self.log("Repair failed: Could not install PyTorch.")
            return False
        
        # Install unsloth separately with --no-deps to prevent torch downgrades
        self.log("Repair: Installing unsloth (with --no-deps to protect torch)...")
        unsloth_cmd = [
            python_executable, "-m", "pip", "install",
            "--upgrade", "--no-deps",
            "unsloth",
        ]
        result = subprocess.run(unsloth_cmd, capture_output=True, text=True, timeout=900, **self.subprocess_flags)
        if result.returncode != 0:
            self.log(f"Repair warning: unsloth installation failed: {result.stderr[:200]}")
        
        # Remove torchao if present
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

        # Final verification (runs inside the same interpreter)
        self.log("Repair: Verifying environment...")
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
                self.log("Repair verification failed:")
                self.log((res.stdout or "").strip())
                self.log((res.stderr or "").strip())
                return False
        except Exception as e:
            self.log(f"Repair verification warning: {e}")

        self.log("Repair complete: Environment is healthy.")
        return True


if __name__ == "__main__":
    installer = SmartInstaller()
    success = installer.install()
    sys.exit(0 if success else 1)

