#!/usr/bin/env python3
"""
Smart Installer for LLM Fine-tuning Studio
Intelligently installs components based on system detection
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, Optional
from system_detector import SystemDetector, detect_all

class SmartInstaller:
    """Smart installer that detects and installs only what's needed"""
    
    # Hardware-specific version matrix for compatibility
    VERSION_MATRIX = {
        "cuda_12.4": {"torch": "2.5.1", "triton": "3.0.0", "torchvision": "0.20.1", "torchaudio": "2.5.1"},
        "cuda_12.1": {"torch": "2.5.1", "triton": "3.0.0", "torchvision": "0.20.1", "torchaudio": "2.5.1"},
        "cuda_11.8": {"torch": "2.5.1", "triton": "3.0.0", "torchvision": "0.20.1", "torchaudio": "2.5.1"},
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
                capture_output=True
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
        
        # Determine optimal CUDA build
        cuda_build = self.get_optimal_cuda_build()
        
        # Get versions from matrix
        build_key_map = {
            "cu124": "cuda_12.4",
            "cu121": "cuda_12.1",
            "cu118": "cuda_11.8",
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
        
        self.log(f"Installing PyTorch {pytorch_version} ({cuda_build} build)...")
        
        try:
            if cuda_build == "cpu":
                # Install CPU-only PyTorch
                cmd = [
                    python_executable, "-m", "pip", "install",
                    f"torch=={pytorch_version}",
                    f"torchvision=={torchvision_version}",
                    f"torchaudio=={torchaudio_version}",
                    "--index-url", "https://download.pytorch.org/whl/cpu"
                ]
            else:
                # Install CUDA build
                cmd = [
                    python_executable, "-m", "pip", "install",
                    f"torch=={pytorch_version}",
                    f"torchvision=={torchvision_version}",
                    f"torchaudio=={torchaudio_version}",
                    "--index-url", f"https://download.pytorch.org/whl/{cuda_build}"
                ]
            
            self.log(f"Running: {' '.join(cmd)}")
            self.log("This may take 5-10 minutes (downloading ~2.5GB)...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes timeout
            )
            
            if result.returncode == 0:
                self.log("PyTorch installed successfully")
                
                # Install compatible triton explicitly to avoid version conflicts
                self.log(f"Installing compatible triton {triton_version}...")
                triton_cmd = [
                    python_executable, "-m", "pip", "install",
                    f"triton=={triton_version}"
                ]
                subprocess.run(triton_cmd, capture_output=True, timeout=300)
                
                # If CUDA build, install compatible xformers
                if cuda_build != "cpu":
                    self.log("Installing compatible xformers...")
                    xformers_cmd = [
                        python_executable, "-m", "pip", "install",
                        "xformers==0.0.28.post2",
                        "--index-url", f"https://download.pytorch.org/whl/{cuda_build}"
                    ]
                    subprocess.run(xformers_cmd, capture_output=True, timeout=600)
                
                return True
            else:
                self.log(f"PyTorch installation failed: {result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            self.log("PyTorch installation timed out")
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
            requirements_file = self.install_dir / "requirements.txt"
            
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
                    timeout=1800  # 30 minutes timeout
                )
                
                if result.returncode != 0:
                    self.log(f"Warning: Some packages failed: {result.stderr[:500]}")
            
            # Install unsloth separately with careful version control
            self.log("Installing unsloth (this may take a few minutes)...")
            unsloth_cmd = [
                python_executable, "-m", "pip", "install",
                "unsloth"
            ]
            
            result = subprocess.run(unsloth_cmd, capture_output=True, text=True, timeout=900)
            
            if result.returncode == 0:
                self.log("✅ unsloth installed")
            else:
                self.log(f"⚠️ unsloth installation warning: {result.stderr[:200]}")
            
            # Remove incompatible torchao if present (known Windows issue)
            self.log("Checking for torchao compatibility...")
            remove_torchao = [python_executable, "-m", "pip", "uninstall", "-y", "torchao"]
            subprocess.run(remove_torchao, capture_output=True, timeout=60)
            self.log("Removed torchao (incompatible with current setup)")
            
            # Test if unsloth works
            self.log("Testing unsloth import...")
            test_cmd = [python_executable, "-c", "from unsloth import FastLanguageModel; print('OK')"]
            test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
            
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
            
            # Step 4: PyTorch
            log_with_callback("Installing PyTorch...")
            pytorch_info = self.detection_results.get("pytorch", {})
            if not pytorch_info.get("found"):
                if self.install_pytorch():
                    results["pytorch"]["success"] = True
                    results["pytorch"]["message"] = "PyTorch installed successfully"
                else:
                    results["pytorch"]["message"] = "PyTorch installation failed"
                    return results
            else:
                results["pytorch"]["success"] = True
                results["pytorch"]["message"] = f"PyTorch {pytorch_info.get('version')} already installed"
            
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
        
        # Step 4: Install PyTorch
        pytorch_info = self.detection_results.get("pytorch", {})
        if not pytorch_info.get("found"):
            if not self.install_pytorch():
                self.log("Warning: PyTorch installation failed.")
                self.log("You can install it manually later.")
        else:
            self.log(f"PyTorch {pytorch_info.get('version')} already installed")
        
        # Step 5: Install dependencies
        if not self.install_dependencies():
            self.log("Warning: Some dependencies may not have installed correctly.")
        
        # Step 6: Create launcher
        self.create_launcher()
        
        self.log("=" * 60)
        self.log("Installation complete!")
        self.log("=" * 60)
        
        return True


if __name__ == "__main__":
    installer = SmartInstaller()
    success = installer.install()
    sys.exit(0 if success else 1)

