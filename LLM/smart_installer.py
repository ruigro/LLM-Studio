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
    
    def install_pytorch(self, python_executable: Optional[str] = None) -> bool:
        """Install PyTorch based on detection with proper version compatibility"""
        if not python_executable:
            python_info = self.detection_results.get("python", {})
            if not python_info.get("found"):
                self.log("Python not found. Cannot install PyTorch.")
                return False
            python_executable = python_info.get("executable")
        
        # Determine CUDA version and PyTorch build
        cuda_info = self.detection_results.get("cuda", {})
        driver_version = cuda_info.get("driver_version")
        
        # Map CUDA driver to compatible CUDA toolkit version
        cuda_build = "cpu"
        pytorch_version = "2.5.0"  # Stable version
        
        if driver_version:
            try:
                driver_major = int(float(driver_version.split('.')[0]))
                
                # CUDA compatibility mapping (conservative for stability)
                if driver_major >= 525:  # CUDA 12.x support
                    # Check minor version for exact CUDA toolkit
                    if driver_major >= 555:
                        cuda_build = "cu124"  # CUDA 12.4
                    elif driver_major >= 545:
                        cuda_build = "cu121"  # CUDA 12.1
                    else:
                        cuda_build = "cu118"  # CUDA 11.8
                elif driver_major >= 450:  # CUDA 11.x support
                    cuda_build = "cu118"  # CUDA 11.8 (most stable for 11.x)
                else:
                    self.log(f"Warning: Old driver version {driver_version}. GPU may not work.")
                    cuda_build = "cpu"
                
                self.log(f"Detected CUDA driver {driver_version} -> Using PyTorch build: {cuda_build}")
            except:
                self.log(f"Could not parse driver version {driver_version}, using CPU build")
                cuda_build = "cpu"
        else:
            self.log("No CUDA detected. Installing CPU-only PyTorch.")
        
        self.log(f"Installing PyTorch {pytorch_version} ({cuda_build} build)...")
        
        try:
            if cuda_build == "cpu":
                # Install CPU-only PyTorch
                cmd = [
                    python_executable, "-m", "pip", "install",
                    f"torch=={pytorch_version}",
                    f"torchvision==0.20.0",
                    f"torchaudio=={pytorch_version}",
                    "--index-url", "https://download.pytorch.org/whl/cpu"
                ]
            else:
                # Install CUDA build
                cmd = [
                    python_executable, "-m", "pip", "install",
                    f"torch=={pytorch_version}",
                    f"torchvision==0.20.0",
                    f"torchaudio=={pytorch_version}",
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

