#!/usr/bin/env python3
"""
System Detection Module for LLM Fine-tuning Studio
Detects Python, PyTorch, CUDA, Visual C++ Redistributables, and hardware capabilities
"""

import os
import sys
import platform
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

class SystemDetector:
    """Detects system components and hardware capabilities"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.detection_results = {}
    
    def detect_all(self) -> Dict:
        """Run all detection methods and return results"""
        results = {
            "python": self.detect_python(),
            "pytorch": self.detect_pytorch(),
            "cuda": self.detect_cuda(),
            "hardware": self.detect_hardware(),
            "vcredist": self.detect_vcredist() if self.platform == "windows" else None,
            "recommendations": {}
        }
        
        # Generate recommendations
        results["recommendations"] = self.get_recommendations(results)
        
        self.detection_results = results
        return results
    
    def detect_python(self) -> Dict:
        """Detect Python installation"""
        result = {
            "found": False,
            "version": None,
            "path": None,
            "executable": None,
            "pip_available": False
        }
        
        # Check current Python
        if sys.executable:
            try:
                version = sys.version_info
                result["found"] = True
                result["version"] = f"{version.major}.{version.minor}.{version.micro}"
                result["executable"] = sys.executable
                result["path"] = os.path.dirname(sys.executable)
                
                # Check pip
                try:
                    import pip
                    result["pip_available"] = True
                except ImportError:
                    # Try running pip command
                    try:
                        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                      capture_output=True, check=True, timeout=5)
                        result["pip_available"] = True
                    except:
                        pass
            except Exception as e:
                pass
        
        # If not found, check common paths
        if not result["found"]:
            if self.platform == "windows":
                common_paths = [
                    r"C:\Python*",
                    r"C:\Program Files\Python*",
                    r"C:\Program Files (x86)\Python*",
                    os.path.expanduser(r"~\AppData\Local\Programs\Python\Python*")
                ]
                # Also check PATH
                path_dirs = os.environ.get("PATH", "").split(os.pathsep)
                for path_dir in path_dirs:
                    python_exe = os.path.join(path_dir, "python.exe")
                    if os.path.exists(python_exe):
                        try:
                            version_output = subprocess.run(
                                [python_exe, "--version"],
                                capture_output=True, text=True, timeout=5
                            )
                            if version_output.returncode == 0:
                                version_str = version_output.stdout.strip()
                                result["found"] = True
                                result["executable"] = python_exe
                                result["path"] = path_dir
                                # Extract version
                                import re
                                match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_str)
                                if match:
                                    result["version"] = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
                                break
                        except:
                            continue
            
            elif self.platform in ["linux", "darwin"]:
                # Try common commands
                for cmd in ["python3", "python"]:
                    try:
                        version_output = subprocess.run(
                            [cmd, "--version"],
                            capture_output=True, text=True, timeout=5
                        )
                        if version_output.returncode == 0:
                            which_output = subprocess.run(
                                ["which", cmd],
                                capture_output=True, text=True, timeout=5
                            )
                            if which_output.returncode == 0:
                                result["found"] = True
                                result["executable"] = which_output.stdout.strip()
                                result["path"] = os.path.dirname(result["executable"])
                                version_str = version_output.stdout.strip()
                                import re
                                match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_str)
                                if match:
                                    result["version"] = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
                                
                                # Check pip
                                try:
                                    subprocess.run([result["executable"], "-m", "pip", "--version"],
                                                 capture_output=True, check=True, timeout=5)
                                    result["pip_available"] = True
                                except:
                                    pass
                                break
                    except:
                        continue
        
        return result
    
    def detect_pytorch(self) -> Dict:
        """Detect PyTorch installation"""
        result = {
            "found": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device": "cpu"
        }
        
        try:
            import torch
            result["found"] = True
            result["version"] = torch.__version__
            
            # Check CUDA availability
            if torch.cuda.is_available():
                result["cuda_available"] = True
                result["cuda_version"] = torch.version.cuda
                result["device"] = "cuda"
            else:
                result["device"] = "cpu"
        except ImportError:
            # PyTorch not installed
            pass
        except Exception as e:
            # Error detecting PyTorch
            result["error"] = str(e)
        
        return result
    
    def detect_cuda(self) -> Dict:
        """Detect CUDA installation and GPU hardware"""
        result = {
            "found": False,
            "available": False,  # Add this for consistency
            "version": None,
            "driver_version": None,
            "gpus": [],
            "toolkit_path": None
        }
        
        # Try nvidia-smi first (works on all platforms)
        try:
            nvidia_smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            
            if nvidia_smi.returncode == 0:
                result["found"] = True
                result["available"] = True  # GPU is available
                lines = nvidia_smi.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            gpu_info = {
                                "name": parts[0],
                                "memory": parts[1],
                                "driver_version": parts[2]
                            }
                            result["gpus"].append(gpu_info)
                            if not result["driver_version"]:
                                result["driver_version"] = parts[2]
                
                # Try to get CUDA version from nvidia-smi
                try:
                    cuda_version_cmd = subprocess.run(
                        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=5
                    )
                    if cuda_version_cmd.returncode == 0:
                        # Get compute capability, map to CUDA version
                        compute_cap = cuda_version_cmd.stdout.strip().split('\n')[0].strip()
                        result["cuda_version"] = f"Compute {compute_cap}"
                except:
                    pass
                
                # Also try to get CUDA version from PyTorch if available
                try:
                    import torch
                    if torch.cuda.is_available() and torch.version.cuda:
                        result["cuda_version"] = torch.version.cuda
                except:
                    pass
        except FileNotFoundError:
            # nvidia-smi not found
            pass
        except Exception:
            pass
        
        # Check CUDA toolkit installation paths
        if self.platform == "windows":
            cuda_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA"
            ]
            for base_path in cuda_paths:
                if os.path.exists(base_path):
                    # Look for version directories
                    try:
                        versions = [d for d in os.listdir(base_path) 
                                   if os.path.isdir(os.path.join(base_path, d)) and d.replace('.', '').isdigit()]
                        if versions:
                            # Get latest version
                            versions.sort(key=lambda x: [int(i) for i in x.split('.')], reverse=True)
                            result["version"] = versions[0]
                            result["toolkit_path"] = os.path.join(base_path, versions[0])
                            result["found"] = True
                            # Don't set available=True unless we have actual GPUs
                            break
                    except:
                        continue
        
        elif self.platform == "linux":
            cuda_paths = ["/usr/local/cuda", "/usr/lib/cuda"]
            for cuda_path in cuda_paths:
                if os.path.exists(cuda_path):
                    result["toolkit_path"] = cuda_path
                    # Try to get version
                    version_file = os.path.join(cuda_path, "version.txt")
                    if os.path.exists(version_file):
                        try:
                            with open(version_file, 'r') as f:
                                content = f.read()
                                import re
                                match = re.search(r"CUDA Version (\d+\.\d+)", content)
                                if match:
                                    result["version"] = match.group(1)
                                    result["found"] = True
                        except:
                            pass
        
        elif self.platform == "darwin":  # macOS
            cuda_path = "/usr/local/cuda"
            if os.path.exists(cuda_path):
                result["toolkit_path"] = cuda_path
                result["found"] = True
        
        return result
    
    def detect_hardware(self) -> Dict:
        """Detect hardware capabilities"""
        result = {
            "cpu_name": None,  # Add top-level cpu_name for easy access
            "cpu": {
                "cores": None,
                "architecture": platform.machine(),
                "processor": platform.processor()
            },
            "gpu": {
                "found": False,
                "vendor": None,
                "model": None,
                "memory_gb": None
            },
            "ram_gb": None,
            "disk_space_gb": None
        }
        
        # Get CPU name/processor info
        try:
            if self.platform == "windows":
                # Use WMI to get actual CPU name on Windows
                try:
                    import subprocess
                    cpu_cmd = subprocess.run(
                        ["wmic", "cpu", "get", "name"],
                        capture_output=True, text=True, timeout=5
                    )
                    if cpu_cmd.returncode == 0:
                        lines = cpu_cmd.stdout.strip().split('\n')
                        if len(lines) > 1:
                            result["cpu_name"] = lines[1].strip()
                        else:
                            result["cpu_name"] = platform.processor()
                    else:
                        result["cpu_name"] = platform.processor()
                except:
                    result["cpu_name"] = platform.processor()
            else:
                processor = platform.processor()
                if processor:
                    result["cpu_name"] = processor
                else:
                    # Fallback to cores + architecture
                    cores = os.cpu_count()
                    arch = platform.machine()
                    result["cpu_name"] = f"{cores}-core {arch}"
        except:
            result["cpu_name"] = "Unknown"
        
        # CPU cores
        try:
            if self.platform == "windows":
                result["cpu"]["cores"] = os.cpu_count()
            else:
                result["cpu"]["cores"] = os.cpu_count()
        except:
            pass
        
        # RAM
        try:
            if self.platform == "windows":
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                result["ram_gb"] = round(stat.ullTotalPhys / (1024**3), 2)
            elif self.platform == "linux":
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            result["ram_gb"] = round(kb / (1024**2), 2)
                            break
            elif self.platform == "darwin":
                mem_output = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5
                )
                if mem_output.returncode == 0:
                    bytes = int(mem_output.stdout.strip())
                    result["ram_gb"] = round(bytes / (1024**3), 2)
        except:
            pass
        
        # GPU (from CUDA detection or system)
        cuda_info = self.detect_cuda()
        if cuda_info["found"] and cuda_info["gpus"]:
            gpu = cuda_info["gpus"][0]
            result["gpu"]["found"] = True
            result["gpu"]["vendor"] = "NVIDIA"
            result["gpu"]["model"] = gpu.get("name", "Unknown")
            # Parse memory
            try:
                mem_str = gpu.get("memory", "")
                if "MiB" in mem_str:
                    mem_mib = int(mem_str.replace("MiB", "").strip())
                    result["gpu"]["memory_gb"] = round(mem_mib / 1024, 2)
            except:
                pass
        
        # Disk space (check current directory)
        try:
            if self.platform == "windows":
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(os.getcwd()),
                    ctypes.pointer(ctypes.c_ulonglong()),
                    ctypes.pointer(ctypes.c_ulonglong()),
                    ctypes.pointer(free_bytes)
                )
                result["disk_space_gb"] = round(free_bytes.value / (1024**3), 2)
            else:
                import shutil
                stat = shutil.disk_usage(os.getcwd())
                result["disk_space_gb"] = round(stat.free / (1024**3), 2)
        except:
            pass
        
        return result
    
    def detect_vcredist(self) -> Dict:
        """Detect Visual C++ Redistributables (Windows only)"""
        result = {
            "found": False,
            "versions": [],
            "required_dlls": {
                "msvcp140.dll": False,
                "vcruntime140.dll": False
            }
        }
        
        if self.platform != "windows":
            return result
        
        try:
            import winreg
            
            # Check registry for installed versions
            reg_paths = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            ]
            
            for hkey, path in reg_paths:
                try:
                    key = winreg.OpenKey(hkey, path)
                    try:
                        i = 0
                        while True:
                            try:
                                subkey_name = winreg.EnumKey(key, i)
                                if "VC" in subkey_name or "Visual C++" in subkey_name or "vcredist" in subkey_name.lower():
                                    result["versions"].append(subkey_name)
                                    result["found"] = True
                                i += 1
                            except WindowsError:
                                break
                    finally:
                        winreg.CloseKey(key)
                except FileNotFoundError:
                    continue
                except Exception:
                    continue
            
            # Check for required DLLs in system directories
            system_dirs = [
                os.path.join(os.environ.get("SYSTEMROOT", "C:\\Windows"), "System32"),
                os.path.join(os.environ.get("SYSTEMROOT", "C:\\Windows"), "SysWOW64"),
            ]
            
            for dll_name in result["required_dlls"].keys():
                for sys_dir in system_dirs:
                    dll_path = os.path.join(sys_dir, dll_name)
                    if os.path.exists(dll_path):
                        result["required_dlls"][dll_name] = True
                        result["found"] = True
                        break
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def get_recommendations(self, detection_results: Dict) -> Dict:
        """Generate installation recommendations based on detection"""
        recommendations = {
            "python": "use_existing" if detection_results["python"]["found"] else "install",
            "pytorch": "install",
            "pytorch_build": "cpu",
            "vcredist": "install" if (self.platform == "windows" and 
                                     not detection_results.get("vcredist", {}).get("found", False)) else "skip"
        }
        
        # Determine PyTorch build recommendation
        cuda_info = detection_results.get("cuda", {})
        hardware_info = detection_results.get("hardware", {})
        pytorch_info = detection_results.get("pytorch", {})
        
        # Check if CUDA is available in PyTorch (most reliable)
        if pytorch_info.get("cuda_available"):
            cuda_version = pytorch_info.get("cuda_version")
            if cuda_version:
                # Map CUDA version to PyTorch CUDA build
                major_version = cuda_version.split('.')[0]
                if major_version == "12":
                    recommendations["pytorch_build"] = "cu121"  # CUDA 12.1
                elif major_version == "11":
                    recommendations["pytorch_build"] = "cu118"  # CUDA 11.8
                else:
                    recommendations["pytorch_build"] = "cpu"
        # Fallback: Check CUDA detection and GPU
        elif cuda_info.get("found") and hardware_info.get("gpu", {}).get("found"):
            # CUDA detected but PyTorch doesn't have CUDA support
            # Check driver version to infer CUDA version
            driver_version = cuda_info.get("driver_version")
            if driver_version:
                try:
                    driver_major = int(driver_version.split('.')[0])
                    # Driver 560+ typically supports CUDA 12.x
                    if driver_major >= 560:
                        recommendations["pytorch_build"] = "cu121"
                    elif driver_major >= 520:
                        recommendations["pytorch_build"] = "cu118"
                    else:
                        recommendations["pytorch_build"] = "cpu"
                except:
                    recommendations["pytorch_build"] = "cu121"  # Default to latest
            else:
                recommendations["pytorch_build"] = "cu121"  # Default to latest CUDA
        else:
            recommendations["pytorch_build"] = "cpu"
        
        return recommendations


def detect_all() -> Dict:
    """Convenience function to run all detection"""
    detector = SystemDetector()
    return detector.detect_all()


if __name__ == "__main__":
    # Test detection
    detector = SystemDetector()
    results = detector.detect_all()
    print(json.dumps(results, indent=2))

