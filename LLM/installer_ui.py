#!/usr/bin/env python3
"""
Optional GUI Installer for LLM Fine-tuning Studio
Provides a graphical interface for installation with detection results display
"""

import sys
import os
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

from system_detector import SystemDetector
from smart_installer import SmartInstaller

class InstallerGUI:
    """Graphical installer interface"""
    
    def __init__(self):
        if not TKINTER_AVAILABLE:
            print("tkinter not available. Using console installer instead.")
            self.run_console_installer()
            return
        
        self.root = tk.Tk()
        self.root.title("LLM Fine-tuning Studio - Installer")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        self.detector = SystemDetector()
        self.installer = None
        self.detection_results = {}
        
        self.create_ui()
        self.run_detection()
    
    def create_ui(self):
        """Create the installer UI"""
        # Title
        title_frame = tk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="LLM Fine-tuning Studio",
            font=("Arial", 20, "bold")
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Smart Installer with Auto-Detection",
            font=("Arial", 12)
        )
        subtitle_label.pack()
        
        # Detection results frame
        detection_frame = tk.LabelFrame(self.root, text="System Detection Results", padx=10, pady=10)
        detection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.detection_text = scrolledtext.ScrolledText(
            detection_frame,
            height=15,
            wrap=tk.WORD,
            font=("Courier", 10)
        )
        self.detection_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.detect_button = tk.Button(
            button_frame,
            text="Re-detect System",
            command=self.run_detection,
            width=15
        )
        self.detect_button.pack(side=tk.LEFT, padx=5)
        
        self.install_button = tk.Button(
            button_frame,
            text="Install",
            command=self.start_installation,
            width=15,
            state=tk.DISABLED
        )
        self.install_button.pack(side=tk.LEFT, padx=5)
        
        self.verify_button = tk.Button(
            button_frame,
            text="Verify Installation",
            command=self.verify_installation,
            width=15
        )
        self.verify_button.pack(side=tk.LEFT, padx=5)
        
        self.close_button = tk.Button(
            button_frame,
            text="Close",
            command=self.root.quit,
            width=15
        )
        self.close_button.pack(side=tk.RIGHT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            mode='indeterminate',
            length=400
        )
        self.progress.pack(pady=10)
    
    def run_detection(self):
        """Run system detection and display results"""
        self.detection_text.delete(1.0, tk.END)
        self.detection_text.insert(tk.END, "Detecting system components...\n\n")
        self.root.update()
        
        self.progress.start()
        
        try:
            self.detection_results = self.detector.detect_all()
            self.display_results()
            self.install_button.config(state=tk.NORMAL)
        except Exception as e:
            self.detection_text.insert(tk.END, f"Error during detection: {e}\n")
            messagebox.showerror("Detection Error", f"Failed to detect system: {e}")
        finally:
            self.progress.stop()
    
    def display_results(self):
        """Display detection results in text widget"""
        self.detection_text.delete(1.0, tk.END)
        
        results = self.detection_results
        
        # Python
        python_info = results.get("python", {})
        if python_info.get("found"):
            self.detection_text.insert(tk.END, f"✓ Python {python_info.get('version')} found\n")
            self.detection_text.insert(tk.END, f"  Location: {python_info.get('executable')}\n")
            if python_info.get("pip_available"):
                self.detection_text.insert(tk.END, "  ✓ pip available\n")
            else:
                self.detection_text.insert(tk.END, "  ✗ pip not available\n")
        else:
            self.detection_text.insert(tk.END, "✗ Python not found\n")
            self.detection_text.insert(tk.END, "  → Will need to install Python\n")
        
        self.detection_text.insert(tk.END, "\n")
        
        # PyTorch
        pytorch_info = results.get("pytorch", {})
        if pytorch_info.get("found"):
            self.detection_text.insert(tk.END, f"✓ PyTorch {pytorch_info.get('version')} installed\n")
            if pytorch_info.get("cuda_available"):
                self.detection_text.insert(tk.END, f"  ✓ CUDA available (Version {pytorch_info.get('cuda_version')})\n")
            else:
                self.detection_text.insert(tk.END, "  ⚠ CPU mode only\n")
        else:
            self.detection_text.insert(tk.END, "✗ PyTorch not installed\n")
            self.detection_text.insert(tk.END, "  → Will install PyTorch\n")
        
        self.detection_text.insert(tk.END, "\n")
        
        # CUDA
        cuda_info = results.get("cuda", {})
        if cuda_info.get("found"):
            self.detection_text.insert(tk.END, f"✓ CUDA {cuda_info.get('version')} detected\n")
            gpus = cuda_info.get("gpus", [])
            if gpus:
                self.detection_text.insert(tk.END, f"  Found {len(gpus)} GPU(s):\n")
                for gpu in gpus:
                    self.detection_text.insert(tk.END, f"    - {gpu.get('name')} ({gpu.get('memory')})\n")
        else:
            self.detection_text.insert(tk.END, "⚠ CUDA not detected\n")
            self.detection_text.insert(tk.END, "  → Will use CPU mode\n")
        
        self.detection_text.insert(tk.END, "\n")
        
        # Hardware
        hardware_info = results.get("hardware", {})
        if hardware_info.get("ram_gb"):
            self.detection_text.insert(tk.END, f"💾 RAM: {hardware_info['ram_gb']:.1f} GB\n")
        if hardware_info.get("gpu", {}).get("found"):
            gpu = hardware_info["gpu"]
            self.detection_text.insert(tk.END, f"🎮 GPU: {gpu.get('model', 'Unknown')}\n")
            if gpu.get("memory_gb"):
                self.detection_text.insert(tk.END, f"  Memory: {gpu['memory_gb']:.1f} GB\n")
        
        self.detection_text.insert(tk.END, "\n")
        
        # Visual C++ (Windows)
        if sys.platform == "win32":
            vcredist_info = results.get("vcredist", {})
            if vcredist_info.get("found"):
                self.detection_text.insert(tk.END, "✓ Visual C++ Redistributables found\n")
            else:
                self.detection_text.insert(tk.END, "✗ Visual C++ Redistributables not found\n")
                self.detection_text.insert(tk.END, "  → Will install if needed\n")
        
        self.detection_text.insert(tk.END, "\n")
        
        # Recommendations
        recommendations = results.get("recommendations", {})
        self.detection_text.insert(tk.END, "Recommendations:\n")
        pytorch_build = recommendations.get("pytorch_build", "cpu")
        self.detection_text.insert(tk.END, f"  PyTorch build: {pytorch_build}\n")
    
    def start_installation(self):
        """Start the installation process"""
        response = messagebox.askyesno(
            "Confirm Installation",
            "This will install LLM Fine-tuning Studio and required dependencies.\n\n"
            "Continue?"
        )
        
        if not response:
            return
        
        self.install_button.config(state=tk.DISABLED)
        self.progress.start()
        
        try:
            self.installer = SmartInstaller()
            self.installer.detection_results = self.detection_results
            
            # Redirect installer output to text widget
            import io
            import contextlib
            
            class TextRedirect:
                def __init__(self, text_widget):
                    self.text_widget = text_widget
                
                def write(self, text):
                    self.text_widget.insert(tk.END, text)
                    self.text_widget.see(tk.END)
                    self.root.update()
            
            self.detection_text.delete(1.0, tk.END)
            self.detection_text.insert(tk.END, "Starting installation...\n\n")
            
            # Run installation
            success = self.installer.install()
            
            self.progress.stop()
            
            if success:
                messagebox.showinfo(
                    "Installation Complete",
                    "Installation completed successfully!\n\n"
                    "You can now launch the application."
                )
            else:
                messagebox.showwarning(
                    "Installation Warning",
                    "Installation completed with warnings.\n\n"
                    "Check the output above for details."
                )
        
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Installation Error", f"Installation failed: {e}")
        finally:
            self.install_button.config(state=tk.NORMAL)
    
    def verify_installation(self):
        """Run verification"""
        try:
            from verify_installation import verify_installation
            success = verify_installation()
            
            if success:
                messagebox.showinfo("Verification", "Installation verification passed!")
            else:
                messagebox.showwarning("Verification", "Some issues were found. Check console output.")
        except Exception as e:
            messagebox.showerror("Verification Error", f"Verification failed: {e}")
    
    def run_console_installer(self):
        """Fallback to console installer if tkinter not available"""
        print("=" * 60)
        print("LLM Fine-tuning Studio - Console Installer")
        print("=" * 60)
        print()
        
        installer = SmartInstaller()
        installer.install()
    
    def run(self):
        """Run the GUI installer"""
        if TKINTER_AVAILABLE:
            self.root.mainloop()
        else:
            self.run_console_installer()


if __name__ == "__main__":
    app = InstallerGUI()
    if TKINTER_AVAILABLE:
        app.run()
    else:
        app.run_console_installer()

