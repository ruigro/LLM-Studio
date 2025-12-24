#!/usr/bin/env python3
"""
Bootstrap Setup - Creates venv and installs PySide6 before launching main setup wizard
Uses tkinter (built-in) so no dependencies needed
"""
import sys
import subprocess
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import threading

class BootstrapSetup:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LLM Fine-tuning Studio - Bootstrap")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.root.winfo_screenheight() // 2) - (400 // 2)
        self.root.geometry(f"600x400+{x}+{y}")
        
        self._build_ui()
        
    def _build_ui(self):
        # Header
        header = tk.Label(
            self.root,
            text="🚀 LLM Fine-tuning Studio",
            font=("Arial", 24, "bold"),
            fg="#4A90E2"
        )
        header.pack(pady=20)
        
        subtitle = tk.Label(
            self.root,
            text="First-Time Setup - Preparing Environment",
            font=("Arial", 12)
        )
        subtitle.pack(pady=5)
        
        # Status
        self.status_label = tk.Label(
            self.root,
            text="⚙️ Initializing...",
            font=("Arial", 11),
            fg="#333"
        )
        self.status_label.pack(pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            length=500,
            mode='indeterminate'
        )
        self.progress.pack(pady=10)
        
        # Log area
        log_frame = tk.Frame(self.root)
        log_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(
            log_frame,
            height=10,
            width=70,
            font=("Consolas", 9),
            yscrollcommand=scrollbar.set,
            bg="#f5f5f5",
            fg="#333"
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
        
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def update_status(self, text):
        self.status_label.config(text=text)
        self.root.update()
        
    def run_setup(self):
        self.progress.start(10)
        
        try:
            # Step 1: Create venv
            self.update_status("⚙️ Creating virtual environment...")
            self.log("Creating virtual environment...")
            
            venv_path = Path(".venv")
            if not venv_path.exists():
                result = subprocess.run(
                    [sys.executable, "-m", "venv", ".venv"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode != 0:
                    raise Exception(f"Failed to create venv: {result.stderr}")
                
                self.log("✓ Virtual environment created")
            else:
                self.log("✓ Virtual environment already exists")
            
            # Step 2: Get venv Python
            if sys.platform == "win32":
                venv_python = venv_path / "Scripts" / "python.exe"
            else:
                venv_python = venv_path / "bin" / "python"
            
            if not venv_python.exists():
                raise Exception("Venv Python not found")
            
            # Step 3: Upgrade pip
            self.update_status("⚙️ Upgrading pip...")
            self.log("Upgrading pip...")
            
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                capture_output=True,
                timeout=60
            )
            self.log("✓ Pip upgraded")
            
            # Step 4: Install PySide6
            self.update_status("⚙️ Installing PySide6...")
            self.log("Installing PySide6 (this may take 1-2 minutes)...")
            
            result = subprocess.run(
                [str(venv_python), "-m", "pip", "install", "PySide6"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to install PySide6: {result.stderr}")
            
            self.log("✓ PySide6 installed")
            
            # Step 5: Launch main setup wizard
            self.progress.stop()
            self.update_status("✅ Environment ready! Launching setup wizard...")
            self.log("\n✅ Bootstrap complete!")
            self.log("Launching main setup wizard...")
            
            self.root.after(1000, self._launch_main_setup)
            
        except Exception as e:
            self.progress.stop()
            self.update_status("❌ Setup failed!")
            self.log(f"\n❌ Error: {str(e)}")
            messagebox.showerror(
                "Bootstrap Failed",
                f"Failed to set up environment:\n\n{str(e)}\n\n"
                f"Please ensure:\n"
                f"- You have internet connection\n"
                f"- Python 3.8+ is installed\n"
                f"- You have write permissions in this directory"
            )
            self.root.after(2000, self.root.quit)
    
    def _launch_main_setup(self):
        """Launch the main PySide6 setup wizard"""
        try:
            venv_path = Path(".venv")
            if sys.platform == "win32":
                venv_python = venv_path / "Scripts" / "python.exe"
            else:
                venv_python = venv_path / "bin" / "python"
            
            # Close this bootstrap window
            self.root.destroy()
            
            # Launch main setup
            subprocess.Popen(
                [str(venv_python), "first_run_setup.py"],
                cwd=os.getcwd()
            )
            
        except Exception as e:
            messagebox.showerror("Launch Failed", f"Failed to launch setup wizard:\n{str(e)}")
    
    def start(self):
        # Run setup in background thread
        setup_thread = threading.Thread(target=self.run_setup, daemon=True)
        setup_thread.start()
        
        # Start GUI
        self.root.mainloop()


if __name__ == "__main__":
    app = BootstrapSetup()
    app.start()

