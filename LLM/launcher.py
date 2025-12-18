#!/usr/bin/env python3
"""
Launcher for LLM Fine-tuning Studio
Handles application startup, system checks, and browser opening
"""

import os
import sys
import subprocess
import webbrowser
import time
import platform
from pathlib import Path

# Check if running from PyInstaller bundle
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    BASE_DIR = Path(sys._MEIPASS)
    APP_DIR = Path(sys.executable).parent
else:
    # Running as script
    BASE_DIR = Path(__file__).parent.absolute()
    APP_DIR = BASE_DIR

def check_system():
    """Check system requirements and show status"""
    print("=" * 60)
    print("LLM Fine-tuning Studio - System Check")
    print("=" * 60)
    print()
    
    # Check Python
    print(f"Python: {sys.version.split()[0]} at {sys.executable}")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA: Available (Version {torch.version.cuda})")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA: Not available (CPU mode)")
    except ImportError:
        print("PyTorch: Not installed")
        print("  Warning: Some features may not work")
    
    # Check Streamlit
    try:
        import streamlit
        print(f"Streamlit: {streamlit.__version__}")
    except ImportError:
        print("Streamlit: Not installed")
        print("  Error: Cannot start GUI without Streamlit")
        return False
    
    print()
    print("=" * 60)
    print()
    
    return True

def launch_streamlit():
    """Launch Streamlit server"""
    # Change to app directory
    os.chdir(APP_DIR)
    
    # Find gui.py
    gui_file = APP_DIR / "gui.py"
    if not gui_file.exists():
        print(f"Error: gui.py not found at {gui_file}")
        print(f"Current directory: {os.getcwd()}")
        input("Press Enter to exit...")
        return False
    
    # Set up Streamlit config if needed
    streamlit_config_dir = Path.home() / ".streamlit"
    streamlit_config_dir.mkdir(exist_ok=True)
    
    config_file = streamlit_config_dir / "config.toml"
    if not config_file.exists():
        # Create default config
        config_content = """[server]
headless = false
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
"""
        with open(config_file, 'w') as f:
            f.write(config_content)
    
    print("Starting Streamlit server...")
    print(f"GUI file: {gui_file}")
    print()
    print("The GUI will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the server.")
    print()
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)  # Wait for server to start
        try:
            webbrowser.open("http://localhost:8501")
        except Exception:
            pass
    
    # Start browser opener in background thread
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Launch Streamlit
    try:
        # Use streamlit command
        cmd = [sys.executable, "-m", "streamlit", "run", str(gui_file), "--server.headless", "false"]
        
        # If running from bundle, may need to set paths
        if getattr(sys, 'frozen', False):
            # Add bundle path to Python path
            env = os.environ.copy()
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = str(BASE_DIR) + os.pathsep + env['PYTHONPATH']
            else:
                env['PYTHONPATH'] = str(BASE_DIR)
            
            subprocess.run(cmd, env=env)
        else:
            subprocess.run(cmd)
    
    except KeyboardInterrupt:
        print("\nShutting down server...")
        return True
    except Exception as e:
        print(f"Error starting Streamlit: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Check if port 8501 is available")
        print("3. Try running manually: streamlit run gui.py")
        input("\nPress Enter to exit...")
        return False
    
    return True

def main():
    """Main entry point"""
    # Check system
    if not check_system():
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Launch Streamlit
    success = launch_streamlit()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

