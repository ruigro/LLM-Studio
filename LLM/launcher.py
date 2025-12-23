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
import socket
import traceback
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

def find_venv_python():
    """Find the virtual environment Python executable"""
    # Check for venv in the same directory as this script
    venv_python = APP_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    
    # Check for venv in parent directory
    venv_python = APP_DIR.parent / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    
    # Fallback to current Python
    return sys.executable

def create_streamlit_credentials():
    """Create Streamlit credentials.toml file to skip email prompt"""
    try:
        streamlit_config_dir = Path.home() / ".streamlit"
        streamlit_config_dir.mkdir(exist_ok=True)
        
        credentials_file = streamlit_config_dir / "credentials.toml"
        
        # Create credentials file with empty email to skip prompt
        if not credentials_file.exists():
            credentials_content = """[general]
email = ""
"""
            with open(credentials_file, 'w') as f:
                f.write(credentials_content)
            print(f"Created Streamlit credentials file at {credentials_file}")
        else:
            # Update existing file to ensure email is empty
            try:
                with open(credentials_file, 'r') as f:
                    content = f.read()
                # Check if email is already set
                if 'email' not in content or 'email = ""' not in content:
                    # Update or add email field
                    if '[general]' in content:
                        # Replace existing email line
                        lines = content.split('\n')
                        updated_lines = []
                        email_found = False
                        for line in lines:
                            if line.strip().startswith('email'):
                                updated_lines.append('email = ""')
                                email_found = True
                            else:
                                updated_lines.append(line)
                        if not email_found:
                            # Add email after [general]
                            for i, line in enumerate(updated_lines):
                                if line.strip() == '[general]':
                                    updated_lines.insert(i + 1, 'email = ""')
                                    break
                        content = '\n'.join(updated_lines)
                    else:
                        # Add [general] section with email
                        content = '[general]\nemail = ""\n' + content
                    
                    with open(credentials_file, 'w') as f:
                        f.write(content)
            except Exception as e:
                print(f"Warning: Could not update credentials file: {e}")
    except Exception as e:
        print(f"Warning: Could not create Streamlit credentials file: {e}")

def kill_process_on_port(port):
    """Kill any process using the specified port"""
    try:
        if platform.system() == "Windows":
            # Find process using the port
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/PID", pid],
                                capture_output=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                            )
                            print(f"Killed process {pid} on port {port}")
                        except Exception:
                            pass
    except Exception as e:
        print(f"Warning: Could not check for processes on port {port}: {e}")

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
    try:
        # Change to app directory
        os.chdir(APP_DIR)
        
        # Find gui.py
        gui_file = APP_DIR / "gui.py"
        if not gui_file.exists():
            print(f"Error: gui.py not found at {gui_file}")
            print(f"Current directory: {os.getcwd()}")
            input("Press Enter to exit...")
            return False
    except Exception as e:
        print(f"Error changing to app directory: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        return False
    
    # Kill any existing processes on port 8501
    print("Checking for existing processes on port 8501...")
    kill_process_on_port(8501)
    time.sleep(1)  # Give it a moment to clean up
    
    # Create Streamlit credentials to skip email prompt
    print("Setting up Streamlit configuration...")
    create_streamlit_credentials()
    
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
    print("Please wait 5-10 seconds for the server to start...")
    print("The GUI will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the server.")
    print()
    
    # Find the best Python executable to use
    python_exe = find_venv_python()
    print(f"Using Python: {python_exe}")
    
    # Function to check if Streamlit is ready
    def is_streamlit_ready(port=8501, max_attempts=20, delay=1):
        """Check if Streamlit server is ready by trying to connect"""
        for attempt in range(max_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                if result == 0:
                    return True
            except Exception:
                pass
            time.sleep(delay)
        return False
    
    # Open browser after Streamlit is ready
    def open_browser():
        print("Waiting for Streamlit server to start...")
        if is_streamlit_ready():
            print("Streamlit server is ready!")
            try:
                webbrowser.open("http://localhost:8501")
                print("\n✓ Browser opened successfully!")
            except Exception as e:
                print(f"\nCould not open browser automatically: {e}")
                print("Please open http://localhost:8501 manually in your browser.")
        else:
            print("\n⚠ Streamlit server did not start in time.")
            print("Please check the console for errors and try opening http://localhost:8501 manually.")
    
    # Start browser opener in background thread
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Launch Streamlit
    process = None
    try:
        # Set environment variables to skip Streamlit's first-run prompts
        env = os.environ.copy()
        env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        env['STREAMLIT_SERVER_HEADLESS'] = 'false'
        # Skip email prompt by setting to empty
        env['STREAMLIT_EMAIL'] = ''
        
        # If running from bundle, may need to set paths
        if getattr(sys, 'frozen', False):
            # Add bundle path to Python path
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = str(BASE_DIR) + os.pathsep + env['PYTHONPATH']
            else:
                env['PYTHONPATH'] = str(BASE_DIR)
        
        # Use streamlit command with the venv Python
        # Add flags to skip prompts and disable browser auto-open (we'll open it manually)
        cmd = [
            python_exe, "-m", "streamlit", "run", str(gui_file),
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--server.runOnSave", "true"
        ]
        
        # Use Popen for better control and error handling
        # Let Streamlit output directly to console (stdout/stderr=None)
        # Redirect stdin to DEVNULL to prevent blocking prompts
        process = subprocess.Popen(
            cmd,
            cwd=str(APP_DIR),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=None,  # Let Streamlit output directly to console
            stderr=None   # Let Streamlit output directly to console
        )
        
        # Wait for process to complete (or until interrupted)
        try:
            return_code = process.wait()
            if return_code != 0:
                print(f"\nStreamlit process exited with code {return_code}")
                return False
            
        except KeyboardInterrupt:
            print("\nShutting down server...")
            if process:
                try:
                    process.terminate()
                    # Wait a bit for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                except Exception:
                    pass
            return True
    
    except FileNotFoundError as e:
        print(f"\nError: Python executable not found at {python_exe}")
        print(f"Details: {e}")
        print("Make sure the virtual environment is set up correctly.")
        print("Try running: python -m venv .venv")
        input("\nPress Enter to exit...")
        return False
    except Exception as e:
        print(f"\nError starting Streamlit: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Check if port 8501 is available")
        print("3. Try running manually: streamlit run gui.py")
        print(f"4. Python executable used: {python_exe}")
        input("\nPress Enter to exit...")
        return False
    finally:
        # Ensure process is cleaned up
        if process and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
    
    return True

def main():
    """Main entry point"""
    try:
        # Check system
        if not check_system():
            input("Press Enter to exit...")
            sys.exit(1)
        
        # Launch Streamlit
        success = launch_streamlit()
        
        if not success:
            print("\nFailed to launch Streamlit. Check the error messages above.")
            input("Press Enter to exit...")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        print("\nPlease report this error if it persists.")
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()

