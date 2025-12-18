#!/usr/bin/env python3
"""
Streamlit GUI for LLM Fine-tuning
Run with: streamlit run gui.py
"""
import streamlit as st
import os
import json
import subprocess
import threading
import time
from pathlib import Path
import pandas as pd

# Import system detector
try:
    from system_detector import SystemDetector
    SYSTEM_DETECTOR_AVAILABLE = True
except ImportError:
    SYSTEM_DETECTOR_AVAILABLE = False
    SystemDetector = None

# Try to import huggingface_hub for model downloads
try:
    from huggingface_hub import snapshot_download, list_models, HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    snapshot_download = None
    list_models = None
    HfApi = None

# Try to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    torch = None
    TORCH_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="LLM Fine-tuning Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Hide Streamlit deploy button and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    button[title="View app source"] {display: none;}
    button[title="Deploy this app"] {display: none;}
    button[kind="header"] {display: none;}
    
    /* Custom Navbar Styles */
    .navbar {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .navbar-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .nav-item {
        color: white;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        font-weight: 500;
        cursor: pointer;
        border: 2px solid transparent;
    }
    .nav-item:hover {
        background-color: rgba(255,255,255,0.2);
        transform: translateY(-2px);
        border-color: rgba(255,255,255,0.3);
    }
    .nav-item.active {
        background-color: rgba(255,255,255,0.3);
        border-color: white;
        font-weight: bold;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 0.5rem;
        padding: 0.5rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 0.4rem 0.6rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.3rem 0;
        display: inline-block;
        width: auto;
        max-width: 100%;
        line-height: 1.3;
        font-size: 0.9em;
    }
    .success-box {
        background-color: #d4edda;
        padding: 0.4rem 0.6rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.3rem 0;
        display: inline-block;
        width: auto;
        max-width: 100%;
        line-height: 1.3;
        font-size: 0.9em;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
        transition: all 0.3s ease;
    }
    .model-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'training_status' not in st.session_state:
    st.session_state.training_status = "idle"
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'training_process' not in st.session_state:
    st.session_state.training_process = None
if 'training_thread' not in st.session_state:
    st.session_state.training_thread = None
if 'gpu_available' not in st.session_state:
    if TORCH_AVAILABLE and torch is not None:
        try:
            st.session_state.gpu_available = torch.cuda.is_available()
        except:
            st.session_state.gpu_available = False
    else:
        st.session_state.gpu_available = False

if 'gpu_name' not in st.session_state:
    if st.session_state.gpu_available and TORCH_AVAILABLE and torch is not None:
        try:
            st.session_state.gpu_name = torch.cuda.get_device_name(0)
        except:
            st.session_state.gpu_name = "CPU Only"
    else:
        st.session_state.gpu_name = "CPU Only"

if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home"

if 'downloaded_models' not in st.session_state:
    st.session_state.downloaded_models = []

if 'downloading_models' not in st.session_state:
    st.session_state.downloading_models = {}

# Model presets with metadata
MODEL_PRESETS = {
    "Llama 3.2 3B Instruct (4-bit)": {
        "id": "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit",
        "size": "~2.5 GB",
        "description": "Fast 3B parameter model, great for quick experiments",
        "tags": ["llama", "3b", "instruct", "4-bit"]
    },
    "Llama 3.2 1B Instruct (4-bit)": {
        "id": "unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit",
        "size": "~800 MB",
        "description": "Ultra-lightweight 1B model, fastest training",
        "tags": ["llama", "1b", "instruct", "4-bit"]
    },
    "Llama 3.1 8B Instruct (4-bit)": {
        "id": "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit",
        "size": "~5 GB",
        "description": "Powerful 8B model with better performance",
        "tags": ["llama", "8b", "instruct", "4-bit"]
    },
    "Mistral 7B Instruct (4-bit)": {
        "id": "unsloth/mistral-7b-instruct-bnb-4bit",
        "size": "~4.5 GB",
        "description": "High-quality Mistral model, excellent for instruction following",
        "tags": ["mistral", "7b", "instruct", "4-bit"]
    },
    "Qwen 2.5 3B Instruct (4-bit)": {
        "id": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "size": "~2.5 GB",
        "description": "Qwen 2.5 model with strong multilingual capabilities",
        "tags": ["qwen", "3b", "instruct", "4-bit", "multilingual"]
    },
    "Qwen 2.5 7B Instruct (4-bit)": {
        "id": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "size": "~4.5 GB",
        "description": "Larger Qwen model with enhanced performance",
        "tags": ["qwen", "7b", "instruct", "4-bit", "multilingual"]
    },
    "Custom Model": {
        "id": None,
        "size": "Varies",
        "description": "Enter a custom Hugging Face model ID",
        "tags": ["custom"]
    }
}

def load_trained_models():
    """Load list of trained model checkpoints"""
    output_dir = "./fine_tuned_adapter"
    models = []
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and "Checkpoint" in item:
                models.append(item)
    return sorted(models, reverse=True)

def load_downloaded_models():
    """Load list of downloaded models"""
    models_dir = "./models"
    downloaded = []
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                # Check if it's a valid model directory (has config.json)
                config_path = os.path.join(item_path, "config.json")
                if os.path.exists(config_path):
                    downloaded.append(item)
    return sorted(downloaded)

def is_model_downloaded(model_id):
    """Check if a model is already downloaded"""
    if not model_id:
        return False
    models_dir = "./models"
    model_name = model_id.replace("/", "_")
    model_path = os.path.join(models_dir, model_name)
    return os.path.exists(model_path) and os.path.isdir(model_path)

def download_model(model_id, progress_callback=None):
    """Download a model from Hugging Face"""
    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub is not installed")
    
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_name = model_id.replace("/", "_")
    model_path = os.path.join(models_dir, model_name)
    
    if os.path.exists(model_path):
        return model_path
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        return model_path
    except Exception as e:
        raise Exception(f"Failed to download model: {str(e)}")

def search_huggingface_models(query, limit=20):
    """Search for models on Hugging Face Hub"""
    if not HF_HUB_AVAILABLE:
        return []
    
    try:
        api = HfApi()
        models = api.list_models(
            search=query,
            sort="downloads",
            direction=-1,
            limit=limit
        )
        results = []
        for m in models:
            results.append({
                "id": m.id,
                "downloads": getattr(m, 'downloads', 0) or 0,
                "tags": getattr(m, 'tags', []) or [],
                "author": getattr(m, 'author', '') or '',
                "likes": getattr(m, 'likes', 0) or 0
            })
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def stop_training():
    """Stop the training process"""
    process = st.session_state.training_process
    if process is not None:
        try:
            # Try graceful termination first
            process.terminate()
            time.sleep(1)
            
            # If still running, force kill
            if process.poll() is None:
                process.kill()
                time.sleep(0.5)
            
            st.session_state.training_logs.append("=" * 60)
            st.session_state.training_logs.append("🛑 Training stopped by user")
            st.session_state.training_status = "idle"
            st.session_state.training_process = None
            st.success("✅ Training stopped successfully")
        except Exception as e:
            st.error(f"❌ Error stopping training: {e}")
            # Try alternative method - kill by process name
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info['cmdline']
                        if cmdline and 'finetune.py' in ' '.join(cmdline):
                            proc.kill()
                            st.success("✅ Training process killed")
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
            except ImportError:
                # psutil not available, try Windows taskkill
                import platform
                if platform.system() == "Windows":
                    try:
                        # Kill Python processes running finetune.py
                        subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq finetune.py*"], 
                                     capture_output=True, timeout=5)
                    except:
                        pass
                st.warning("⚠️ Could not stop process automatically. Please kill it manually from Task Manager.")
    else:
        st.warning("⚠️ No training process found to stop")
        st.session_state.training_status = "idle"

def render_navbar():
    """Render custom navigation navbar"""
    pages = [
        ("🏠", "Home", "🏠 Home"),
        ("🎯", "Train", "🎯 Train Model"),
        ("📥", "Download", "📥 Download Models"),
        ("🧪", "Test", "🧪 Test Model"),
        ("✅", "Validate", "✅ Validate Model"),
        ("📊", "History", "📊 Training History")
    ]
    
    # Add custom CSS for active navbar button styling
    st.markdown("""
    <style>
    /* Style active navbar buttons */
    div[data-testid*="stButton"] > button[kind="primary"] {
        background: linear-gradient(90deg, #ff7f0e 0%, #ff4500 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: 2px solid #ff7f0e !important;
        box-shadow: 0 4px 8px rgba(255, 127, 14, 0.3) !important;
        transform: scale(1.02);
    }
    /* Style inactive navbar buttons */
    div[data-testid*="stButton"] > button[kind="secondary"] {
        background-color: #f0f0f0 !important;
        color: #666 !important;
        border: 2px solid #ddd !important;
    }
    div[data-testid*="stButton"] > button[kind="secondary"]:hover {
        background-color: #e0e0e0 !important;
        border-color: #1f77b4 !important;
        color: #1f77b4 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create horizontal navigation using columns
    cols = st.columns(len(pages))
    for idx, (icon, label, page_id) in enumerate(pages):
        with cols[idx]:
            is_active = st.session_state.current_page == page_id
            button_type = "primary" if is_active else "secondary"
            button_label = f"{icon} {label}"
            
            if st.button(button_label, key=f"nav_{page_id}", use_container_width=True, type=button_type):
                st.session_state.current_page = page_id
                st.rerun()

def run_training_safe(config):
    """Wrapper to catch and log any thread errors"""
    # Create error file immediately in case thread crashes
    error_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_error.txt")
    try:
        run_training(config)
    except Exception as e:
        # Write error to file immediately so it's visible
        try:
            with open(error_file, 'w', encoding='utf-8') as f:
                import traceback
                f.write(f"Thread error at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {str(e)}\n")
                f.write("=" * 60 + "\n")
                f.write(traceback.format_exc())
        except:
            pass
        # Also try to update session state
        try:
            st.session_state.training_status = "failed"
            if 'training_logs' in st.session_state:
                st.session_state.training_logs.append(f"❌ Thread error: {str(e)}")
        except:
            pass

def run_training(config):
    """Run training in background thread"""
    import sys
    
    # Create log file IMMEDIATELY at the start so errors can be logged
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.txt")
    error_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_error.txt")
    
    # Open log file immediately and write startup info
    try:
        log_file = open(log_file_path, 'w', encoding='utf-8', buffering=1)
        log_file.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 60 + "\n")
        log_file.flush()
    except Exception as e:
        # If we can't even create the log file, write to error file
        try:
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Failed to create log file: {str(e)}\n")
        except:
            pass
        return
    
    # Use Python with -u flag for unbuffered output
    python_exe = sys.executable
    cmd = [
        python_exe, "-u", "finetune.py",  # -u flag for unbuffered output
        "--model-name", config["model_name"],
        "--data-path", config["data_path"],
        "--output-dir", config["output_dir"],
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lora-r", str(config["lora_r"]),
        "--lora-alpha", str(config["lora_alpha"]),
        "--lora-dropout", str(config["lora_dropout"]),
        "--grad-accum", str(config["grad_accum"]),
        "--max-seq-length", str(config["max_seq_length"]),
    ]
    
    if config.get("max_examples"):
        cmd.extend(["--max-examples", str(config["max_examples"])])
    
    # Use unbuffered output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # Log the command being run
    cmd_str = " ".join(cmd)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Write to log file immediately (file already open)
    log_file.write(f"Command: {cmd_str}\n")
    log_file.write(f"Working directory: {working_dir}\n")
    log_file.write(f"Python executable: {python_exe}\n")
    log_file.flush()
    
    # Note: We don't update session_state from background thread - UI reads from log file
    
    try:
        # Verify finetune.py exists
        finetune_path = os.path.join(working_dir, "finetune.py")
        if not os.path.exists(finetune_path):
            error_msg = f"❌ Error: finetune.py not found at {finetune_path}"
            log_file.write(error_msg + "\n")
            log_file.flush()
            log_file.close()
            return
        
        # Verify dataset exists
        if not os.path.exists(config["data_path"]):
            error_msg = f"❌ Error: Dataset not found at {config['data_path']}"
            log_file.write(error_msg + "\n")
            log_file.flush()
            log_file.close()
            return
        
        # Write more info to log file
        log_file.write(f"Finetune script: {finetune_path}\n")
        log_file.write(f"Dataset: {config['data_path']}\n")
        log_file.write(f"Model: {config['model_name']}\n")
        log_file.write(f"Epochs: {config['epochs']}\n")
        log_file.flush()
        
        # Try to start the process
        try:
            log_file.write("Attempting to start subprocess...\n")
            log_file.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                cwd=working_dir,
                env=env
            )
            
            # Verify process was created
            if process is None:
                error_msg = "❌ Failed to create subprocess - process is None"
                log_file.write(error_msg + "\n")
                log_file.flush()
                log_file.close()
                return
            
            pid_msg = f"Process started with PID: {process.pid}"
            log_file.write(pid_msg + "\n")
            log_file.flush()
            
            # Note: We can't update session_state from background thread reliably
            # Process tracking will be done by UI reading from log file
            
            # Wait a moment to see if process crashes immediately
            time.sleep(0.2)  # Reduced from 0.5s for faster detection
            if process.poll() is not None:
                # Process crashed immediately
                return_code = process.returncode
                error_msg = f"❌ Process crashed immediately with exit code {return_code}"
                log_file.write(error_msg + "\n")
                # Try to read any error output
                try:
                    remaining = process.stdout.read()
                    if remaining:
                        log_file.write("Error output:\n")
                        for line in remaining.splitlines():
                            if line.strip():
                                log_file.write(line + "\n")
                except:
                    pass
                log_file.flush()
                log_file.close()
                return
            
        except FileNotFoundError as e:
            error_msg = f"❌ Error: Python executable not found: {python_exe}"
            log_file.write(error_msg + "\n")
            log_file.write(f"Error details: {str(e)}\n")
            log_file.flush()
            log_file.close()
            return
        except Exception as e:
            error_msg = f"❌ Error starting process: {str(e)}"
            import traceback
            tb = traceback.format_exc()
            log_file.write(error_msg + "\n")
            log_file.write(tb + "\n")
            log_file.flush()
            log_file.close()
            return
        
        # Verify process is still running before starting output reading
        if process is None or process.poll() is not None:
            error_msg = "❌ Process is None or already finished before output reading started"
            log_file.write(error_msg + "\n")
            log_file.flush()
            log_file.close()
            return
        
        log_file.write("Process is running, starting output reader...\n")
        log_file.flush()
        
        # Read output line by line in real-time using a separate thread
        import queue
        
        output_queue = queue.Queue()
        stop_reading = threading.Event()
        
        def read_output():
            """Read process output in a separate thread"""
            try:
                while not stop_reading.is_set():
                    output = process.stdout.readline()
                    if output:
                        output_queue.put(output.rstrip())
                    elif process.poll() is not None:
                        # Process finished, read remaining
                        try:
                            remaining = process.stdout.read()
                            if remaining:
                                for line in remaining.splitlines():
                                    if line.strip():
                                        output_queue.put(line.rstrip())
                        except:
                            pass
                        output_queue.put(None)  # Signal end
                        break
            except Exception as e:
                output_queue.put(f"ERROR reading output: {e}")
                output_queue.put(None)
        
        # Start output reader thread
        reader_thread = threading.Thread(target=read_output, daemon=True)
        reader_thread.start()
        
        # Process output from queue with timeout
        timeout_count = 0
        max_timeout = 60  # Wait up to 60 seconds for first output
        
        while True:
            try:
                # Wait for output with timeout
                line = output_queue.get(timeout=1.0)
                if line is None:
                    break  # End of output
                if line:
                    log_file.write(line + "\n")
                    log_file.flush()
                    timeout_count = 0  # Reset timeout on successful read
            except queue.Empty:
                # Check if process is still running
                if process.poll() is not None:
                    # Process finished, try to get remaining items
                    try:
                        while True:
                            line = output_queue.get_nowait()
                            if line is None:
                                break
                            if line:
                                log_file.write(line + "\n")
                                log_file.flush()
                    except queue.Empty:
                        pass
                    break
                
                # If no output for a while, add a status message
                timeout_count += 1
                if timeout_count == 5:  # After 5 seconds
                    log_file.write("⏳ Waiting for output from training process...\n")
                    log_file.flush()
                elif timeout_count >= max_timeout:
                    log_file.write("⚠️ No output received for 60 seconds. Process may be stuck.\n")
                    log_file.flush()
                    break
        
        stop_reading.set()
        
        log_file.close()
        
        # Get return code
        return_code = process.poll()
        if return_code is None:
            return_code = 0
        
        # Add final status message to log file
        log_file.write("=" * 60 + "\n")
        if return_code == 0:
            log_file.write("✅ Training completed successfully!\n")
        else:
            log_file.write(f"❌ Training failed with exit code {return_code}\n")
        log_file.flush()
        log_file.close()
    
    except Exception as e:
        error_msg = f"❌ Error starting training: {str(e)}"
        st.session_state.training_logs.append(error_msg)
        st.session_state.training_status = "failed"
        st.session_state.training_process = None
        import traceback
        st.session_state.training_logs.append(traceback.format_exc())
        try:
            log_file.close()
        except:
            pass

def main():
    st.markdown('<h1 class="main-header">🤖 LLM Fine-tuning Studio</h1>', unsafe_allow_html=True)
    
    # Render custom navbar
    render_navbar()
    
    # Sidebar with system info (keep for reference, but navigation is in navbar)
    with st.sidebar:
        st.markdown("### 🖥️ System Information")
        
        # Enhanced system detection if available
        if SYSTEM_DETECTOR_AVAILABLE and SystemDetector:
            try:
                detector = SystemDetector()
                detection_results = detector.detect_all()
                
                # Python info
                python_info = detection_results.get("python", {})
                if python_info.get("found"):
                    st.caption(f"🐍 Python {python_info.get('version', 'Unknown')}")
                
                # PyTorch info
                pytorch_info = detection_results.get("pytorch", {})
                if pytorch_info.get("found"):
                    pytorch_version = pytorch_info.get("version", "Unknown")
                    if pytorch_info.get("cuda_available"):
                        cuda_version = pytorch_info.get("cuda_version", "Unknown")
                        st.caption(f"🔥 PyTorch {pytorch_version} (CUDA {cuda_version})")
                    else:
                        st.caption(f"🔥 PyTorch {pytorch_version} (CPU)")
                else:
                    st.caption("🔥 PyTorch: Not installed")
                
                # Hardware info
                hardware_info = detection_results.get("hardware", {})
                if hardware_info.get("ram_gb"):
                    st.caption(f"💾 RAM: {hardware_info['ram_gb']:.1f} GB")
                
                st.divider()
            except Exception as e:
                pass
        
        # GPU/CPU Status
        if not TORCH_AVAILABLE:
            st.markdown(f"""
            <div class="error-box">
                <strong>⚠️ PyTorch Error</strong><br>
                GPU detection unavailable. Training will use CPU.<br>
                <small>Error: {TORCH_ERROR[:100] if 'TORCH_ERROR' in globals() else 'Unknown'}</small>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.gpu_available:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ GPU Available</strong><br>
                {st.session_state.gpu_name}<br>
                CUDA Version: {torch.version.cuda if torch and torch.version.cuda else 'N/A'}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ CPU Mode</strong><br>
                Training will use CPU (slower)
            </div>
            """, unsafe_allow_html=True)
        
        # Memory info
        if st.session_state.gpu_available and TORCH_AVAILABLE and torch is not None:
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.metric("GPU Memory", f"{gpu_memory:.1f} GB")
            except:
                pass
        
        st.divider()
        
        # Quick stats
        trained_models = load_trained_models()
        st.metric("Models Trained", len(trained_models))
        
        downloaded_models = load_downloaded_models()
        st.metric("Models Downloaded", len(downloaded_models))
        
        if st.session_state.training_status == "training":
            st.markdown("""
            <div class="info-box">
                <strong>🔄 Training in Progress</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Use current_page from session state
    page = st.session_state.current_page
    
    if page == "🏠 Home":
        st.header("Welcome to LLM Fine-tuning Studio")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🚀 Features
            
            This application provides a beautiful, user-friendly interface to:
            
            - **🎯 Train Models**: Select from popular pre-trained models and fine-tune them with your data
            - **📤 Upload Datasets**: Easy drag-and-drop for JSONL format datasets
            - **🧪 Test Models**: Interactive chat interface to test your fine-tuned models
            - **✅ Validate Performance**: Run validation tests and view detailed results
            - **📊 Track History**: View all your trained models and training logs
            
            ### 📋 Quick Start Guide
            
            1. **Prepare Your Dataset**: Create a JSONL file with format:
               ```json
               {"instruction": "Your instruction here", "output": "Expected output here"}
               ```
            
            2. **Go to Train Model**: Select a base model and upload your dataset
            
            3. **Configure Training**: Adjust epochs, batch size, and LoRA parameters
            
            4. **Start Training**: Click the train button and monitor progress
            
            5. **Test Your Model**: Use the Test Model tab to try your fine-tuned model
            """)
        
        with col2:
            st.markdown("### 📊 System Status")
            
            # Device info
            if not TORCH_AVAILABLE:
                st.error("⚠️ PyTorch not available - CPU mode only")
            elif st.session_state.gpu_available:
                st.success(f"✅ GPU: {st.session_state.gpu_name}")
                try:
                    if torch is not None:
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        st.metric("GPU Memory", f"{gpu_memory:.1f} GB")
                except:
                    pass
            else:
                st.warning("⚠️ CPU Mode")
            
            st.metric("Models Trained", len(trained_models))
            st.metric("Status", "Ready" if st.session_state.training_status == "idle" else "Training")
            
            st.markdown("### 💡 Tips")
            st.info("""
            - Use GPU for faster training
            - Start with fewer epochs for testing
            - Monitor training logs for progress
            - Test models before full validation
            """)
    
    elif page == "📥 Download Models":
        st.header("📥 Download Models")
        
        if not HF_HUB_AVAILABLE:
            st.markdown("""
            <div class="error-box">
                <strong>❌ Hugging Face Hub not available</strong><br>
                Please install: <code>pip install huggingface_hub</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            tab1, tab2 = st.tabs(["📋 Predefined Models", "🔍 Search Hugging Face"])
            
            with tab1:
                st.subheader("Popular Models for Fine-tuning")
                st.markdown("Select from our curated list of models optimized for fine-tuning:")
                
                downloaded_models = load_downloaded_models()
                
                for model_name, model_info in MODEL_PRESETS.items():
                    if model_info["id"] is None:
                        continue
                    
                    model_id = model_info["id"]
                    is_downloaded = is_model_downloaded(model_id)
                    
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            with st.container():
                                st.markdown(f"**{model_name}**")
                                st.caption(f"🆔 `{model_id}` | 📦 {model_info['size']}")
                                st.caption(f"📝 {model_info['description']}")
                                st.caption(f"🏷️ {', '.join(model_info['tags'])}")
                                st.markdown("---")
                        
                        with col2:
                            if is_downloaded:
                                st.success("✅ Downloaded")
                            else:
                                st.info("Not Downloaded")
                        
                        with col3:
                            download_key = f"download_{model_id}"
                            if model_id in st.session_state.downloading_models:
                                st.warning("Downloading...")
                            elif is_downloaded:
                                if st.button("🗑️ Remove", key=f"remove_{model_id}"):
                                    model_name_clean = model_id.replace("/", "_")
                                    model_path = f"./models/{model_name_clean}"
                                    import shutil
                                    try:
                                        shutil.rmtree(model_path)
                                        st.success(f"Removed {model_name}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                            else:
                                if st.button("📥 Download", key=download_key):
                                    st.session_state.downloading_models[model_id] = True
                                    st.rerun()
                        
                        if model_id in st.session_state.downloading_models:
                            with st.status(f"Downloading {model_name}...", expanded=True) as status:
                                try:
                                    progress_bar = st.progress(0)
                                    status.update(label=f"Downloading {model_id}...")
                                    
                                    def update_progress(current, total):
                                        if total > 0:
                                            progress = current / total
                                            progress_bar.progress(progress)
                                    
                                    download_path = download_model(model_id)
                                    progress_bar.progress(1.0)
                                    status.update(label=f"✅ Downloaded {model_name}!", state="complete")
                                    st.success(f"Model downloaded to: {download_path}")
                                    del st.session_state.downloading_models[model_id]
                                    time.sleep(2)
                                    st.rerun()
                                except Exception as e:
                                    status.update(label=f"❌ Download failed: {str(e)}", state="error")
                                    del st.session_state.downloading_models[model_id]
                                    st.error(f"Error downloading model: {str(e)}")
            
            with tab2:
                st.subheader("Search Hugging Face Hub")
                st.markdown("Search for models on Hugging Face Hub:")
                
                search_query = st.text_input("Search models", placeholder="e.g., llama instruct, mistral, qwen")
                
                if search_query:
                    with st.spinner("Searching..."):
                        results = search_huggingface_models(search_query, limit=20)
                    
                    if results:
                        st.success(f"Found {len(results)} models")
                        
                        for model in results:
                            model_id = model["id"]
                            is_downloaded = is_model_downloaded(model_id)
                            
                            with st.container():
                                col1, col2, col3 = st.columns([3, 1, 1])
                                
                                with col1:
                                    tags_list = model.get("tags", [])[:5] if model.get("tags") else []
                                    tags_display = ", ".join(tags_list) if tags_list else "No tags"
                                    downloads_count = model.get("downloads", 0) or 0
                                    
                                    # Use Streamlit components instead of raw HTML for better rendering
                                    with st.container():
                                        st.markdown(f"**{model_id}**")
                                        st.caption(f"📥 Downloads: {downloads_count:,}")
                                        if tags_list:
                                            st.caption(f"🏷️ Tags: {tags_display}")
                                        st.markdown("---")
                                
                                with col2:
                                    if is_downloaded:
                                        st.success("✅ Downloaded")
                                    else:
                                        st.info("Not Downloaded")
                                
                                with col3:
                                    download_key = f"download_hf_{model_id}"
                                    if model_id in st.session_state.downloading_models:
                                        st.warning("Downloading...")
                                    elif is_downloaded:
                                        if st.button("🗑️ Remove", key=f"remove_hf_{model_id}"):
                                            model_name_clean = model_id.replace("/", "_")
                                            model_path = f"./models/{model_name_clean}"
                                            import shutil
                                            try:
                                                shutil.rmtree(model_path)
                                                st.success(f"Removed {model_id}")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Error: {e}")
                                    else:
                                        if st.button("📥 Download", key=download_key):
                                            st.session_state.downloading_models[model_id] = True
                                            st.rerun()
                                
                                if model_id in st.session_state.downloading_models:
                                    with st.status(f"Downloading {model_id}...", expanded=True) as status:
                                        try:
                                            progress_bar = st.progress(0)
                                            status.update(label=f"Downloading {model_id}...")
                                            
                                            download_path = download_model(model_id)
                                            progress_bar.progress(1.0)
                                            status.update(label=f"✅ Downloaded {model_id}!", state="complete")
                                            st.success(f"Model downloaded to: {download_path}")
                                            del st.session_state.downloading_models[model_id]
                                            time.sleep(2)
                                            st.rerun()
                                        except Exception as e:
                                            status.update(label=f"❌ Download failed: {str(e)}", state="error")
                                            del st.session_state.downloading_models[model_id]
                                            st.error(f"Error downloading model: {str(e)}")
                    else:
                        st.info("No models found. Try a different search query.")
                else:
                    st.info("Enter a search query to find models on Hugging Face Hub.")
    
    elif page == "🎯 Train Model":
        st.header("🎯 Train a Fine-tuned Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Model Configuration")
            
            # Model selection - include downloaded models
            downloaded_models = load_downloaded_models()
            
            # Build model options list
            model_options = list(MODEL_PRESETS.keys())
            if downloaded_models:
                model_options.append("───────── Downloaded Models ─────────")
                model_options.extend([f"📥 {m}" for m in downloaded_models])
            model_options.append("Custom Model")
            
            model_preset = st.selectbox("Select Base Model", model_options)
            model_name = None
            
            if model_preset == "Custom Model":
                model_name = st.text_input("Enter Model Name/Path", value="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit")
            elif model_preset.startswith("📥 "):
                # Downloaded model selected - use local path
                model_dir_name = model_preset.replace("📥 ", "")
                model_name = os.path.abspath(f"./models/{model_dir_name}")
                # Also get the original model ID if available (for display)
                original_id = model_dir_name.replace("_", "/")
                if not os.path.exists(model_name):
                    st.error(f"Model not found at {model_name}")
                    model_name = None
                else:
                    st.success(f"**Downloaded:** `{original_id}` | Local: `{os.path.basename(model_name)}`")
            elif model_preset == "───────── Downloaded Models ─────────":
                st.info("Please select a model from the list above.")
            else:
                model_info = MODEL_PRESETS[model_preset]
                model_name = model_info["id"]
                st.info(f"**Selected:** `{model_name}` | {model_info['size']}")
            
            # Dataset upload
            st.subheader("📤 Dataset Upload")
            uploaded_file = st.file_uploader(
                "Upload Training Dataset (JSONL format)",
                type=["jsonl", "json"],
                help="Each line should be: {\"instruction\": \"...\", \"output\": \"...\"}"
            )
            
            if uploaded_file:
                # Save uploaded file
                data_path = "train_data.jsonl"
                with open(data_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Preview dataset
                st.markdown(f"""
                <div class="success-box">
                    ✅ Dataset uploaded: <strong>{uploaded_file.name}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Show preview
                with st.expander("📖 Preview Dataset (first 5 examples)", expanded=True):
                    with open(data_path, "r") as f:
                        lines = f.readlines()[:5]
                        preview_data = []
                        for line in lines:
                            try:
                                preview_data.append(json.loads(line))
                            except:
                                pass
                        if preview_data:
                            st.json(preview_data)
                        else:
                            st.warning("Could not parse dataset. Please check format.")
                
                # Dataset stats
                with open(data_path, "r") as f:
                    total_lines = sum(1 for _ in f)
                st.metric("Total Examples", total_lines)
            else:
                data_path = st.text_input("Or enter path to existing dataset", value="train_data.jsonl")
                if os.path.exists(data_path):
                    st.success(f"✅ Found dataset: {data_path}")
                    with open(data_path, "r") as f:
                        total_lines = sum(1 for _ in f)
                    st.metric("Total Examples", total_lines)
        
        with col2:
            st.subheader("⚙️ Training Parameters")
            
            epochs = st.slider("Epochs", 1, 10, 3, help="Number of training epochs")
            batch_size = st.slider("Batch Size", 1, 8, 1, help="Training batch size per device")
            
            st.markdown("### 🎛️ LoRA Parameters")
            lora_r = st.slider("LoRA R", 4, 64, 8, step=4, help="LoRA rank")
            lora_alpha = st.slider("LoRA Alpha", 8, 128, 16, step=8, help="LoRA alpha scaling")
            lora_dropout = st.slider("LoRA Dropout", 0.0, 0.5, 0.05, step=0.01, help="LoRA dropout rate")
            
            st.markdown("### 🔧 Advanced")
            max_seq_length = st.number_input("Max Sequence Length", 512, 4096, 2048, step=256)
            grad_accum = st.slider("Gradient Accumulation", 1, 32, 8, help="Gradient accumulation steps")
            max_examples = st.number_input("Max Examples (for testing)", 0, 10000, 0, 
                                         help="0 = use all examples. Limit for quick test runs.")
            
            # Device info
            st.markdown("### 💻 Device")
            if not TORCH_AVAILABLE:
                st.error("⚠️ PyTorch error - Training will use CPU")
            elif st.session_state.gpu_available:
                st.success(f"✅ Training on: {st.session_state.gpu_name}")
            else:
                st.warning("⚠️ Training on: CPU")
        
        # Training button and status
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            train_button = st.button("🚀 Start Training", type="primary", use_container_width=True)
        
        if train_button:
            if model_name is None:
                st.markdown("""
                <div class="error-box">
                    ❌ Please select a valid model.
                </div>
                """, unsafe_allow_html=True)
            elif not os.path.exists(data_path):
                st.markdown(f"""
                <div class="error-box">
                    ❌ Dataset not found: <code>{data_path}</code>
                </div>
                """, unsafe_allow_html=True)
            else:
                config = {
                    "model_name": model_name,
                    "data_path": data_path,
                    "output_dir": "./fine_tuned_adapter",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "grad_accum": grad_accum,
                    "max_seq_length": max_seq_length,
                    "max_examples": max_examples if max_examples > 0 else None,
                }
                
                # Check if training is already running
                if st.session_state.training_status == "training":
                    st.warning("⚠️ Training is already in progress. Please stop it first.")
                else:
                    # Pre-flight checks
                    import sys
                    python_exe = sys.executable
                    finetune_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune.py")
                    
                    # Verify files exist
                    checks_passed = True
                    if not os.path.exists(finetune_path):
                        st.error(f"❌ finetune.py not found at: {finetune_path}")
                        checks_passed = False
                    if not os.path.exists(data_path):
                        st.error(f"❌ Dataset not found at: {data_path}")
                        checks_passed = False
                    if not os.path.exists(python_exe):
                        st.error(f"❌ Python executable not found at: {python_exe}")
                        checks_passed = False
                    
                    if not checks_passed:
                        st.stop()
                    
                    st.session_state.training_status = "training"
                    st.session_state.training_logs = []
                    st.session_state.training_logs.append("=" * 60)
                    st.session_state.training_logs.append("🚀 Starting training...")
                    st.session_state.training_logs.append(f"Python: {python_exe}")
                    st.session_state.training_logs.append(f"Script: {finetune_path}")
                    st.session_state.training_logs.append(f"Model: {model_name}")
                    st.session_state.training_logs.append(f"Dataset: {data_path}")
                    st.session_state.training_logs.append(f"Epochs: {epochs}")
                    st.session_state.training_logs.append(f"Device: {'GPU' if st.session_state.gpu_available else 'CPU'}")
                    st.session_state.training_logs.append("=" * 60)
                    
                    # Write debug info before starting thread
                    debug_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_debug.txt")
                    try:
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(f"Starting training thread at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Config: {config}\n")
                            f.write(f"Python: {sys.executable}\n")
                            f.write(f"Working dir: {os.path.dirname(os.path.abspath(__file__))}\n")
                            f.write(f"Finetune.py exists: {os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetune.py'))}\n")
                            f.write(f"Dataset exists: {os.path.exists(data_path)}\n")
                    except Exception as e:
                        st.error(f"Failed to write debug file: {e}")
                    
                    # Start training in background with safe wrapper
                    thread = threading.Thread(target=run_training_safe, args=(config,), name="TrainingThread")
                    thread.daemon = True
                    thread.start()
                    st.session_state.training_thread = thread
                    
                    # Verify thread started
                    time.sleep(0.1)
                    if not thread.is_alive():
                        error_msg = "❌ Training thread died immediately! Check training_error.txt"
                        st.error(error_msg)
                        st.session_state.training_status = "failed"
                        st.session_state.training_logs.append(error_msg)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ Training started! Device: <strong>{'GPU' if st.session_state.gpu_available else 'CPU'}</strong><br>
                        Check the logs below for real-time progress.
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                    st.rerun()
        
        # Show training status
        if st.session_state.training_status == "training":
            # Check for error files first
            error_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_error.txt")
            if os.path.exists(error_file):
                try:
                    with open(error_file, 'r', encoding='utf-8') as f:
                        error_content = f.read()
                        if error_content:
                            st.error("❌ Training thread error detected!")
                            st.text_area("Error Details", error_content, height=200)
                            st.session_state.training_status = "failed"
                except:
                    pass
            
            # Status indicator
            col_status1, col_status2 = st.columns([3, 1])
            with col_status1:
                st.markdown("""
                <div class="info-box">
                    🔄 <strong>Training in progress...</strong> Check logs below for real-time updates.
                </div>
                """, unsafe_allow_html=True)
            with col_status2:
                if st.button("🛑 Stop Training", key="stop_training_main", type="secondary", use_container_width=True):
                    stop_training()
                    st.rerun()
            
            # Check if process is still alive
            process = st.session_state.training_process
            if process is not None:
                if process.poll() is not None:
                    # Process has finished
                    return_code = process.returncode
                    st.session_state.training_status = "completed" if return_code == 0 else "failed"
                    st.session_state.training_process = None
                    st.rerun()
            elif st.session_state.training_status == "training":
                # Process not tracked but status says training - check if it failed
                st.warning("⚠️ Process not tracked - may have failed to start. Check logs below.")
            
            with st.expander("📋 View Training Logs (Live)", expanded=True):
                if st.session_state.training_logs:
                    # Show last 300 lines for better visibility
                    log_text = "\n".join(st.session_state.training_logs[-300:])
                    st.text_area("", log_text, height=500, disabled=True, key="training_logs_display")
                    
                    # Auto-scroll to bottom
                    st.markdown("""
                    <script>
                    setTimeout(function() {
                        var textarea = document.querySelector('textarea[data-testid*="training_logs_display"]');
                        if (textarea) {
                            textarea.scrollTop = textarea.scrollHeight;
                        }
                    }, 100);
                    </script>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("🔄 Refresh Logs", key="refresh_train_logs"):
                            st.rerun()
                    with col2:
                        if st.button("🛑 Stop Training", key="stop_training_expander"):
                            stop_training()
                            st.rerun()
                else:
                    st.info("Waiting for training logs...")
                    st.caption("If logs don't appear, check that finetune.py is running correctly.")
                    
                    # Check if log file exists and read from it as backup
                    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.txt")
                    if os.path.exists(log_file_path):
                        try:
                            with open(log_file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                if lines:
                                    # Update logs from file
                                    file_logs = [line.rstrip() for line in lines if line.strip()]
                                    if file_logs:
                                        st.session_state.training_logs = file_logs[-100:]
                                        st.rerun()
                        except Exception as e:
                            st.caption(f"Error reading log file: {e}")
            
            # Show process status and diagnostics
            st.markdown("### 🔍 Process Status & Diagnostics")
            
            # First, check if process was stored
            if 'training_process' not in st.session_state or st.session_state.training_process is None:
                st.error("⚠️ Process not tracked - may have failed to start")
                st.caption("This usually means the process crashed immediately or failed to start.")
                
                # Show what should have happened
                st.markdown("**Expected process info:**")
                import sys
                python_exe = sys.executable
                finetune_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune.py")
                st.code(f"Python: {python_exe}\nScript: {finetune_path}")
                
                # Test button
                if st.button("🧪 Test if finetune.py works", key="test_finetune"):
                    test_cmd = [python_exe, "-u", "finetune.py", "--help"]
                    try:
                        result = subprocess.run(
                            test_cmd,
                            capture_output=True,
                            text=True,
                            timeout=10,
                            cwd=os.path.dirname(os.path.abspath(__file__))
                        )
                        if result.returncode == 0:
                            st.success("✅ finetune.py can be executed!")
                            st.text_area("Help output", result.stdout, height=200)
                        else:
                            st.error(f"❌ finetune.py failed with code {result.returncode}")
                            st.text_area("Error output", result.stderr, height=200)
                    except Exception as e:
                        st.error(f"❌ Error testing: {e}")
                        import traceback
                        st.text(traceback.format_exc())
            
            process = st.session_state.training_process
            if process is not None:
                status = process.poll()
                if status is None:
                    st.success(f"✅ Process running (PID: {process.pid})")
                    # Check if process is actually doing something
                    try:
                        import psutil
                        proc = psutil.Process(process.pid)
                        cpu_percent = proc.cpu_percent(interval=0.1)
                        memory_mb = proc.memory_info().rss / 1024 / 1024
                        st.caption(f"CPU: {cpu_percent:.1f}% | Memory: {memory_mb:.1f} MB")
                        if cpu_percent < 0.1 and memory_mb < 100:
                            st.warning("⚠️ Process appears to be idle - may be stuck or waiting")
                    except:
                        pass
                else:
                    st.warning(f"⚠️ Process finished with code: {status}")
                    st.session_state.training_status = "completed" if status == 0 else "failed"
            else:
                st.error("⚠️ Process not tracked - may have failed to start")
                # Try to find running finetune.py processes
                try:
                    import psutil
                    found_processes = []
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            cmdline = proc.info['cmdline']
                            if cmdline and 'finetune.py' in ' '.join(cmdline):
                                found_processes.append(f"PID {proc.info['pid']}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    if found_processes:
                        st.info(f"Found finetune.py processes: {', '.join(found_processes)}")
                        st.caption("These processes may be from a previous training session")
                except ImportError:
                    st.caption("Install psutil for better process tracking: pip install psutil")
            
            # Diagnostic buttons
            col_diag1, col_diag2 = st.columns(2)
            with col_diag1:
                if st.button("📄 Check Log File", key="check_log_file_diag"):
                    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.txt")
                    if os.path.exists(log_file_path):
                        try:
                            with open(log_file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if content:
                                    st.text_area("Log File Contents", content, height=300, key="log_file_viewer")
                                    # Update session state logs
                                    st.session_state.training_logs = [line.rstrip() for line in content.splitlines() if line.strip()]
                                    st.rerun()
                                else:
                                    st.warning("Log file is empty")
                        except Exception as e:
                            st.error(f"Error reading log file: {e}")
                    else:
                        st.warning("Log file not found - process may not have started")
            with col_diag2:
                if st.button("🔍 Check Running Processes", key="check_processes"):
                    try:
                        import psutil
                        found = []
                        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                            try:
                                cmdline = proc.info['cmdline']
                                if cmdline and 'finetune.py' in ' '.join(cmdline):
                                    mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                                    found.append(f"PID {proc.info['pid']}: CPU {proc.info['cpu_percent']:.1f}%, Mem {mem_mb:.1f}MB")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        if found:
                            st.info("Found training processes:\n" + "\n".join(found))
                        else:
                            st.warning("No finetune.py processes found")
                    except ImportError:
                        st.error("psutil not installed. Install with: pip install psutil")
            
            # Auto-refresh mechanism - refresh every 2 seconds when training
            time.sleep(2)
            st.rerun()
        
        elif st.session_state.training_status == "completed":
            st.markdown("""
            <div class="success-box">
                ✅ Training completed successfully!
            </div>
            """, unsafe_allow_html=True)
            st.session_state.training_status = "idle"
            st.balloons()
        
        elif st.session_state.training_status == "failed":
            st.markdown("""
            <div class="error-box">
                ❌ Training failed. Check logs for details.
            </div>
            """, unsafe_allow_html=True)
            st.session_state.training_status = "idle"
    
    elif page == "🧪 Test Model":
        st.header("🧪 Test Your Fine-tuned Model")
        
        trained_models = load_trained_models()
        
        if not trained_models:
            st.markdown("""
            <div class="warning-box">
                ⚠️ No trained models found. Train a model first!
            </div>
            """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📦 Model Selection")
                selected_model = st.selectbox("Select Trained Model", trained_models)
                
                # Get base model from model_map.json if available
                model_map_path = "./fine_tuned_adapter/model_map.json"
                base_model = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
                if os.path.exists(model_map_path):
                    with open(model_map_path, "r") as f:
                        model_map = json.load(f)
                        # Reverse lookup
                        for k, v in model_map.items():
                            if selected_model.startswith(v):
                                base_model = k
                                break
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>Base Model:</strong><br>
                    <code>{base_model}</code><br><br>
                    <strong>Adapter:</strong><br>
                    <code>{selected_model}</code>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("⚙️ Generation Settings")
                max_tokens = st.slider("Max Tokens", 32, 512, 128)
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1, 
                                       help="Higher = more creative, Lower = more focused")
            
            with col2:
                st.subheader("💬 Test Prompt")
                prompt = st.text_area(
                    "Enter your prompt",
                    value="### Instruction:\nSay hello\n\n### Response:\n",
                    height=150,
                    help="Use the format: ### Instruction:\\nYour question\\n\\n### Response:\\n"
                )
                
                if st.button("🚀 Generate Response", type="primary", use_container_width=True):
                    adapter_dir = f"./fine_tuned_adapter/{selected_model}"
                    
                    with st.spinner("🔄 Generating response..."):
                        try:
                            cmd = [
                                "python", "run_adapter.py",
                                "--adapter-dir", adapter_dir,
                                "--base-model", base_model,
                                "--prompt", prompt,
                                "--max-new-tokens", str(max_tokens),
                                "--temperature", str(temperature),
                            ]
                            
                            result = subprocess.run(
                                cmd,
                                capture_output=True,
                                text=True,
                                timeout=120,
                                cwd=os.path.dirname(os.path.abspath(__file__))
                            )
                            
                            if result.returncode == 0:
                                # Extract output from stdout
                                output = result.stdout
                                st.markdown("""
                                <div class="success-box">
                                    ✅ Generation Complete!
                                </div>
                                """, unsafe_allow_html=True)
                                st.text_area("Generated Response", output, height=200, key="generated_output")
                            else:
                                st.markdown(f"""
                                <div class="error-box">
                                    ❌ Error: {result.stderr[:500]}
                                </div>
                                """, unsafe_allow_html=True)
                        except subprocess.TimeoutExpired:
                            st.error("⏱️ Generation timed out. Try reducing max tokens.")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    
    elif page == "✅ Validate Model":
        st.header("✅ Validate Model Performance")
        
        trained_models = load_trained_models()
        
        if not trained_models:
            st.markdown("""
            <div class="warning-box">
                ⚠️ No trained models found. Train a model first!
            </div>
            """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_model = st.selectbox("Select Model", trained_models)
                
                model_map_path = "./fine_tuned_adapter/model_map.json"
                base_model = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
                if os.path.exists(model_map_path):
                    with open(model_map_path, "r") as f:
                        model_map = json.load(f)
                        for k, v in model_map.items():
                            if selected_model.startswith(v):
                                base_model = k
                                break
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>Model:</strong> {selected_model}
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("⚙️ Validation Settings")
                max_tokens = st.slider("Max Tokens", 32, 512, 128)
            
            with col2:
                st.subheader("📋 Validation Prompts")
                uploaded_prompts = st.file_uploader(
                    "Upload Validation Prompts (JSONL)",
                    type=["jsonl", "json"],
                    help="Format: {\"id\": \"1\", \"type\": \"positive\", \"prompt\": \"...\"}"
                )
                
                prompts_file = "validation_prompts.jsonl"
                if uploaded_prompts:
                    with open(prompts_file, "wb") as f:
                        f.write(uploaded_prompts.getbuffer())
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ Prompts uploaded: <strong>{uploaded_prompts.name}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Preview
                    with st.expander("📖 Preview Prompts", expanded=False):
                        with open(prompts_file, "r") as f:
                            lines = f.readlines()[:5]
                            preview_data = []
                            for line in lines:
                                try:
                                    preview_data.append(json.loads(line))
                                except:
                                    pass
                            if preview_data:
                                st.json(preview_data)
                
                if st.button("🚀 Run Validation", type="primary", use_container_width=True):
                    if not os.path.exists(prompts_file):
                        st.markdown("""
                        <div class="error-box">
                            ❌ Please upload validation prompts first!
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        adapter_dir = f"./fine_tuned_adapter/{selected_model}"
                        output_file = "validation_results.jsonl"
                        
                        with st.spinner("🔄 Running validation..."):
                            cmd = [
                                "python", "validate_prompts.py",
                                "--adapter-dir", adapter_dir,
                                "--base-model", base_model,
                                "--prompts", prompts_file,
                                "--out", output_file,
                                "--max-new-tokens", str(max_tokens),
                            ]
                            
                            result = subprocess.run(
                                cmd, 
                                capture_output=True, 
                                text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))
                            )
                            
                            if result.returncode == 0:
                                st.markdown("""
                                <div class="success-box">
                                    ✅ Validation Complete!
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Load and display results
                                if os.path.exists(output_file):
                                    results = []
                                    with open(output_file, "r") as f:
                                        for line in f:
                                            results.append(json.loads(line))
                                    
                                    # Results table
                                    st.subheader("📊 Results")
                                    df = pd.DataFrame(results)
                                    st.dataframe(df, use_container_width=True)
                                    
                                    # Summary metrics
                                    total = len(results)
                                    passed = sum(1 for r in results if r.get("pass", False))
                                    failed = total - passed
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total", total)
                                    with col2:
                                        st.metric("✅ Passed", passed, f"{(passed/total*100):.1f}%")
                                    with col3:
                                        st.metric("❌ Failed", failed, f"{(failed/total*100):.1f}%")
                                    
                                    # Download results
                                    st.download_button(
                                        label="📥 Download Results",
                                        data=json.dumps(results, indent=2),
                                        file_name="validation_results.json",
                                        mime="application/json"
                                    )
                            else:
                                st.markdown(f"""
                                <div class="error-box">
                                    ❌ Validation failed: {result.stderr[:500]}
                                </div>
                                """, unsafe_allow_html=True)
    
    elif page == "📊 Training History":
        st.header("📊 Training History & Logs")
        
        trained_models = load_trained_models()
        
        if trained_models:
            st.subheader("📁 Trained Models")
            for model in trained_models:
                with st.expander(f"📦 {model}", expanded=False):
                    model_path = f"./fine_tuned_adapter/{model}"
                    st.code(model_path)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"Full path: `{os.path.abspath(model_path)}`")
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_{model}"):
                            import shutil
                            try:
                                shutil.rmtree(model_path)
                                st.success(f"✅ Deleted {model}")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
        else:
            st.markdown("""
            <div class="info-box">
                ℹ️ No trained models yet. Start training to see history here.
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.training_logs:
            st.subheader("📋 Latest Training Logs")
            log_text = "\n".join(st.session_state.training_logs)
            st.text_area("", log_text, height=400, disabled=True, key="history_logs")
            
            if st.button("🔄 Refresh Logs"):
                st.rerun()
        else:
            st.info("No training logs available yet.")

if __name__ == "__main__":
    main()

