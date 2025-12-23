#!/usr/bin/env python3
"""
Simplified GUI for LLM Fine-tuning
Just runs train_basic.py and shows the logs - NO COMPLICATIONS
"""
import streamlit as st
import os
import subprocess
import time
import json
import sys
import psutil

# Page config
st.set_page_config(
    page_title="LLM Fine-Tuning Studio",
    page_icon="🚀",
    layout="wide"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    div[data-testid="stButton"] > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🚀 LLM Fine-Tuning Studio</div>', unsafe_allow_html=True)

# Initialize session state
if 'training_status' not in st.session_state:
    st.session_state.training_status = "idle"
if 'training_process' not in st.session_state:
    st.session_state.training_process = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = 0

# Main tab
if st.session_state.training_status == "idle":
    st.markdown("### 📝 Configure Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Settings")
        dataset_path = st.text_input("Dataset Path", "train_data.jsonl")
        output_dir = st.text_input("Output Directory", "./fine_tuned_adapter")
        epochs = st.number_input("Epochs", min_value=1, max_value=10, value=1)
    
    with col2:
        st.markdown("#### Advanced Settings")
        st.info("Using optimized defaults for LoRA training")
        st.markdown("""
        - **LoRA Rank (r)**: 8
        - **LoRA Alpha**: 16
        - **Batch Size**: 1
        - **Learning Rate**: 2e-4
        """)
    
    if st.button("🚀 Start Training", type="primary"):
        if not os.path.exists(dataset_path):
            st.error(f"Dataset not found: {dataset_path}")
        else:
            # Save config
            config = {
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "epochs": epochs
            }
            config_file = os.path.join(os.path.dirname(__file__), "train_config.json")
            with open(config_file, 'w') as f:
                json.dump(config, f)
            
            # Update train_basic.py to use these settings
            st.session_state.training_status = "starting"
            st.rerun()

elif st.session_state.training_status == "starting":
    st.markdown("### 🔄 Starting Training...")
    
    # Start training in background
    log_file = os.path.join(os.path.dirname(__file__), "training_log.txt")
    script_path = os.path.join(os.path.dirname(__file__), "train_basic.py")
    
    # Clear old logs
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Load config
    config_data = None
    config_file = os.path.join(os.path.dirname(__file__), "train_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
    
    # Build command
    cmd = [
        sys.executable, "-u", script_path,
        "--data-path", config_data.get("dataset_path", "train_data.jsonl"),
        "--output-dir", config_data.get("output_dir", "./fine_tuned_adapter"),
        "--epochs", str(config_data.get("epochs", 1))
    ]
    
    # Start process
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'
    
    with open(log_file, 'w', encoding='utf-8', buffering=1) as f:
        f.write(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("=" * 60 + "\n")
        f.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(__file__),
            env=env,
            text=True,
            bufsize=1
        )
    
    st.session_state.training_process = process
    st.session_state.training_status = "training"
    st.session_state.last_refresh = time.time()
    time.sleep(0.5)  # Brief pause to let process start
    st.rerun()

elif st.session_state.training_status == "training":
    st.markdown("### 🔄 Training In Progress")
    
    # Check process status
    process = st.session_state.training_process
    process_alive = False
    if process:
        try:
            # Check if process is still running
            process_alive = process.poll() is None
            if not process_alive:
                return_code = process.returncode
                if return_code == 0:
                    st.session_state.training_status = "completed"
                    st.rerun()
                else:
                    st.session_state.training_status = "failed"
                    st.rerun()
        except Exception as e:
            st.warning(f"Error checking process: {e}")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        if process_alive:
            st.info(f"🟢 Training is running (PID: {process.pid}) - Auto-refreshing every 2 seconds")
        else:
            st.warning("⚠️ Process status unknown")
    with col2:
        if st.button("🛑 Stop"):
            # Kill the process
            if process and process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except:
                    pass
            st.session_state.training_status = "idle"
            st.session_state.training_process = None
            st.rerun()
    
    # Read and display logs
    log_file = os.path.join(os.path.dirname(__file__), "training_log.txt")
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                logs = f.read()
                st.text_area("Training Logs", logs, height=600, key="logs_display")
                
                # Check if done (look for completion message)
                if "Done! Model saved" in logs:
                    st.session_state.training_status = "completed"
                    st.rerun()
        except Exception as e:
            st.error(f"Error reading logs: {e}")
    else:
        st.warning("Log file not found yet...")
    
    # Auto-refresh mechanism using placeholder rerun
    current_time = time.time()
    if current_time - st.session_state.last_refresh > 2:
        st.session_state.last_refresh = current_time
        st.rerun()
    else:
        # Use a small delay then trigger rerun via JavaScript injection
        st.markdown("""
        <script>
        setTimeout(function() {
            window.parent.postMessage({type: 'streamlit:rerun'}, '*');
        }, 2000);
        </script>
        """, unsafe_allow_html=True)

elif st.session_state.training_status == "completed":
    st.success("✅ Training completed successfully!")
    
    log_file = os.path.join(os.path.dirname(__file__), "training_log.txt")
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            logs = f.read()
            st.text_area("Final Logs", logs, height=400)
    
    if st.button("Start New Training"):
        st.session_state.training_status = "idle"
        st.rerun()

elif st.session_state.training_status == "failed":
    st.error("❌ Training failed!")
    
    log_file = os.path.join(os.path.dirname(__file__), "training_log.txt")
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            logs = f.read()
            st.text_area("Error Logs", logs, height=400)
    
    if st.button("Try Again"):
        st.session_state.training_status = "idle"
        st.rerun()

