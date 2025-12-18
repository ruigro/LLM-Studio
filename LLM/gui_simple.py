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
    
    # Start process
    import sys
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(__file__),
            env=env
        )
    
    st.session_state.training_pid = process.pid
    st.session_state.training_status = "training"
    time.sleep(1)  # Give it a second to start
    st.rerun()

elif st.session_state.training_status == "training":
    st.markdown("### 🔄 Training In Progress")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.info("Training is running... Logs update every 2 seconds")
    with col2:
        if st.button("🛑 Stop"):
            st.session_state.training_status = "idle"
            st.rerun()
    
    # Read and display logs
    log_file = os.path.join(os.path.dirname(__file__), "training_log.txt")
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            logs = f.read()
            st.text_area("Training Logs", logs, height=600, key="logs_display")
            
            # Check if done
            if "Done! Model saved" in logs:
                st.session_state.training_status = "completed"
                st.rerun()
            elif "Traceback" in logs or "Error" in logs:
                # Check if it's an actual error or just a warning
                if "Training..." in logs:
                    pass  # It's running fine
                else:
                    st.session_state.training_status = "failed"
                    st.rerun()
    
    # Auto-refresh
    time.sleep(2)
    st.rerun()

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

