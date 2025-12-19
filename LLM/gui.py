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
from datetime import datetime
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
    /* FORCE 30px top spacing - multiple selectors for maximum specificity */
    section.main > div.block-container {
        padding-top: 30px !important;
        margin-top: 0 !important;
    }
    .stApp section.main > div {
        padding-top: 30px !important;
        margin-top: 0 !important;
    }
    div.block-container {
        padding-top: 30px !important;
        margin-top: 0 !important;
    }
    section[data-testid="stMain"] > div {
        padding-top: 30px !important;
        margin-top: 0 !important;
    }
    .main > div:first-child {
        padding-top: 30px !important;
        margin-top: 0 !important;
    }
    
    /* Hide Streamlit deploy button and menu but keep sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        visibility: visible !important;
        display: block !important;
    }
    button[title="View app source"] {display: none;}
    button[title="Deploy this app"] {display: none;}
    button[kind="header"] {display: none;}
    /* Keep sidebar toggle button visible */
    button[kind="header"][data-testid*="baseButton-header"] {
        display: block !important;
    }
    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
    }
    
    /* Custom Navbar Styles - Improved Colors */
    .navbar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .navbar-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
    }
    .nav-item {
        color: #ffffff !important;
        text-decoration: none !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 0.5rem !important;
        transition: all 0.3s ease !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        border: 2px solid rgba(255,255,255,0.4) !important;
        background: rgba(255,255,255,0.2) !important;
        backdrop-filter: blur(10px) !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5) !important;
        font-size: 1rem !important;
    }
    .nav-item:hover {
        background-color: rgba(255,255,255,0.3) !important;
        transform: translateY(-2px) !important;
        border-color: rgba(255,255,255,0.6) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
    }
    .nav-item.active {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border-color: #ffffff !important;
        font-weight: 900 !important;
        box-shadow: 0 4px 12px rgba(255,255,255,0.6) !important;
        text-shadow: none !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 0;
        padding: 0.5rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 0.4rem 0.6rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.3rem 0;
        display: inline-block;
        width: auto;
        max-width: 100%;
        line-height: 1.3;
        font-size: 0.9em;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 0.4rem 0.6rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
        margin: 0.3rem 0;
        display: inline-block;
        width: auto;
        max-width: 100%;
        line-height: 1.3;
        font-size: 0.9em;
    }
    .success-box strong {
        color: #155724;
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
    # === 4 NEWEST MODELS (2024-2025) ===
    "Llama 3.3 70B Instruct (4-bit)": {
        "id": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "size": "~35 GB",
        "description": "Latest Llama 3.3 70B model with enhanced capabilities",
        "tags": ["llama", "70b", "instruct", "4-bit"],
        "category": "newest"
    },
    "Qwen2.5 72B Instruct (4-bit)": {
        "id": "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        "size": "~36 GB",
        "description": "State-of-the-art Qwen 2.5 72B model",
        "tags": ["qwen", "72b", "instruct", "4-bit", "multilingual"],
        "category": "newest"
    },
    "Gemma 2 27B Instruct (4-bit)": {
        "id": "unsloth/gemma-2-27b-it-bnb-4bit",
        "size": "~14 GB",
        "description": "Google's Gemma 2 27B instruction-tuned model",
        "tags": ["gemma", "27b", "instruct", "4-bit"],
        "category": "newest"
    },
    "Phi-4 14B (4-bit)": {
        "id": "unsloth/Phi-4-bnb-4bit",
        "size": "~7 GB",
        "description": "Microsoft's latest Phi-4 14B model",
        "tags": ["phi", "14b", "4-bit"],
        "category": "newest"
    },
    
    # === 20 MOST POPULAR MODELS ===
    "Llama 3.2 1B Instruct (4-bit)": {
        "id": "unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit",
        "size": "~800 MB",
        "description": "Ultra-lightweight 1B model, fastest training",
        "tags": ["llama", "1b", "instruct", "4-bit"]
    },
    "Llama 3.2 3B Instruct (4-bit)": {
        "id": "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit",
        "size": "~2.5 GB",
        "description": "Fast 3B parameter model, great for quick experiments",
        "tags": ["llama", "3b", "instruct", "4-bit"]
    },
    "Llama 3.1 8B Instruct (4-bit)": {
        "id": "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit",
        "size": "~5 GB",
        "description": "Powerful 8B model with better performance",
        "tags": ["llama", "8b", "instruct", "4-bit"]
    },
    "Llama 3.1 70B Instruct (4-bit)": {
        "id": "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "size": "~35 GB",
        "description": "Powerful Llama 3.1 70B model",
        "tags": ["llama", "70b", "instruct", "4-bit"]
    },
    "Mistral 7B Instruct v0.3 (4-bit)": {
        "id": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "size": "~4.5 GB",
        "description": "Latest Mistral 7B v0.3 instruction model",
        "tags": ["mistral", "7b", "instruct", "4-bit"]
    },
    "Mistral Nemo 12B Instruct (4-bit)": {
        "id": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "size": "~6.5 GB",
        "description": "Mistral Nemo 12B with enhanced capabilities",
        "tags": ["mistral", "12b", "instruct", "4-bit"]
    },
    "Qwen 2.5 3B Instruct (4-bit)": {
        "id": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "size": "~2.5 GB",
        "description": "Compact Qwen 2.5 model with multilingual support",
        "tags": ["qwen", "3b", "instruct", "4-bit", "multilingual"]
    },
    "Qwen 2.5 7B Instruct (4-bit)": {
        "id": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "size": "~4.5 GB",
        "description": "Popular Qwen 2.5 7B model",
        "tags": ["qwen", "7b", "instruct", "4-bit", "multilingual"]
    },
    "Qwen 2.5 14B Instruct (4-bit)": {
        "id": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "size": "~8 GB",
        "description": "Qwen 2.5 14B with strong reasoning",
        "tags": ["qwen", "14b", "instruct", "4-bit", "multilingual"]
    },
    "Qwen 2.5 32B Instruct (4-bit)": {
        "id": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        "size": "~18 GB",
        "description": "Large Qwen 2.5 32B model",
        "tags": ["qwen", "32b", "instruct", "4-bit", "multilingual"]
    },
    "Gemma 2 2B Instruct (4-bit)": {
        "id": "unsloth/gemma-2-2b-it-bnb-4bit",
        "size": "~1.5 GB",
        "description": "Google's smallest Gemma 2 model",
        "tags": ["gemma", "2b", "instruct", "4-bit"]
    },
    "Gemma 2 9B Instruct (4-bit)": {
        "id": "unsloth/gemma-2-9b-it-bnb-4bit",
        "size": "~5 GB",
        "description": "Google's Gemma 2 9B instruction model",
        "tags": ["gemma", "9b", "instruct", "4-bit"]
    },
    "Phi-3.5 Mini Instruct (4-bit)": {
        "id": "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "size": "~2.5 GB",
        "description": "Microsoft's Phi-3.5 Mini model",
        "tags": ["phi", "3.8b", "instruct", "4-bit"]
    },
    "Phi-3 Medium Instruct (4-bit)": {
        "id": "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
        "size": "~7.5 GB",
        "description": "Microsoft's Phi-3 Medium 14B model",
        "tags": ["phi", "14b", "instruct", "4-bit"]
    },
    "Yi 1.5 9B Chat (4-bit)": {
        "id": "unsloth/Yi-1.5-9B-Chat-bnb-4bit",
        "size": "~5 GB",
        "description": "Yi 1.5 9B chat model with strong performance",
        "tags": ["yi", "9b", "chat", "4-bit"]
    },
    "DeepSeek V2.5 (4-bit)": {
        "id": "unsloth/DeepSeek-V2.5-bnb-4bit",
        "size": "~10 GB",
        "description": "DeepSeek V2.5 with enhanced reasoning",
        "tags": ["deepseek", "v2.5", "4-bit"]
    },
    "Mixtral 8x7B Instruct (4-bit)": {
        "id": "unsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit",
        "size": "~24 GB",
        "description": "Mistral's Mixtral MoE model",
        "tags": ["mixtral", "8x7b", "instruct", "4-bit", "moe"]
    },
    "Zephyr 7B Beta (4-bit)": {
        "id": "unsloth/zephyr-7b-beta-bnb-4bit",
        "size": "~4.5 GB",
        "description": "Popular Zephyr 7B Beta model",
        "tags": ["zephyr", "7b", "4-bit"]
    },
    "OpenHermes 2.5 Mistral 7B (4-bit)": {
        "id": "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit",
        "size": "~4.5 GB",
        "description": "OpenHermes 2.5 based on Mistral 7B",
        "tags": ["openhermes", "mistral", "7b", "4-bit"]
    },
    "Nous Hermes 2 Mixtral 8x7B (4-bit)": {
        "id": "unsloth/Nous-Hermes-2-Mixtral-8x7B-DPO-bnb-4bit",
        "size": "~24 GB",
        "description": "Nous Hermes 2 on Mixtral architecture",
        "tags": ["nous", "hermes", "mixtral", "8x7b", "4-bit"]
    },
    "Starling 7B Alpha (4-bit)": {
        "id": "unsloth/Starling-LM-7B-alpha-bnb-4bit",
        "size": "~4.5 GB",
        "description": "Starling 7B Alpha RLHF model",
        "tags": ["starling", "7b", "4-bit", "rlhf"]
    },
    "Custom Model": {
        "id": None,
        "size": "Varies",
        "description": "Enter a custom Hugging Face model ID",
        "tags": ["custom"]
    }
}

def load_trained_models():
    """Load list of trained model checkpoints with metadata"""
    models_dir = "./models_trained"
    models = []
    
    # Check new models_trained directory first
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                # Check if it has adapter files
                adapter_file = os.path.join(item_path, "adapter_model.safetensors")
                metadata_file = os.path.join(item_path, "training_metadata.json")
                
                if os.path.exists(adapter_file):
                    # Try to load metadata
                    display_name = item
                    base_model = "Unknown"
                    
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                                display_name = metadata.get("model_display_name", item)
                                base_model = metadata.get("model_name", "Unknown")
                                timestamp = metadata.get("timestamp", "")
                                if timestamp:
                                    formatted_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"
                                    display_name = f"{display_name} ({formatted_time})"
                        except:
                            pass
                    
                    models.append((item_path, display_name, base_model))
    
    # Also check legacy fine_tuned_adapter directory
    legacy_dir = "./fine_tuned_adapter"
    if os.path.exists(legacy_dir):
        adapter_file = os.path.join(legacy_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            models.append((legacy_dir, "Legacy Model", "Unknown"))
        
        # Check for checkpoint subdirectories
        for item in os.listdir(legacy_dir):
            item_path = os.path.join(legacy_dir, item)
            if os.path.isdir(item_path) and ("checkpoint" in item.lower()):
                if os.path.exists(os.path.join(item_path, "adapter_model.safetensors")):
                    models.append((item_path, f"Checkpoint: {item}", "Unknown"))
    
    return sorted(models, key=lambda x: x[0], reverse=True)

def load_all_available_models():
    """Load both base models and fine-tuned models for testing"""
    models = []
    
    def detect_model_type(model_name):
        """Detect if model is instruct/chat or base"""
        name_lower = model_name.lower()
        # Check for instruct/chat indicators
        instruct_keywords = ["instruct", "chat", "-it", "alpaca", "vicuna", "wizard", "nemotron"]
        if any(keyword in name_lower for keyword in instruct_keywords):
            return "instruct"
        return "base"
    
    # Add downloaded base models
    downloaded_dir = "./models"
    if os.path.exists(downloaded_dir):
        for item in os.listdir(downloaded_dir):
            item_path = os.path.join(downloaded_dir, item)
            if os.path.isdir(item_path):
                # Check if it's a valid model directory
                if os.path.exists(os.path.join(item_path, "config.json")):
                    model_type = detect_model_type(item)
                    type_label = "INSTRUCT" if model_type == "instruct" else "BASE"
                    models.append({
                        "name": f"[{type_label}] {item}",
                        "path": item_path,
                        "type": "base",
                        "model_type": model_type,
                        "display_name": item.replace("_", "/")
                    })
    
    # Add fine-tuned models (always treated as instruct after training)
    trained_models = load_trained_models()
    for path, display_name, base_model in trained_models:
        models.append({
            "name": f"[FINE-TUNED] {display_name}",
            "path": path,
            "type": "fine-tuned",
            "model_type": "instruct",  # Fine-tuned models are instruction-tuned
            "base_model": base_model
        })
    
    return models

def clean_model_output(text):
    """Clean model output from training artifacts and formatting issues"""
    import re
    import html
    
    # Unescape HTML entities (&#x27; etc)
    text = html.unescape(text)
    
    # NUCLEAR OPTION: Remove ALL Hugging Face security warnings
    # This catches EVERYTHING from the warning start to the actual response
    
    # Step 1: Remove the entire warning block (from "A new version" to end or until we find actual content)
    text = re.sub(
        r'(?:❌ Error: )?A new version of the following files was downloaded.*?(?=\n\n[A-Z]|$)', 
        '', 
        text, 
        flags=re.DOTALL
    )
    
    # Step 2: Remove any standalone file list lines
    text = re.sub(r'^- [a-z_]+\.py\s*$', '', text, flags=re.MULTILINE)
    
    # Step 3: Remove "Make sure to double-check" lines
    text = re.sub(r'^\. Make sure to double-check.*?$', '', text, flags=re.MULTILINE)
    
    # Step 4: Remove any line containing huggingface.co URLs
    text = re.sub(r'^.*https://huggingface\.co/.*$', '', text, flags=re.MULTILINE)
    
    # Step 5: If text starts with warning fragments, remove them
    if text.startswith('- ') or text.startswith('. Make sure'):
        lines = text.split('\n')
        # Skip lines until we find actual content
        for i, line in enumerate(lines):
            if line and not line.startswith('-') and not line.startswith('. Make sure') and len(line) > 10:
                text = '\n'.join(lines[i:])
                break
    
    # Remove training artifacts
    artifacts = ["Note:", "Code:", "### Instruction:", "### Response:", 
                 "<div", "</div>", "class=", "```python", "```"]
    for artifact in artifacts:
        if artifact in text:
            text = text.split(artifact)[0].strip()
    
    # Remove incomplete sentences
    text = re.sub(r'\s+\S*$', '', text) if not text.endswith(('.', '!', '?', '"', "'")) else text
    
    # Clean up extra whitespace and blank lines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    
    return text.strip()

def generate_model_response(model_info, prompt):
    """Generate response from a model (base or fine-tuned)"""
    import subprocess
    import sys
    
    # Get model type for proper formatting
    model_type = model_info.get("model_type", "base")
    
    try:
        if model_info["type"] == "base":
            # For base models, use run_adapter.py with --base-model flag and NO --adapter-dir
            # This tells it to load ONLY the base model without any adapter
            cmd = [
                sys.executable,
                "run_adapter.py",
                "--base-model", model_info["path"],
                "--prompt", prompt,
                "--max-new-tokens", "512",
                "--temperature", "0.7",
                "--model-type", model_type,
                "--no-adapter"  # Flag to skip adapter loading
            ]
        else:
            # Generate from fine-tuned model (with adapter)
            cmd = [
                sys.executable,
                "run_adapter.py",
                "--adapter-dir", model_info["path"],
                "--prompt", prompt,
                "--max-new-tokens", "512",
                "--temperature", "0.7",
                "--model-type", model_type
            ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            # Extract output if marked
            if "--- OUTPUT ---" in output:
                output = output.split("--- OUTPUT ---", 1)[1].strip()
            return clean_model_output(output)
        else:
            error_msg = result.stderr.strip()
            # Show only first 200 chars of error
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            return f"❌ Error: {error_msg}"
    
    except subprocess.TimeoutExpired:
        return "⏱️ Timeout: Response took too long"
    except Exception as e:
        return f"❌ Error: {str(e)[:200]}"


def analyze_model(model_name):
    """Analyze model characteristics and return profile"""
    if not model_name:
        return None
    
    model_lower = model_name.lower()
    
    # Detect model type
    if "instruct" in model_lower or "chat" in model_lower:
        model_type = "Instruct"
    else:
        model_type = "Base"
    
    # Detect size (3B, 7B, 8B, 13B, etc.)
    import re
    size_match = re.search(r'(\d+\.?\d*)[bB]', model_name)
    if size_match:
        size_value = float(size_match.group(1))
        model_size = f"{size_value}B"
    else:
        size_value = 8.0  # Default assumption
        model_size = "Unknown"
    
    # Detect quantization
    if "4bit" in model_lower or "4-bit" in model_lower or "bnb-4bit" in model_lower:
        quantization = "4-bit"
    elif "8bit" in model_lower or "8-bit" in model_lower:
        quantization = "8-bit"
    else:
        quantization = "Full Precision"
    
    return {
        "type": model_type,
        "size": model_size,
        "size_value": size_value,
        "quantization": quantization,
        "name": model_name
    }

def get_recommended_params(model_profile, dataset_size=10):
    """Get recommended training parameters based on model profile"""
    if not model_profile:
        return None
    
    # Base recommendations
    params = {}
    
    # Learning Rate calculation
    base_lr = 2e-4  # Default for base models
    
    if model_profile["type"] == "Instruct":
        base_lr = 5e-5  # Much lower for instruct models
    
    # Adjust for model size
    if model_profile["size_value"] >= 7:
        base_lr = base_lr * 0.5  # Half for larger models
    
    # Adjust for quantization
    if "4-bit" in model_profile["quantization"]:
        base_lr = base_lr * 0.8  # 20% lower for quantized
    
    params["learning_rate"] = base_lr
    
    # LoRA parameters
    if model_profile["size_value"] >= 7:
        params["lora_r"] = 8
        params["lora_alpha"] = 16
    else:
        params["lora_r"] = 8
        params["lora_alpha"] = 16
    
    params["lora_dropout"] = 0.05
    
    # Batch size
    if model_profile["size_value"] >= 7:
        params["batch_size"] = 1
    else:
        params["batch_size"] = 2
    
    # Epochs based on dataset size
    if dataset_size < 20:
        params["epochs"] = 1
    elif dataset_size < 100:
        params["epochs"] = 2
    else:
        params["epochs"] = 3
    
    # Other params
    params["grad_accum"] = 8
    params["max_seq_length"] = 2048
    
    return params

def check_parameter_safety(param_name, value, recommended_value, model_profile):
    """Check if parameter value is safe and return status"""
    if recommended_value is None:
        return "unknown", "No recommendation available"
    
    if param_name == "learning_rate":
        ratio = value / recommended_value
        if 0.8 <= ratio <= 1.2:
            return "optimal", "✅ Safe learning rate"
        elif ratio > 2.0:
            if model_profile and model_profile["type"] == "Instruct":
                return "danger", f"⚠️ {ratio:.1f}x higher than recommended! May destroy instruction-following"
            else:
                return "warning", f"⚠️ {ratio:.1f}x higher than recommended. May cause instability"
        elif ratio > 1.2:
            return "warning", f"⚠️ {ratio:.1f}x higher than recommended"
        else:
            return "acceptable", "Lower LR = slower but safer training"
    
    elif param_name == "epochs":
        if value == recommended_value:
            return "optimal", "✅ Optimal epoch count"
        elif value > recommended_value * 2:
            return "danger", f"⚠️ {value} epochs may cause severe overfitting!"
        elif value > recommended_value:
            return "warning", f"⚠️ Risk of overfitting with {value} epochs"
        else:
            return "acceptable", "Fewer epochs = less overfitting risk"
    
    elif param_name == "batch_size":
        if value == recommended_value:
            return "optimal", "✅ Optimal batch size"
        elif value > recommended_value and model_profile and model_profile["size_value"] >= 7:
            return "warning", f"⚠️ Batch size {value} may cause OOM on large models"
        else:
            return "acceptable", "Batch size acceptable"
    
    return "acceptable", "Within acceptable range"

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

def fetch_model_description(model_id):
    """Fetch model card description from Hugging Face"""
    if not HF_HUB_AVAILABLE:
        return "Description unavailable"
    
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        # Get card data if available
        if hasattr(info, 'card_data') and info.card_data:
            # Try to get description from card metadata
            if hasattr(info.card_data, 'get'):
                desc = info.card_data.get('model-index', [{}])[0].get('description', '')
                if desc:
                    return desc
        
        # Fallback: Try to get from README (first 1000 chars)
        try:
            from huggingface_hub import hf_hub_download
            readme_path = hf_hub_download(model_id, "README.md", repo_type="model")
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # First 1000 chars
                # Strip markdown headers and extract text
                lines = [l for l in content.split('\n') if l.strip() and not l.startswith('#')]
                return ' '.join(lines[:3])  # First 3 paragraphs
        except:
            pass
            
        return "No description available"
    except Exception as e:
        return f"Error fetching description: {str(e)[:50]}"

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
    /* FORCE WHITE TEXT IN ALL BUTTONS */
    button {
        color: #ffffff !important;
    }
    button span, button p, button div {
        color: #ffffff !important;
    }
    /* Style active navbar buttons */
    div[data-testid*="stButton"] > button[kind="primary"] {
        background: linear-gradient(90deg, #ff7f0e 0%, #ff4500 100%) !important;
        color: #ffffff !important;
        font-weight: bold !important;
        border: 2px solid #ff7f0e !important;
        box-shadow: 0 4px 8px rgba(255, 127, 14, 0.3) !important;
        transform: scale(1.02);
    }
    div[data-testid*="stButton"] > button[kind="primary"] * {
        color: #ffffff !important;
    }
    /* Style inactive navbar buttons */
    div[data-testid*="stButton"] > button[kind="secondary"] {
        background-color: rgba(102, 126, 234, 0.4) !important;
        color: #ffffff !important;
        border: 2px solid rgba(255,255,255,0.4) !important;
    }
    div[data-testid*="stButton"] > button[kind="secondary"] * {
        color: #ffffff !important;
    }
    div[data-testid*="stButton"] > button[kind="secondary"]:hover {
        background-color: rgba(102, 126, 234, 0.6) !important;
        border-color: #ffffff !important;
        color: #ffffff !important;
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
        python_exe, "-u", "train_basic.py",  # Use the working training script
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
    
    # Add optional max_examples if specified
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
        finetune_path = os.path.join(working_dir, "train_basic.py")
        if not os.path.exists(finetune_path):
            error_msg = f"❌ Error: train_basic.py not found at {finetune_path}"
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
                encoding='utf-8',
                errors='replace',  # Replace invalid UTF-8 chars instead of crashing
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
                    try:
                        output = process.stdout.readline()
                        if output:
                            # Handle Unicode encoding issues
                            try:
                                # Try to decode as UTF-8, fallback to errors='replace'
                                decoded = output.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                                output_queue.put(decoded.rstrip())
                            except:
                                # If that fails, use errors='replace' to skip bad chars
                                output_queue.put(output.rstrip())
                        elif process.poll() is not None:
                            # Process finished, read remaining
                            try:
                                remaining = process.stdout.read()
                                if remaining:
                                    # Handle Unicode for remaining output too
                                    try:
                                        decoded = remaining.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                                        for line in decoded.splitlines():
                                            if line.strip():
                                                output_queue.put(line.rstrip())
                                    except:
                                        for line in remaining.splitlines():
                                            if line.strip():
                                                output_queue.put(line.rstrip())
                            except Exception as read_err:
                                output_queue.put(f"Error reading remaining output: {read_err}")
                            output_queue.put(None)  # Signal end
                            break
                    except UnicodeDecodeError as ude:
                        # Handle Unicode decode errors gracefully
                        output_queue.put(f"[Unicode decode error: {ude}]")
                    except Exception as read_err:
                        output_queue.put(f"ERROR reading output: {read_err}")
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
                    log_file.write("⚠️ No output for 60s. Model may be saving (this is normal at the end)...\n")
                    log_file.flush()
                    # DON'T break - let the process finish naturally!
                    # Model saving is silent and can take time
        
        stop_reading.set()
        
        # Get return code BEFORE closing file
        return_code = process.poll()
        if return_code is None:
            return_code = 0
        
        # Add final status message to log file BEFORE closing
        try:
            if not log_file.closed:
                log_file.write("=" * 60 + "\n")
                if return_code == 0:
                    log_file.write("✅ Training completed successfully!\n")
                else:
                    log_file.write(f"❌ Training failed with exit code {return_code}\n")
                log_file.flush()
                log_file.close()
        except ValueError:
            # File already closed, that's okay
            pass
    
    except Exception as e:
        error_msg = f"❌ Error starting training: {str(e)}"
        import traceback
        tb = traceback.format_exc()
        
        # Write to log file (should exist since we create it at start)
        try:
            if 'log_file' in locals() and not log_file.closed:
                log_file.write(error_msg + "\n")
                log_file.write(tb + "\n")
                log_file.flush()
                log_file.close()
            else:
                # If log file doesn't exist or is closed, write to error file
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
                    f.write(tb + "\n")
        except:
            # Last resort: write to error file
            try:
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
                    f.write(tb + "\n")
            except:
                pass

def main():
    # JavaScript to force top padding as fallback
    st.components.v1.html("""
    <script>
        // Force 30px top padding on page load
        window.addEventListener('load', function() {
            setTimeout(function() {
                const selectors = [
                    'section.main > div.block-container',
                    '.stApp section.main > div',
                    'div.block-container',
                    'section[data-testid="stMain"] > div'
                ];
                selectors.forEach(function(selector) {
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(function(el) {
                        el.style.paddingTop = '30px';
                        el.style.marginTop = '0px';
                    });
                });
            }, 100);
        });
    </script>
    """, height=0)
    
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
                st.markdown("### 📚 Curated Models for Fine-tuning")
                
                # Add CSS for bigger text and scrollable container
                st.markdown("""
                <style>
                .scrollable-models-container {
                    max-height: 600px;
                    overflow-y: auto;
                    padding-right: 10px;
                }
                .scrollable-models-container::-webkit-scrollbar {
                    width: 8px;
                }
                .scrollable-models-container::-webkit-scrollbar-track {
                    background: rgba(255,255,255,0.1);
                    border-radius: 10px;
                }
                .scrollable-models-container::-webkit-scrollbar-thumb {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px;
                }
                .scrollable-models-container::-webkit-scrollbar-thumb:hover {
                    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
                }
                .model-card {
                    background: rgba(102, 126, 234, 0.1);
                    padding: 1rem;
                    border-radius: 10px;
                    border-left: 4px solid #667eea;
                    margin-bottom: 1rem;
                }
                .model-card-content {
                    display: flex;
                    gap: 1rem;
                    align-items: flex-start;
                }
                .model-main-info {
                    flex: 0 0 60%;
                    min-width: 0;
                }
                .model-description {
                    flex: 1;
                    max-height: 80px;
                    overflow-y: auto;
                    padding: 0.5rem;
                    background: rgba(255,255,255,0.05);
                    border-radius: 6px;
                    border-left: 2px solid rgba(102, 126, 234, 0.5);
                    font-size: 0.85rem;
                    color: #cccccc;
                    line-height: 1.4;
                }
                .model-description::-webkit-scrollbar {
                    width: 4px;
                }
                .model-description::-webkit-scrollbar-track {
                    background: rgba(255,255,255,0.05);
                    border-radius: 4px;
                }
                .model-description::-webkit-scrollbar-thumb {
                    background: rgba(102, 126, 234, 0.6);
                    border-radius: 4px;
                }
                .model-description::-webkit-scrollbar-thumb:hover {
                    background: rgba(102, 126, 234, 0.8);
                }
                .model-name {
                    font-size: 1.2rem;
                    font-weight: bold;
                    color: #ffffff;
                    margin-bottom: 0.5rem;
                }
                .model-info {
                    font-size: 1rem;
                    color: #cccccc;
                    margin: 0.3rem 0;
                }
                .new-badge {
                    background: linear-gradient(90deg, #ff7f0e 0%, #ff4500 100%);
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 0.7rem;
                    font-weight: bold;
                    margin-left: 8px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                downloaded_models = load_downloaded_models()
                
                # Separate newest and popular models
                newest_models = []
                popular_models = []
                
                for name, info in MODEL_PRESETS.items():
                    if info["id"] is not None:
                        if info.get("category") == "newest":
                            newest_models.append((name, info))
                        else:
                            popular_models.append((name, info))
                
                # Combine: newest first, then popular
                models_list = newest_models + popular_models
                
                # Wrap in scrollable div
                st.markdown('<div class="scrollable-models-container">', unsafe_allow_html=True)
                
                # Create 2-column grid
                for i in range(0, len(models_list), 2):
                    cols = st.columns(2)
                    
                    for col_idx, col in enumerate(cols):
                        if i + col_idx < len(models_list):
                            model_name, model_info = models_list[i + col_idx]
                            model_id = model_info["id"]
                            is_downloaded = is_model_downloaded(model_id)
                            
                            with col:
                                # Category badge for newest models
                                category_badge = ""
                                if model_info.get("category") == "newest":
                                    category_badge = '<span class="new-badge">NEW</span>'
                                
                                st.markdown(f"""
                                <div class="model-card">
                                    <div class="model-card-content">
                                        <div class="model-main-info">
                                            <div class="model-name">{model_name}{category_badge}</div>
                                            <div class="model-info">📦 {model_info['size']}</div>
                                            <div class="model-info">🆔 {model_id}</div>
                                        </div>
                                        <div class="model-description">
                                            {model_info.get('description', 'No description available')}
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Status and button row
                                btn_col1, btn_col2 = st.columns([1, 1])
                                with btn_col1:
                                    if is_downloaded:
                                        st.success("✅ Ready")
                                    else:
                                        st.info("📥 Not Downloaded")
                                
                                with btn_col2:
                                    download_key = f"download_{model_id}"
                                    if model_id in st.session_state.downloading_models:
                                        st.warning("⏳ Downloading...")
                                    elif is_downloaded:
                                        if st.button("🗑️", key=f"remove_{model_id}", use_container_width=True):
                                            model_name_clean = model_id.replace("/", "_")
                                            model_path = f"./models/{model_name_clean}"
                                            import shutil
                                            try:
                                                shutil.rmtree(model_path)
                                                st.success(f"Removed")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Error: {e}")
                                    else:
                                        if st.button("📥 Download", key=download_key, type="primary", use_container_width=True):
                                            st.session_state.downloading_models[model_id] = True
                                            st.rerun()
                                
                                if model_id in st.session_state.downloading_models:
                                    with st.status(f"Downloading...", expanded=True) as status:
                                        try:
                                            progress_bar = st.progress(0)
                                            download_path = download_model(model_id)
                                            progress_bar.progress(1.0)
                                            status.update(label=f"✅ Complete!", state="complete")
                                            del st.session_state.downloading_models[model_id]
                                            time.sleep(1)
                                            st.rerun()
                                        except Exception as e:
                                            status.update(label=f"❌ Failed", state="error")
                                            del st.session_state.downloading_models[model_id]
                                            st.error(f"Error: {str(e)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.subheader("Search Hugging Face Hub")
                st.markdown("Search for models on Hugging Face Hub:")
                
                # Add CSS for scrollable description box
                st.markdown("""
                <style>
                /* Scrollable description box styling */
                div[style*="overflow-y: auto"]::-webkit-scrollbar {
                    width: 6px;
                }
                div[style*="overflow-y: auto"]::-webkit-scrollbar-track {
                    background: rgba(255,255,255,0.1);
                    border-radius: 10px;
                }
                div[style*="overflow-y: auto"]::-webkit-scrollbar-thumb {
                    background: rgba(102, 126, 234, 0.5);
                    border-radius: 10px;
                }
                div[style*="overflow-y: auto"]::-webkit-scrollbar-thumb:hover {
                    background: rgba(102, 126, 234, 0.8);
                }
                </style>
                """, unsafe_allow_html=True)
                
                search_query = st.text_input("Search models", placeholder="e.g., llama instruct, mistral, qwen")
                
                if search_query:
                    with st.spinner("Searching..."):
                        results = search_huggingface_models(search_query, limit=20)
                    
                    if results:
                        st.success(f"Found {len(results)} models")
                        
                        for model in results:
                            model_id = model["id"]
                            is_downloaded = is_model_downloaded(model_id)
                            
                            # Fetch description
                            description = fetch_model_description(model_id)
                            
                            with st.container():
                                # New layout: 60% info, 40% description
                                info_col, desc_col = st.columns([3, 2])
                                
                                with info_col:
                                    st.markdown(f"**{model_id}**")
                                    downloads_count = model.get("downloads", 0) or 0
                                    st.caption(f"📥 Downloads: {downloads_count:,}")
                                    tags_list = model.get("tags", [])[:5] if model.get("tags") else []
                                    if tags_list:
                                        st.caption(f"🏷️ {', '.join(tags_list)}")
                                    
                                    # Status and button in sub-columns
                                    status_col, btn_col = st.columns([1, 1])
                                    with status_col:
                                        if is_downloaded:
                                            st.success("✅ Ready")
                                        else:
                                            st.info("📥 Available")
                                    
                                    with btn_col:
                                        download_key = f"download_hf_{model_id}"
                                        if model_id in st.session_state.downloading_models:
                                            st.warning("⏳...")
                                        elif is_downloaded:
                                            if st.button("🗑️", key=f"remove_hf_{model_id}", use_container_width=True):
                                                model_name_clean = model_id.replace("/", "_")
                                                model_path = f"./models/{model_name_clean}"
                                                import shutil
                                                try:
                                                    shutil.rmtree(model_path)
                                                    st.success(f"Removed")
                                                    st.rerun()
                                                except Exception as e:
                                                    st.error(f"Error: {e}")
                                        else:
                                            if st.button("📥", key=download_key, type="primary", use_container_width=True):
                                                st.session_state.downloading_models[model_id] = True
                                                st.rerun()
                                
                                with desc_col:
                                    # Scrollable description box
                                    st.markdown(f'''
                                    <div style="
                                        height: 120px;
                                        overflow-y: auto;
                                        padding: 10px;
                                        background: rgba(102, 126, 234, 0.1);
                                        border-radius: 8px;
                                        border-left: 3px solid #667eea;
                                        font-size: 0.9rem;
                                        color: #cccccc;
                                    ">
                                        {description}
                                    </div>
                                    ''', unsafe_allow_html=True)
                                
                                st.markdown("---")
                                
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
                    try:
                        with open(data_path, "r", encoding='utf-8') as f:
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
                    except Exception as e:
                        st.error(f"Error reading dataset: {str(e)}")
                
                # Dataset stats
                try:
                    with open(data_path, "r", encoding='utf-8') as f:
                        total_lines = sum(1 for _ in f)
                    st.metric("Total Examples", total_lines)
                except Exception as e:
                    st.error(f"Error counting examples: {str(e)}")
            else:
                data_path = st.text_input("Or enter path to existing dataset", value="train_data.jsonl")
                if os.path.exists(data_path):
                    st.success(f"✅ Found dataset: {data_path}")
                    try:
                        with open(data_path, "r", encoding='utf-8') as f:
                            total_lines = sum(1 for _ in f)
                        st.metric("Total Examples", total_lines)
                    except Exception as e:
                        st.error(f"Error reading dataset: {str(e)}")
        
        with col2:
            # Analyze selected model and show profile card
            model_profile = None
            recommended_params = None
            dataset_size = 10  # Default
            
            if os.path.exists("train_data.jsonl"):
                try:
                    with open("train_data.jsonl", "r", encoding='utf-8') as f:
                        dataset_size = sum(1 for _ in f)
                except:
                    dataset_size = 10  # Default if error
            
            if model_name:
                model_profile = analyze_model(model_name)
                recommended_params = get_recommended_params(model_profile, dataset_size)
            
            # Model Profile Card
            if model_profile:
                st.markdown("""
                <style>
                .profile-card {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 1rem;
                    border-radius: 12px;
                    margin-bottom: 1rem;
                    color: white;
                }
                .profile-title {
                    font-size: 1.1rem;
                    font-weight: bold;
                    margin-bottom: 0.5rem;
                }
                .profile-info {
                    display: flex;
                    gap: 1rem;
                    margin: 0.5rem 0;
                }
                .profile-badge {
                    background: rgba(255,255,255,0.2);
                    padding: 0.3rem 0.6rem;
                    border-radius: 6px;
                    font-size: 0.9rem;
                }
                .safety-optimal {
                    color: #10b981;
                    font-weight: bold;
                }
                .safety-warning {
                    color: #fbbf24;
                    font-weight: bold;
                }
                .safety-danger {
                    color: #ef4444;
                    font-weight: bold;
                }
                </style>
                """, unsafe_allow_html=True)
                
                type_color = "🟢" if model_profile["type"] == "Instruct" else "🔵"
                st.markdown(f"""
                <div class="profile-card">
                    <div class="profile-title">📊 MODEL PROFILE</div>
                    <div class="profile-info">
                        <span class="profile-badge">{type_color} {model_profile["type"]}</span>
                        <span class="profile-badge">📏 {model_profile["size"]}</span>
                        <span class="profile-badge">⚡ {model_profile["quantization"]}</span>
                    </div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                        {'✅ Optimized for fine-tuning' if model_profile["type"] == "Instruct" else '⚠️ Base model - needs careful tuning'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <style>
            /* Compact futuristic slider styling */
            .stSlider > div > div > div {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            }
            .stSlider > div > div > div > div {
                background-color: white;
                border: 2px solid #667eea;
            }
            .stNumberInput > div > div > input {
                background-color: rgba(102, 126, 234, 0.1);
                border: 1px solid #667eea;
                border-radius: 8px;
                padding: 0.5rem;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.subheader("⚙️ Training Parameters")
            
            # Auto-Apply Recommended Settings Button
            if recommended_params:
                if st.button("✨ Use Recommended Settings", type="primary", use_container_width=True):
                    st.session_state.smart_epochs = recommended_params["epochs"]
                    st.session_state.smart_batch_size = recommended_params["batch_size"]
                    st.session_state.smart_lora_r = recommended_params["lora_r"]
                    st.session_state.smart_lora_alpha = recommended_params["lora_alpha"]
                    st.session_state.smart_lora_dropout = recommended_params["lora_dropout"]
                    st.session_state.smart_grad_accum = recommended_params["grad_accum"]
                    st.session_state.smart_max_seq = recommended_params["max_seq_length"]
                    st.rerun()
            
            # Model Name Input
            model_display_name = st.text_input(
                "📝 Model Name",
                value=f"My_Model_{datetime.now().strftime('%m%d')}",
                help="Give your model a memorable name",
                placeholder="e.g., CustomerService_Bot_v1"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                # Epochs with safety indicator
                epochs = st.slider(
                    "Epochs", 1, 10, 
                    st.session_state.get("smart_epochs", 1),
                    help="Number of training epochs"
                )
                if recommended_params:
                    status, msg = check_parameter_safety("epochs", epochs, recommended_params["epochs"], model_profile)
                    if status == "optimal":
                        st.markdown(f'<span class="safety-optimal">{msg}</span>', unsafe_allow_html=True)
                    elif status == "warning":
                        st.markdown(f'<span class="safety-warning">{msg}</span>', unsafe_allow_html=True)
                    elif status == "danger":
                        st.markdown(f'<span class="safety-danger">{msg}</span>', unsafe_allow_html=True)
                
                lora_r = st.slider("LoRA R", 4, 64, st.session_state.get("smart_lora_r", 8), step=4, help="LoRA rank")
                max_seq_length = st.number_input("Max Seq Length", 512, 4096, st.session_state.get("smart_max_seq", 2048), step=256)
            
            with col_b:
                # Batch Size with safety indicator
                batch_size = st.slider(
                    "Batch Size", 1, 8, 
                    st.session_state.get("smart_batch_size", 1),
                    help="⚠️ Use 1 for 8B models"
                )
                if recommended_params:
                    status, msg = check_parameter_safety("batch_size", batch_size, recommended_params["batch_size"], model_profile)
                    if status == "warning":
                        st.markdown(f'<span class="safety-warning">{msg}</span>', unsafe_allow_html=True)
                    elif status == "optimal":
                        st.markdown(f'<span class="safety-optimal">{msg}</span>', unsafe_allow_html=True)
                
                lora_alpha = st.slider("LoRA Alpha", 8, 128, st.session_state.get("smart_lora_alpha", 16), step=8, help="LoRA alpha scaling")
                grad_accum = st.slider("Grad Accum", 1, 32, st.session_state.get("smart_grad_accum", 8), help="Gradient accumulation steps")
            
            # Advanced in expander
            with st.expander("🔧 Advanced Settings"):
                lora_dropout = st.slider("LoRA Dropout", 0.0, 0.5, st.session_state.get("smart_lora_dropout", 0.05), step=0.01)
                max_examples = st.number_input("Max Examples (0 = all)", 0, 10000, 0)
                
                # Show learning rate info
                if recommended_params:
                    st.info(f"💡 Recommended Learning Rate: {recommended_params['learning_rate']:.1e}")
            
            # Device info
            st.divider()
            if not TORCH_AVAILABLE:
                st.error("⚠️ PyTorch error - Training will use CPU")
            elif st.session_state.gpu_available:
                st.success(f"✅ GPU: {st.session_state.gpu_name}")
            else:
                st.warning("⚠️ CPU Mode")
        
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
                # Generate output directory with custom name
                # Sanitize model name (remove special chars, replace spaces with underscores)
                safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in model_display_name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"./models_trained/{safe_name}_{timestamp}"
                
                # Create models_trained directory if it doesn't exist
                os.makedirs("./models_trained", exist_ok=True)
                
                config = {
                    "model_name": model_name,
                    "model_display_name": model_display_name,
                    "data_path": data_path,
                    "output_dir": output_dir,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "grad_accum": grad_accum,
                    "max_seq_length": max_seq_length,
                    "max_examples": max_examples if max_examples > 0 else None,
                    "timestamp": timestamp,
                }
                
                # Save metadata JSON
                metadata_file = os.path.join(output_dir, "training_metadata.json")
                os.makedirs(output_dir, exist_ok=True)
                with open(metadata_file, "w") as f:
                    json.dump(config, f, indent=2)
                
                # Check if training is already running
                if st.session_state.training_status == "training":
                    st.warning("⚠️ Training is already in progress. Please stop it first.")
                else:
                        # Pre-flight checks
                        import sys
                        python_exe = sys.executable
                        train_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_basic.py")
                        
                        # Verify files exist
                        checks_passed = True
                        if not os.path.exists(train_script_path):
                            st.error(f"❌ train_basic.py not found at: {train_script_path}")
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
                        st.session_state.output_dir = output_dir  # Store output dir for progress file
                        st.session_state.training_logs = []
                        st.session_state.training_logs.append("=" * 60)
                        st.session_state.training_logs.append("🚀 Starting training...")
                        st.session_state.training_logs.append(f"Python: {python_exe}")
                        st.session_state.training_logs.append(f"Script: {train_script_path}")
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
                                f.write(f"train_basic.py exists: {os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_basic.py'))}\n")
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
            # FUTURISTIC TRAINING VISUALIZATION
            
            # CSS for futuristic design
            st.markdown("""
            <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            @keyframes shimmer {
                0% { background-position: -1000px 0; }
                100% { background-position: 1000px 0; }
            }
            .training-banner {
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
                border: 2px solid rgba(102, 126, 234, 0.5);
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
            }
            .training-status {
                font-size: 2rem;
                font-weight: bold;
                color: #fff;
                animation: pulse 2s infinite;
            }
            .metric-card {
                background: linear-gradient(135deg, rgba(30, 30, 60, 0.8) 0%, rgba(20, 20, 40, 0.9) 100%);
                border: 1px solid rgba(102, 126, 234, 0.4);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 0.5rem;
                box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
                backdrop-filter: blur(10px);
            }
            .metric-label {
                font-size: 0.9rem;
                color: #a0a0ff;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .metric-value {
                font-size: 2.5rem;
                font-weight: bold;
                color: #fff;
                text-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
                margin: 0.5rem 0;
            }
            .progress-container {
                background: rgba(20, 20, 40, 0.5);
                border-radius: 10px;
                padding: 0.3rem;
                margin: 1rem 0;
            }
            .futuristic-progress {
                height: 30px;
                background: linear-gradient(90deg, 
                    #667eea 0%, 
                    #764ba2 25%, 
                    #f093fb 50%, 
                    #ff7f0e 75%, 
                    #ff4500 100%);
                background-size: 200% 100%;
                animation: shimmer 3s infinite linear;
                border-radius: 8px;
                box-shadow: 0 0 15px rgba(102, 126, 234, 0.6);
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Training Status Banner
            st.markdown("""
            <div class="training-banner">
                <div class="training-status">⚡ TRAINING IN PROGRESS</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Read progress from JSON file
            progress_file = os.path.join(st.session_state.get("output_dir", "./fine_tuned_adapter"), "training_progress.json")
            st.caption(f"📂 Looking for progress at: {progress_file}")
            progress_data = None
            
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                except:
                    pass
            
            if progress_data:
                # Progress bars
                epoch = progress_data.get("epoch", 0)
                total_epochs = progress_data.get("total_epochs", 1)
                step = progress_data.get("step", 0)
                total_steps = progress_data.get("total_steps", 1)
                
                epoch_progress = epoch / total_epochs if total_epochs > 0 else 0
                step_progress = step / total_steps if total_steps > 0 else 0
                
                # Epoch Progress
                st.markdown(f"### 🔄 Epoch {int(epoch)}/{int(total_epochs)}")
                st.progress(epoch_progress)
                
                # Step Progress
                st.markdown(f"### 📊 Step {step}/{total_steps} ({int(step_progress * 100)}%)")
                st.progress(step_progress)
                
                # Time info
                elapsed = progress_data.get("elapsed_time", 0)
                eta = progress_data.get("eta_seconds", 0)
                elapsed_str = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
                eta_str = f"{eta // 60:02d}:{eta % 60:02d}"
                
                col_time1, col_time2 = st.columns(2)
                with col_time1:
                    st.metric("⏱️ Elapsed", elapsed_str)
                with col_time2:
                    st.metric("⏳ ETA", eta_str)
                
                # Metric Cards in 2x2 grid
                st.markdown("### 📈 Live Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    loss = progress_data.get("loss", 0)
                    # Color based on loss (red if high, green if low)
                    loss_color = "#10b981" if loss < 1.0 else ("#ff7f0e" if loss < 2.0 else "#ff4500")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Current Loss</div>
                        <div class="metric-value" style="color: {loss_color};">{loss:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    lr = progress_data.get("learning_rate", 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Learning Rate</div>
                        <div class="metric-value">{lr:.2e}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    speed = progress_data.get("samples_per_second", 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🚀 Speed</div>
                        <div class="metric-value">{speed:.2f} samples/s</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    gpu_used = progress_data.get("gpu_memory_used_gb", 0)
                    gpu_total = progress_data.get("gpu_memory_total_gb", 1)
                    gpu_percent = (gpu_used / gpu_total * 100) if gpu_total > 0 else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🎮 GPU Memory</div>
                        <div class="metric-value">{gpu_used:.1f} / {gpu_total:.1f} GB</div>
                        <div style="margin-top: 0.5rem;">
                            <div style="background: rgba(255,255,255,0.1); border-radius: 5px; height: 10px;">
                                <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: {gpu_percent}%; height: 100%; border-radius: 5px;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Loss History Graph
                loss_history = progress_data.get("loss_history", [])
                if loss_history:
                    st.markdown("### 📉 Loss Over Time")
                    import pandas as pd
                    df = pd.DataFrame(loss_history)
                    st.line_chart(df.set_index("step")["loss"])
            else:
                st.info("⏳ Waiting for training data...")
            
            # Stop button
            if st.button("🛑 Stop Training", key="stop_training_main", type="primary"):
                st.session_state.training_status = "idle"
                st.warning("Training stopped by user")
                st.rerun()
            
            # Read logs from file and display
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.txt")
            
            # Detailed logs in expander
            with st.expander("📄 View Detailed Logs", expanded=False):
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                            logs = f.read()
                            st.caption(f"Log file size: {len(logs)} characters | Last updated: {time.strftime('%H:%M:%S')}")
                            st.text_area("Training Logs", logs, height=400, key="live_logs")
                            
                            # Check if training finished or had errors
                            if "✅ Training completed successfully" in logs or "Done! Model saved" in logs:
                                st.session_state.training_status = "completed"
                                st.success("✅ Training completed!")
                                st.rerun()
                            elif "❌ Training failed" in logs or "OutOfMemoryError" in logs or "CUDA out of memory" in logs:
                                st.session_state.training_status = "failed"
                                st.error("❌ Training failed - Check logs for details")
                                if "OutOfMemoryError" in logs or "CUDA out of memory" in logs:
                                    st.error("💡 **Out of Memory!** Reduce batch size or max sequence length")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error reading logs: {e}")
                else:
                    st.warning("⚠️ Log file not found yet. Waiting for training to start...")
            
            # Auto-refresh every 1 second for smooth updates
            time.sleep(1)
            st.rerun()
        
        # Training completed or idle status messages
            
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
        st.header("🧪 Test Models - Side-by-Side Chat")
        
        # Load all available models
        available_models = load_all_available_models()
        
        if not available_models:
            st.warning("⚠️ No models available. Download a base model or train a fine-tuned model first.")
        else:
            # Initialize chat history
            if 'dual_chat_history' not in st.session_state:
                st.session_state.dual_chat_history = []
            
            # Model selection at top
            col_select1, col_select2 = st.columns(2)
            
            with col_select1:
                st.markdown("### 🔵 Model A")
                model_a_options = ["None"] + [m["name"] for m in available_models]
                selected_a = st.selectbox("", model_a_options, key="model_a", label_visibility="collapsed")
                model_a = None
                if selected_a != "None":
                    model_a = next(m for m in available_models if m["name"] == selected_a)
            
            with col_select2:
                st.markdown("### 🟢 Model B")
                model_b_options = ["None"] + [m["name"] for m in available_models]
                selected_b = st.selectbox("", model_b_options, key="model_b", label_visibility="collapsed")
                model_b = None
                if selected_b != "None":
                    model_b = next(m for m in available_models if m["name"] == selected_b)
            
            st.divider()
            
            # CSS for chat bubbles
            st.markdown("""
            <style>
            .user-bubble {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.8rem;
                border-radius: 12px;
                margin: 0.5rem 0;
                text-align: right;
                margin-left: 20%;
            }
            .ai-bubble-a {
                background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
                color: white;
                padding: 0.8rem;
                border-radius: 12px;
                margin: 0.5rem 0;
                margin-right: 20%;
                border-left: 4px solid #667eea;
            }
            .ai-bubble-b {
                background: linear-gradient(135deg, #065f46 0%, #064e3b 100%);
                color: white;
                padding: 0.8rem;
                border-radius: 12px;
                margin: 0.5rem 0;
                margin-right: 20%;
                border-left: 4px solid #10b981;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Two-column chat display with synchronized rows
            st.markdown("#### 💬 Chat History")
            
            if not st.session_state.dual_chat_history:
                st.info("No messages yet")
            else:
                import html as html_module
                
                # Loop through messages and display them in synchronized rows
                for msg in st.session_state.dual_chat_history:
                    # User message row - synchronized across both columns
                    col_user1, col_user2 = st.columns(2)
                    with col_user1:
                        user_content = html_module.escape(msg["user"])
                        st.markdown(f'<div class="user-bubble">{user_content}</div>', unsafe_allow_html=True)
                    with col_user2:
                        user_content = html_module.escape(msg["user"])
                        st.markdown(f'<div class="user-bubble">{user_content}</div>', unsafe_allow_html=True)
                    
                    # AI response row - synchronized across both columns
                    col_ai1, col_ai2 = st.columns(2)
                    
                    with col_ai1:
                        # Model A response
                        if msg.get("response_a"):
                            ai_content = html_module.escape(msg["response_a"])
                            ai_content = ai_content.replace('\n', '<br>')
                            st.markdown(f'<div class="ai-bubble-a">{ai_content}</div>', unsafe_allow_html=True)
                        elif msg.get("generating_a"):
                            st.caption("⏳ Generating...")
                        else:
                            st.caption("⚠️ Model A not selected")
                    
                    with col_ai2:
                        # Model B response
                        if msg.get("response_b"):
                            ai_content = html_module.escape(msg["response_b"])
                            ai_content = ai_content.replace('\n', '<br>')
                            st.markdown(f'<div class="ai-bubble-b">{ai_content}</div>', unsafe_allow_html=True)
                        elif msg.get("generating_b"):
                            st.caption("⏳ Generating...")
                        else:
                            st.caption("⚠️ Model B not selected")
            
            st.divider()
            
            # Chat input at bottom
            with st.form(key="dual_chat_form", clear_on_submit=True):
                col_input, col_send, col_clear = st.columns([8, 1, 1])
                with col_input:
                    user_input = st.text_input("Message", placeholder="Type your message...", label_visibility="collapsed")
                with col_send:
                    send_button = st.form_submit_button("Send", use_container_width=True, type="primary")
                with col_clear:
                    clear_button = st.form_submit_button("Clear", use_container_width=True)
            
            if clear_button:
                st.session_state.dual_chat_history = []
                st.rerun()
            
            if send_button and user_input:
                if model_a is None and model_b is None:
                    st.error("Please select at least one model")
                else:
                    # Add user message immediately and show it
                    new_message = {
                        "user": user_input,
                        "generating_a": model_a is not None,
                        "generating_b": model_b is not None
                    }
                    st.session_state.dual_chat_history.append(new_message)
                    st.session_state.generating_dual = True
                    st.rerun()  # Show user message immediately
            
            # Generate responses after showing user message
            if st.session_state.get("generating_dual", False):
                st.session_state.generating_dual = False
                
                # Get last message
                last_msg = st.session_state.dual_chat_history[-1]
                user_input = last_msg["user"]
                
                # Generate responses
                if model_a and last_msg.get("generating_a"):
                    with st.spinner("Model A thinking..."):
                        response_a = generate_model_response(model_a, user_input)
                        last_msg["response_a"] = response_a
                        last_msg["generating_a"] = False
                
                if model_b and last_msg.get("generating_b"):
                    with st.spinner("Model B thinking..."):
                        response_b = generate_model_response(model_b, user_input)
                        last_msg["response_b"] = response_b
                        last_msg["generating_b"] = False
                
                st.rerun()
    
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

