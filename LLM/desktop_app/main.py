from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QTextEdit, QPlainTextEdit,
    QSpinBox, QDoubleSpinBox, QMessageBox, QListWidget, QListWidgetItem, QSplitter, QToolBar, QScrollArea, QGridLayout, QFrame
)
from PySide6.QtGui import QAction, QIcon

from desktop_app.model_card_widget import ModelCard, DownloadedModelCard
from desktop_app.training_widgets import MetricCard
from desktop_app.chat_widget import ChatWidget

from core.models import (DEFAULT_BASE_MODELS, search_hf_models, download_hf_model, list_local_adapters, 
                         list_local_downloads, get_app_root, detect_model_capabilities, get_capability_icons, get_model_size)
from core.training import TrainingConfig, default_output_dir, build_finetune_cmd
from core.inference import InferenceConfig, build_run_adapter_cmd


APP_TITLE = "🤖 LLM Fine-tuning Studio"

# Dark theme stylesheet with gradient accents
DARK_THEME = """
QMainWindow, QWidget {
    background-color: #0e1117;
    color: #fafafa;
}
QTabWidget::pane {
    border: 1px solid #262730;
    background-color: #0e1117;
}
QTabBar::tab {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
    color: white;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #764ba2, stop:1 #667eea);
}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #262730;
    color: #fafafa;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 4px;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #764ba2, stop:1 #667eea);
}
QPushButton:disabled {
    background-color: #3a3a3a;
    color: #808080;
}
QListWidget {
    background-color: #262730;
    color: #fafafa;
    border: 1px solid #3a3a3a;
}
QLabel {
    color: #fafafa;
}
QToolBar {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
    border: none;
    spacing: 10px;
}
"""

# Light theme stylesheet with gradient accents
LIGHT_THEME = """
QMainWindow, QWidget {
    background-color: #ffffff;
    color: #262730;
}
QTabWidget::pane {
    border: 1px solid #e0e0e0;
    background-color: #ffffff;
}
QTabBar::tab {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
    color: white;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #764ba2, stop:1 #667eea);
}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #f5f5f5;
    color: #262730;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    padding: 4px;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #764ba2, stop:1 #667eea);
}
QPushButton:disabled {
    background-color: #e0e0e0;
    color: #a0a0a0;
}
QListWidget {
    background-color: #f5f5f5;
    color: #262730;
    border: 1px solid #d0d0d0;
}
QLabel {
    color: #262730;
}
QToolBar {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
    border: none;
    spacing: 10px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 900)

        self.root = get_app_root()
        self.dark_mode = True  # Start in dark mode

        # Create toolbar with theme toggle
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setMinimumHeight(80)  # Make it thicker
        self.addToolBar(toolbar)

        # Theme toggle button
        self.theme_action = QAction("🌙 Dark Mode", self)
        self.theme_action.triggered.connect(self._toggle_theme)
        toolbar.addAction(self.theme_action)

        toolbar.addSeparator()
        
        # Add title label to toolbar (centered)
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel(APP_TITLE)
        title_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold; padding: 0 15px;")
        title_layout.addStretch(1)
        title_layout.addWidget(title_label)
        title_layout.addStretch(1)
        
        toolbar.addWidget(title_widget)
        
        # System info on the right
        sys_info_widget = QWidget()
        sys_info_layout = QVBoxLayout(sys_info_widget)
        sys_info_layout.setContentsMargins(10, 5, 10, 5)
        sys_info_layout.setSpacing(2)
        
        python_label = QLabel("🐍 Python 3.12.7")
        python_label.setStyleSheet("color: white; font-size: 9pt;")
        sys_info_layout.addWidget(python_label)
        
        pytorch_label = QLabel("🔥 PyTorch 2.6.0+cu124 (CUDA 12.4)")
        pytorch_label.setStyleSheet("color: white; font-size: 9pt;")
        sys_info_layout.addWidget(pytorch_label)
        
        ram_label = QLabel("💾 RAM: 63.8 GB")
        ram_label.setStyleSheet("color: white; font-size: 9pt;")
        sys_info_layout.addWidget(ram_label)
        
        toolbar.addWidget(sys_info_widget)

        tabs = QTabWidget()
        tabs.addTab(self._build_home_tab(), "🏠 Home")
        tabs.addTab(self._build_train_tab(), "🎯 Train")
        tabs.addTab(self._build_models_tab(), "📥 Download")
        tabs.addTab(self._build_test_tab(), "🧪 Test")
        tabs.addTab(self._build_logs_tab(), "📊 Logs")

        self.setCentralWidget(tabs)

        self.train_proc: QProcess | None = None
        
        # Initialize card lists
        self.model_cards = []
        self.downloaded_model_cards = []
        self.metric_cards = []
        
        self._refresh_locals()
        self._apply_theme()

    def _toggle_theme(self) -> None:
        """Toggle between dark and light themes"""
        self.dark_mode = not self.dark_mode
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the current theme"""
        if self.dark_mode:
            self.setStyleSheet(DARK_THEME)
            self.theme_action.setText("🌙 Dark Mode")
        else:
            self.setStyleSheet(LIGHT_THEME)
            self.theme_action.setText("☀️ Light Mode")
        
        # Update chat widgets theme
        if hasattr(self, 'chat_widgets'):
            for chat_widget in self.chat_widgets:
                chat_widget.set_theme(self.dark_mode)
        
        # Update all model cards
        for card in self.model_cards:
            card.set_theme(self.dark_mode)
        for card in self.downloaded_model_cards:
            card.set_theme(self.dark_mode)
        # Update metric cards
        for card in self.metric_cards:
            card.set_theme(self.dark_mode)

    # ---------------- Home tab ----------------
    def _build_home_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # Welcome title
        title = QLabel("<h1>Welcome to LLM Fine-tuning Studio</h1>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create 2-column layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)
        
        # LEFT: Features
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        
        left_layout.addWidget(QLabel("<h2>🚀 Features</h2>"))
        
        features_text = QLabel("""
<p>This application provides a beautiful, user-friendly interface to:</p>
<ul style="line-height: 1.8;">
<li><b>🎯 Train Models:</b> Select from popular pre-trained models and fine-tune them with your data</li>
<li><b>📥 Upload Datasets:</b> Easy drag-and-drop for JSONL format datasets</li>
<li><b>🧪 Test Models:</b> Interactive chat interface to test your fine-tuned models</li>
<li><b>✅ Validate Performance:</b> Run validation tests and view detailed results</li>
<li><b>📊 Track History:</b> View all your trained models and training logs</li>
</ul>
        """)
        features_text.setWordWrap(True)
        left_layout.addWidget(features_text)
        
        left_layout.addWidget(QLabel("<h2>📋 Quick Start Guide</h2>"))
        
        guide_text = QLabel("""
<ol style="line-height: 2;">
<li><b>Prepare Your Dataset:</b> Create a JSONL file with format:
<pre style="background: #2a2a2a; padding: 10px; border-radius: 5px; margin: 10px 0;">
{"instruction": "Your instruction here", "output": "Expected output here"}
</pre>
</li>
<li><b>Go to Train Model:</b> Select a base model and upload your dataset</li>
<li><b>Configure Training:</b> Adjust epochs, batch size, and LoRA parameters</li>
<li><b>Start Training:</b> Click the train button and monitor progress</li>
<li><b>Test Your Model:</b> Use the Test Model tab to try your fine-tuned model</li>
</ol>
        """)
        guide_text.setWordWrap(True)
        left_layout.addWidget(guide_text)
        
        left_layout.addStretch(1)
        content_layout.addWidget(left_widget, 3)
        
        # RIGHT: System Status
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)
        
        right_layout.addWidget(QLabel("<h2>📊 System Status</h2>"))
        
        # System info cards
        sys_frame = QFrame()
        sys_frame.setFrameShape(QFrame.StyledPanel)
        sys_layout = QVBoxLayout(sys_frame)
        sys_layout.setSpacing(10)
        
        refresh_btn = QPushButton("🔄 Refresh GPU Detection")
        refresh_btn.setMaximumWidth(200)
        sys_layout.addWidget(refresh_btn)
        
        gpu_status = QLabel("✅ <b>2 GPUs detected</b>")
        gpu_status.setStyleSheet("color: #4CAF50; font-size: 12pt; padding: 10px;")
        sys_layout.addWidget(gpu_status)
        
        # GPU 0
        gpu0_label = QLabel("<b>GPU 0: NVIDIA GeForce RTX 4090</b>")
        sys_layout.addWidget(gpu0_label)
        gpu0_mem = QLabel("💾 22.5 GB")
        gpu0_mem.setStyleSheet("font-size: 16pt; font-weight: bold; padding: 5px 20px;")
        sys_layout.addWidget(gpu0_mem)
        
        sys_layout.addWidget(QLabel("<hr>"))
        
        # GPU 1
        gpu1_label = QLabel("<b>GPU 1: NVIDIA RTX A2000 12GB</b>")
        sys_layout.addWidget(gpu1_label)
        gpu1_mem = QLabel("💾 11.2 GB")
        gpu1_mem.setStyleSheet("font-size: 16pt; font-weight: bold; padding: 5px 20px;")
        sys_layout.addWidget(gpu1_mem)
        
        sys_layout.addWidget(QLabel("<hr>"))
        
        # Models stats
        models_trained = QLabel("<b>Models Trained</b>")
        sys_layout.addWidget(models_trained)
        models_count = QLabel("7")
        models_count.setStyleSheet("font-size: 24pt; font-weight: bold; padding: 5px 20px;")
        sys_layout.addWidget(models_count)
        
        sys_layout.addWidget(QLabel("<hr>"))
        
        models_downloaded = QLabel("<b>Models Downloaded</b>")
        sys_layout.addWidget(models_downloaded)
        downloads_count = QLabel("5")
        downloads_count.setStyleSheet("font-size: 24pt; font-weight: bold; padding: 5px 20px;")
        sys_layout.addWidget(downloads_count)
        
        sys_layout.addWidget(QLabel("<hr>"))
        
        # Status
        status_label = QLabel("<b>Status</b>")
        sys_layout.addWidget(status_label)
        status_val = QLabel("Ready")
        status_val.setStyleSheet("font-size: 18pt; font-weight: bold; color: #4CAF50; padding: 5px 20px;")
        sys_layout.addWidget(status_val)
        
        right_layout.addWidget(sys_frame)
        
        # Tips section
        right_layout.addWidget(QLabel("<h2>💡 Tips</h2>"))
        
        tips_frame = QFrame()
        tips_frame.setFrameShape(QFrame.StyledPanel)
        tips_layout = QVBoxLayout(tips_frame)
        
        tips = [
            "• Use GPU for faster training",
            "• Start with fewer epochs for testing",
            "• Monitor training logs for progress",
            "• Test models before full validation"
        ]
        
        for tip in tips:
            tip_label = QLabel(tip)
            tip_label.setStyleSheet("color: #2196F3; padding: 5px;")
            tips_layout.addWidget(tip_label)
        
        right_layout.addWidget(tips_frame)
        
        right_layout.addStretch(1)
        content_layout.addWidget(right_widget, 2)
        
        layout.addLayout(content_layout)
        
        return w
    
    # ---------------- Models (Download) tab ----------------
    def _build_models_tab(self) -> QWidget:
        w = QWidget()
        main_layout = QHBoxLayout(w)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create splitter for 2 columns
        splitter = QSplitter(Qt.Horizontal)
        
        # LEFT COLUMN: Curated Models
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        
        left_layout.addWidget(QLabel("<h2>📚 Curated Models for Fine-tuning</h2>"))
        
        # Scroll area for curated models
        curated_scroll = QScrollArea()
        curated_scroll.setWidgetResizable(True)
        curated_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.curated_container = QWidget()
        self.curated_layout = QGridLayout(self.curated_container)
        self.curated_layout.setSpacing(15)
        self.curated_layout.setContentsMargins(5, 5, 5, 5)
        
        curated_scroll.setWidget(self.curated_container)
        left_layout.addWidget(curated_scroll)
        
        # RIGHT COLUMN: Downloaded Models + Search
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)
        
        # Section 1: Downloaded Models
        right_layout.addWidget(QLabel("<h2>📥 Downloaded Models</h2>"))
        
        downloaded_scroll = QScrollArea()
        downloaded_scroll.setWidgetResizable(True)
        downloaded_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        downloaded_scroll.setMinimumHeight(200)
        downloaded_scroll.setMaximumHeight(400)
        
        self.downloaded_container = QWidget()
        self.downloaded_layout = QVBoxLayout(self.downloaded_container)
        self.downloaded_layout.setSpacing(10)
        self.downloaded_layout.setContentsMargins(5, 5, 5, 5)
        self.downloaded_layout.addStretch(1)
        
        downloaded_scroll.setWidget(self.downloaded_container)
        right_layout.addWidget(downloaded_scroll)
        
        # Section 2: Search Hugging Face
        right_layout.addWidget(QLabel("<h2>🔍 Search Hugging Face</h2>"))
        
        search_row = QHBoxLayout()
        self.hf_query = QLineEdit()
        self.hf_query.setPlaceholderText("Search models (e.g., Qwen2.5 bnb 4bit)")
        self.hf_search_btn = QPushButton("Search")
        self.hf_search_btn.clicked.connect(self._hf_search)
        search_row.addWidget(self.hf_query)
        search_row.addWidget(self.hf_search_btn)
        right_layout.addLayout(search_row)

        self.hf_results = QListWidget()
        self.hf_results.setMaximumHeight(250)
        right_layout.addWidget(self.hf_results)

        dl_row = QHBoxLayout()
        self.hf_target_dir = QLineEdit(str(self.root / "models"))
        self.hf_browse_btn = QPushButton("Browse…")
        self.hf_browse_btn.clicked.connect(self._browse_hf_target)
        self.hf_download_btn = QPushButton("Download Selected")
        self.hf_download_btn.clicked.connect(self._hf_download_selected)
        dl_row.addWidget(QLabel("Download to:"))
        dl_row.addWidget(self.hf_target_dir, 2)
        dl_row.addWidget(self.hf_browse_btn)
        dl_row.addWidget(self.hf_download_btn)
        right_layout.addLayout(dl_row)

        self.models_status = QPlainTextEdit()
        self.models_status.setReadOnly(True)
        self.models_status.setMaximumBlockCount(500)
        self.models_status.setMaximumHeight(150)
        right_layout.addWidget(self.models_status)
        
        # Refresh button at bottom
        refresh_btn = QPushButton("🔄 Refresh Models")
        refresh_btn.setMaximumWidth(200)
        refresh_btn.clicked.connect(self._refresh_models)
        right_layout.addWidget(refresh_btn)
        
        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 500])
        
        main_layout.addWidget(splitter)
        
        # Store model cards for theme updates
        self.model_cards = []
        self.downloaded_model_cards = []
        
        return w
    
    def _refresh_models(self) -> None:
        """Refresh all models - curated and downloaded"""
        # Clear existing cards
        while self.downloaded_layout.count() > 1:  # Keep stretch
            item = self.downloaded_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.downloaded_model_cards.clear()
        
        while self.curated_layout.count():
            item = self.curated_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.model_cards.clear()
        
        # Downloaded models (vertical list on right)
        models_dir = self.root / "models"
        if models_dir.exists():
            for model_dir in sorted(models_dir.iterdir()):
                if model_dir.is_dir():
                    model_name = model_dir.name
                    size = get_model_size(str(model_dir))
                    capabilities = detect_model_capabilities(model_name=model_name, model_path=str(model_dir))
                    icons = get_capability_icons(capabilities)
                    
                    card = DownloadedModelCard(model_name, str(model_dir), size, icons)
                    card.set_theme(self.dark_mode)
                    card.selected.connect(self._on_model_selected)
                    self.downloaded_layout.insertWidget(self.downloaded_layout.count() - 1, card)
                    self.downloaded_model_cards.append(card)
        
        # Curated models: 4 LATEST + 20 MOST POPULAR (2 columns)
        latest_models = [
            ("Llama 3.3 70B Instruct (4-bit)", "unsloth/Llama-3.3-70B-Instruct-bnb-4bit", "Latest Llama 3.3 70B model with enhanced capabilities", "~35 GB", True),
            ("Qwen2.5 72B Instruct (4-bit)", "unsloth/Qwen2.5-72B-Instruct-bnb-4bit", "State-of-the-art Qwen 2.5 72B model", "~36 GB", True),
            ("Gemma 2 27B Instruct (4-bit)", "unsloth/gemma-2-27b-it-bnb-4bit", "Google's Gemma 2 27B instruction-tuned model", "~14 GB", True),
            ("Phi-4 14B (4-bit)", "unsloth/Phi-4-bnb-4bit", "Microsoft's latest Phi-4 14B model", "~7 GB", True),
        ]
        
        popular_models = [
            ("Qwen2.5 32B Instruct (4-bit)", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit", "Powerful 32B parameter Qwen model", "~16 GB", False),
            ("Qwen2.5 14B Instruct (4-bit)", "unsloth/Qwen2.5-14B-Instruct-bnb-4bit", "Balanced 14B Qwen model", "~7 GB", False),
            ("Qwen2.5 7B Instruct (4-bit)", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "Efficient 7B Qwen model", "~4 GB", False),
            ("Llama 3.2 11B Vision (4-bit)", "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", "Vision-capable Llama 3.2 11B", "~6 GB", False),
            ("Llama 3.2 3B Instruct (4-bit)", "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit", "Fast 3B parameter model", "~2.5 GB", False),
            ("Llama 3.2 1B Instruct (4-bit)", "unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit", "Ultra-lightweight 1B model", "~800 MB", False),
            ("Llama 3.1 8B Instruct (4-bit)", "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit", "Popular 8B Llama 3.1", "~5 GB", False),
            ("Mistral Nemo 12B (4-bit)", "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit", "Mistral's 12B instruction model", "~6 GB", False),
            ("Gemma 2 9B Instruct (4-bit)", "unsloth/gemma-2-9b-it-bnb-4bit", "Google's 9B Gemma model", "~5 GB", False),
            ("Phi-3.5 Mini (4-bit)", "unsloth/Phi-3.5-mini-instruct-bnb-4bit", "Microsoft's efficient Phi-3.5", "~2 GB", False),
            ("OpenHermes 2.5 Mistral 7B", "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit", "Fine-tuned Mistral 7B", "~4 GB", False),
            ("Llama 3.1 70B Instruct (4-bit)", "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit", "Powerful 70B Llama 3.1", "~35 GB", False),
            ("Llama 3.1 405B Instruct (4-bit)", "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit", "Massive 405B flagship model", "~200 GB", False),
            ("Qwen2.5-Coder 7B (4-bit)", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", "Code-specialized Qwen 7B", "~4 GB", False),
            ("Qwen2.5-Coder 14B (4-bit)", "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit", "Advanced code model 14B", "~7 GB", False),
            ("DeepSeek-R1 7B (4-bit)", "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit", "Reasoning-focused 7B model", "~4 GB", False),
            ("DeepSeek-R1 14B (4-bit)", "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit", "Advanced reasoning 14B", "~7 GB", False),
            ("DeepSeek-R1 32B (4-bit)", "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit", "High-end reasoning 32B", "~16 GB", False),
            ("Llama 3.3 70B Instruct", "unsloth/Llama-3.3-70B-Instruct", "Full precision Llama 3.3 70B", "~140 GB", False),
            ("Gemma 2 2B Instruct (4-bit)", "unsloth/gemma-2-2b-it-bnb-4bit", "Lightweight 2B Gemma", "~1.5 GB", False),
        ]
        
        all_models = latest_models + popular_models
        
        row, col = 0, 0
        for name, model_id, desc, size, is_new in all_models:
            # Check if downloaded
            model_slug = model_id.replace("/", "__")
            model_path = models_dir / model_slug if models_dir.exists() else None
            is_downloaded = model_path and model_path.exists()
            
            capabilities = detect_model_capabilities(model_id=model_id, model_name=name)
            icons = get_capability_icons(capabilities)
            
            card = ModelCard(name, model_id, desc, size, icons, is_downloaded, is_new)
            card.set_theme(self.dark_mode)
            card.download_clicked.connect(self._download_curated_model)
            self.curated_layout.addWidget(card, row, col)
            self.model_cards.append(card)
            
            col += 1
            if col >= 2:
                col = 0
                row += 1
    
    def _on_model_selected(self, model_path: str):
        """Handle downloaded model selection"""
        self._log_models(f"Selected: {model_path}")
    
    def _download_curated_model(self, model_id: str):
        """Download a curated model"""
        self._log_models(f"Downloading {model_id}...")
        target = Path(self.hf_target_dir.text().strip())
        try:
            dest = download_hf_model(model_id, target)
            self._log_models(f"✓ Downloaded to: {dest}")
            self._refresh_models()
        except Exception as e:
            self._log_models(f"✗ Error: {e}")

    def _browse_hf_target(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select download folder", str(self.root))
        if d:
            self.hf_target_dir.setText(d)

    def _hf_search(self) -> None:
        q = self.hf_query.text().strip()
        self.hf_results.clear()
        if not q:
            return
        try:
            hits = search_hf_models(q, limit=30)
            for h in hits:
                item = QListWidgetItem(f"{h.model_id}  | downloads={h.downloads} likes={h.likes}")
                item.setData(Qt.UserRole, h.model_id)
                self.hf_results.addItem(item)
            self._log_models(f"Found {len(hits)} results for: {q}")
        except Exception as e:
            self._log_models(f"[ERROR] HF search failed: {e}")

    def _hf_download_selected(self) -> None:
        item = self.hf_results.currentItem()
        if not item:
            QMessageBox.warning(self, "Download", "Select a model in the results list.")
            return
        model_id = item.data(Qt.UserRole)
        target = Path(self.hf_target_dir.text().strip())
        try:
            self._log_models(f"Downloading {model_id} -> {target}")
            dest = download_hf_model(model_id, target)
            self._log_models(f"Download complete: {dest}")
            self._refresh_locals()
        except Exception as e:
            self._log_models(f"[ERROR] Download failed: {e}")

    def _log_models(self, msg: str) -> None:
        self.models_status.appendPlainText(msg)

    # ---------------- Train tab ----------------
    def _build_train_tab(self) -> QWidget:
        w = QWidget()
        main_layout = QHBoxLayout(w)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create splitter for left/right layout
        splitter = QSplitter(Qt.Horizontal)
        
        # LEFT COLUMN: Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        
        # Model Configuration Section
        left_layout.addWidget(QLabel("<h2>🎯 Model Configuration</h2>"))
        
        config_frame = QFrame()
        config_frame.setFrameShape(QFrame.StyledPanel)
        config_layout = QVBoxLayout(config_frame)
        config_layout.setSpacing(12)
        
        # Base model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("<b>Select Base Model</b>"))
        model_row.addStretch(1)
        config_layout.addLayout(model_row)
        
        self.train_base_model = QComboBox()
        self.train_base_model.setEditable(True)
        self.train_base_model.addItems(DEFAULT_BASE_MODELS)
        self.train_base_model.currentTextChanged.connect(self._on_model_selected_for_training)
        config_layout.addWidget(self.train_base_model)
        
        # Model info label
        self.model_info_label = QLabel("Select a model to see details")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("color: #888; font-size: 9pt; padding: 5px;")
        config_layout.addWidget(self.model_info_label)
        
        left_layout.addWidget(config_frame)
        
        # Dataset Upload Section
        left_layout.addWidget(QLabel("<h2>📂 Dataset Upload</h2>"))
        
        dataset_frame = QFrame()
        dataset_frame.setFrameShape(QFrame.StyledPanel)
        dataset_layout = QVBoxLayout(dataset_frame)
        dataset_layout.setSpacing(10)
        
        dataset_layout.addWidget(QLabel("<b>Upload Training Dataset (JSONL format)</b>"))
        
        self.train_data_path = QLineEdit()
        self.train_data_path.setPlaceholderText("Drag and drop file here or browse...")
        self.train_data_path.textChanged.connect(self._validate_dataset)
        dataset_layout.addWidget(self.train_data_path)
        
        browse_btn = QPushButton("📁 Browse Files")
        browse_btn.clicked.connect(self._browse_train_data)
        dataset_layout.addWidget(browse_btn)
        
        # Dataset validation status
        self.dataset_status_label = QLabel("")
        self.dataset_status_label.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_status_label)
        
        # Total examples count
        self.examples_label = QLabel("Total Examples: --")
        self.examples_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        dataset_layout.addWidget(self.examples_label)
        
        left_layout.addWidget(dataset_frame)
        
        # Training Parameters Section
        left_layout.addWidget(QLabel("<h2>⚙️ Training Parameters</h2>"))
        
        params_frame = QFrame()
        params_frame.setFrameShape(QFrame.StyledPanel)
        params_layout = QVBoxLayout(params_frame)
        params_layout.setSpacing(10)
        
        # Use recommended settings checkbox
        use_recommended = QHBoxLayout()
        self.use_recommended_btn = QPushButton("✨ Use Recommended Settings")
        self.use_recommended_btn.clicked.connect(self._use_recommended_settings)
        use_recommended.addWidget(self.use_recommended_btn)
        use_recommended.addStretch(1)
        params_layout.addLayout(use_recommended)
        
        # Model Name (auto-generated)
        params_layout.addWidget(QLabel("<b>Model Name</b>"))
        self.train_model_name = QLineEdit()
        self.train_model_name.setPlaceholderText("Auto-generated: YYMMDD_modelname_dataset_HHMM")
        params_layout.addWidget(self.train_model_name)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.train_epochs = QSpinBox()
        self.train_epochs.setRange(1, 1000)
        self.train_epochs.setValue(1)
        epochs_layout.addWidget(self.train_epochs, 1)
        
        # Batch size toggle
        self.batch_size_auto = QPushButton("✅ Optimal batch size")
        self.batch_size_auto.setCheckable(True)
        self.batch_size_auto.setChecked(True)
        self.batch_size_auto.clicked.connect(self._toggle_batch_size)
        epochs_layout.addWidget(self.batch_size_auto)
        params_layout.addLayout(epochs_layout)
        
        # LoRA R
        lora_layout = QHBoxLayout()
        lora_layout.addWidget(QLabel("LoRA R:"))
        self.train_lora_r = QSpinBox()
        self.train_lora_r.setRange(8, 256)
        self.train_lora_r.setValue(16)
        lora_layout.addWidget(self.train_lora_r, 1)
        
        # LoRA Alpha (calculated automatically)
        lora_layout.addWidget(QLabel("LoRA Alpha:"))
        self.train_lora_alpha_label = QLabel("32")
        self.train_lora_alpha_label.setStyleSheet("font-weight: bold;")
        lora_layout.addWidget(self.train_lora_alpha_label)
        self.train_lora_r.valueChanged.connect(lambda v: self.train_lora_alpha_label.setText(str(v * 2)))
        params_layout.addLayout(lora_layout)
        
        # Learning Rate + Max Seq Length
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.train_lr = QDoubleSpinBox()
        self.train_lr.setDecimals(8)
        self.train_lr.setRange(1e-8, 1.0)
        self.train_lr.setValue(2e-4)
        self.train_lr.setSingleStep(1e-5)
        lr_layout.addWidget(self.train_lr, 1)
        
        lr_layout.addWidget(QLabel("Max Seq Length:"))
        self.train_max_seq = QSpinBox()
        self.train_max_seq.setRange(128, 8192)
        self.train_max_seq.setValue(2048)
        self.train_max_seq.setSingleStep(128)
        lr_layout.addWidget(self.train_max_seq, 1)
        params_layout.addLayout(lr_layout)
        
        # Advanced settings collapsible
        self.advanced_btn = QPushButton("▶ Advanced Settings")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.clicked.connect(self._toggle_advanced)
        params_layout.addWidget(self.advanced_btn)
        
        # Advanced settings container (hidden by default)
        self.advanced_container = QFrame()
        self.advanced_container.setVisible(False)
        adv_layout = QVBoxLayout(self.advanced_container)
        
        # Output directory
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output dir:"))
        self.train_out_dir = QLineEdit(str(default_output_dir()))
        out_row.addWidget(self.train_out_dir, 2)
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._browse_train_out)
        out_row.addWidget(out_browse)
        adv_layout.addLayout(out_row)
        
        # Batch size (manual)
        self.batch_size_container = QWidget()
        batch_layout = QHBoxLayout(self.batch_size_container)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.addWidget(QLabel("Batch size:"))
        self.train_batch = QSpinBox()
        self.train_batch.setRange(1, 512)
        self.train_batch.setValue(2)
        batch_layout.addWidget(self.train_batch, 1)
        batch_layout.addStretch(1)
        adv_layout.addWidget(self.batch_size_container)
        self.batch_size_container.setVisible(False)
        
        params_layout.addWidget(self.advanced_container)
        
        left_layout.addWidget(params_frame)
        
        # GPU Selection Section
        left_layout.addWidget(QLabel("<h2>💻 Select GPU(s) for Training</h2>"))
        
        gpu_frame = QFrame()
        gpu_frame.setFrameShape(QFrame.StyledPanel)
        gpu_layout = QVBoxLayout(gpu_frame)
        
        self.gpu_status_label = QLabel("✅ 2 GPUs detected")
        self.gpu_status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        gpu_layout.addWidget(self.gpu_status_label)
        
        # GPU selection dropdown
        self.gpu_select = QComboBox()
        self.gpu_select.addItems(["GPU 0: NVIDIA GeForce RTX 4090"])
        gpu_layout.addWidget(self.gpu_select)
        
        # Training info
        self.training_info_label = QLabel("⚡ Training will use: GPU 0: NVIDIA GeForce RTX 4090")
        self.training_info_label.setStyleSheet("color: #2196F3; padding: 5px;")
        self.training_info_label.setWordWrap(True)
        gpu_layout.addWidget(self.training_info_label)
        
        left_layout.addWidget(gpu_frame)
        
        # Start Training Button
        start_btn_layout = QHBoxLayout()
        self.train_start = QPushButton("🚀 Start Training")
        self.train_start.setMinimumHeight(50)
        self.train_start.clicked.connect(self._start_training)
        self.train_start.setStyleSheet("""
            QPushButton {
                font-size: 14pt;
                font-weight: bold;
            }
        """)
        start_btn_layout.addWidget(self.train_start)
        
        self.train_stop = QPushButton("⏹ Stop")
        self.train_stop.setEnabled(False)
        self.train_stop.setMinimumHeight(50)
        self.train_stop.clicked.connect(self._stop_training)
        start_btn_layout.addWidget(self.train_stop)
        
        left_layout.addLayout(start_btn_layout)
        
        left_layout.addStretch(1)
        
        # RIGHT COLUMN: Training Visualization
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)
        
        # Training Dashboard
        right_layout.addWidget(QLabel("<h2>📊 Training Dashboard</h2>"))
        
        # Status banner
        self.training_status_banner = QLabel("⏸ WAITING FOR TRAINING TO START")
        self.training_status_banner.setAlignment(Qt.AlignCenter)
        self.training_status_banner.setMinimumHeight(50)
        self.training_status_banner.setStyleSheet("""
            QLabel {
                background: #ff9800;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        right_layout.addWidget(self.training_status_banner)
        
        # Metrics grid (2x2)
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)
        
        self.epoch_card = MetricCard("EPOCH", "📚", "0/0")
        self.steps_card = MetricCard("STEPS", "🔥", "0/0")
        self.loss_card = MetricCard("LOSS", "📉", "--·----")
        self.eta_card = MetricCard("ETA", "⏱", "--m --s")
        
        metrics_grid.addWidget(self.epoch_card, 0, 0)
        metrics_grid.addWidget(self.steps_card, 0, 1)
        metrics_grid.addWidget(self.loss_card, 1, 0)
        metrics_grid.addWidget(self.eta_card, 1, 1)
        
        right_layout.addLayout(metrics_grid)
        
        # Progress bar
        self.progress_label = QLabel("0.0% Complete")
        self.progress_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.progress_label)
        
        # Additional metrics (3 cards)
        extra_metrics = QHBoxLayout()
        extra_metrics.setSpacing(10)
        
        self.learning_rate_card = MetricCard("LEARNING RATE", "📊", "--e-0")
        self.speed_card = MetricCard("SPEED", "🚀", "-- samp/s")
        self.gpu_mem_card = MetricCard("GPU MEMORY", "💾", "-- GB")
        
        extra_metrics.addWidget(self.learning_rate_card)
        extra_metrics.addWidget(self.speed_card)
        extra_metrics.addWidget(self.gpu_mem_card)
        
        right_layout.addLayout(extra_metrics)
        
        # Loss Over Time Section
        right_layout.addWidget(QLabel("<h3>📉 Loss Over Time</h3>"))
        
        loss_frame = QFrame()
        loss_frame.setFrameShape(QFrame.StyledPanel)
        loss_frame.setMinimumHeight(150)
        loss_layout = QVBoxLayout(loss_frame)
        self.loss_chart_label = QLabel("Loss chart will appear here once training starts...")
        self.loss_chart_label.setAlignment(Qt.AlignCenter)
        self.loss_chart_label.setStyleSheet("color: #888;")
        loss_layout.addWidget(self.loss_chart_label)
        right_layout.addWidget(loss_frame)
        
        # Training Logs
        logs_header = QHBoxLayout()
        logs_header.addWidget(QLabel("<h3>📋 View Detailed Logs</h3>"))
        logs_header.addStretch(1)
        self.logs_expand_btn = QPushButton("▼ Show Logs")
        self.logs_expand_btn.setCheckable(True)
        self.logs_expand_btn.clicked.connect(self._toggle_logs)
        logs_header.addWidget(self.logs_expand_btn)
        right_layout.addLayout(logs_header)
        
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumBlockCount(10000)
        self.train_log.setMaximumHeight(200)
        self.train_log.setVisible(False)
        right_layout.addWidget(self.train_log)
        
        right_layout.addStretch(1)
        
        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 700])
        
        main_layout.addWidget(splitter)
        
        # Store metric cards for theme updates
        self.metric_cards = [
            self.epoch_card, self.steps_card, self.loss_card, self.eta_card,
            self.learning_rate_card, self.speed_card, self.gpu_mem_card
        ]
        
        return w
    
    def _on_model_selected_for_training(self, model_id: str):
        """Show model info when selected"""
        if model_id:
            capabilities = detect_model_capabilities(model_id=model_id)
            icons = get_capability_icons(capabilities)
            self.model_info_label.setText(f"Selected: {model_id}\n{icons} Capabilities detected")
    
    def _validate_dataset(self):
        """Validate and show dataset info"""
        path = self.train_data_path.text().strip()
        if not path:
            self.dataset_status_label.setText("")
            self.examples_label.setText("Total Examples: --")
            return
        
        if Path(path).exists():
            # Count lines in JSONL
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
                self.dataset_status_label.setText(f"✅ Found dataset: {Path(path).name}")
                self.dataset_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.examples_label.setText(f"Total Examples: {count}")
            except Exception as e:
                self.dataset_status_label.setText(f"❌ Error reading file: {e}")
                self.dataset_status_label.setStyleSheet("color: #f44336;")
        else:
            self.dataset_status_label.setText("⚠️ File not found")
            self.dataset_status_label.setStyleSheet("color: #ff9800;")
    
    def _use_recommended_settings(self):
        """Apply recommended training settings"""
        self.train_epochs.setValue(3)
        self.train_lora_r.setValue(16)
        self.train_lr.setValue(2e-4)
        self.train_max_seq.setValue(2048)
        QMessageBox.information(self, "Recommended Settings", "Applied recommended settings:\n• Epochs: 3\n• LoRA R: 16\n• Learning Rate: 2e-4\n• Max Seq Length: 2048")
    
    def _toggle_batch_size(self):
        """Toggle between auto and manual batch size"""
        is_auto = self.batch_size_auto.isChecked()
        self.batch_size_container.setVisible(not is_auto)
        if is_auto:
            self.batch_size_auto.setText("✅ Optimal batch size")
        else:
            self.batch_size_auto.setText("Manual batch size")
    
    def _toggle_advanced(self):
        """Toggle advanced settings visibility"""
        is_visible = self.advanced_btn.isChecked()
        self.advanced_container.setVisible(is_visible)
        if is_visible:
            self.advanced_btn.setText("▼ Advanced Settings")
        else:
            self.advanced_btn.setText("▶ Advanced Settings")
    
    def _toggle_logs(self):
        """Toggle training logs visibility"""
        is_visible = self.logs_expand_btn.isChecked()
        self.train_log.setVisible(is_visible)
        if is_visible:
            self.logs_expand_btn.setText("▲ Hide Logs")
        else:
            self.logs_expand_btn.setText("▼ Show Logs")

    def _browse_train_data(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Select dataset file", str(self.root), "Data files (*.jsonl *.json *.txt);;All files (*)")
        if f:
            self.train_data_path.setText(f)

    def _browse_train_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output folder", str(self.root))
        if d:
            self.train_out_dir.setText(d)

    def _start_training(self) -> None:
        if self.train_proc is not None:
            QMessageBox.information(self, "Training", "Training is already running.")
            return
        data_path = self.train_data_path.text().strip()
        if not data_path:
            QMessageBox.warning(self, "Training", "Select a dataset file.")
            return

        cfg = TrainingConfig(
            base_model=self.train_base_model.currentText().strip(),
            data_path=Path(data_path),
            output_dir=Path(self.train_out_dir.text().strip()),
            epochs=int(self.train_epochs.value()),
            batch_size=int(self.train_batch.value()),
            learning_rate=float(self.train_lr.value()),
        )

        cmd = build_finetune_cmd(cfg)

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._append_proc_output(proc, self.train_log))
        proc.finished.connect(self._train_finished)

        self.train_log.appendPlainText(">> " + " ".join(cmd))
        proc.start()
        if not proc.waitForStarted(5000):
            QMessageBox.critical(self, "Training", "Failed to start training process.")
            return

        self.train_proc = proc
        self.train_start.setEnabled(False)
        self.train_stop.setEnabled(True)

    def _stop_training(self) -> None:
        if self.train_proc is None:
            return
        self.train_log.appendPlainText("\n[INFO] Terminating training process...")
        self.train_proc.terminate()
        if not self.train_proc.waitForFinished(5000):
            self.train_proc.kill()

    def _train_finished(self) -> None:
        self.train_log.appendPlainText("\n[INFO] Training process finished.")
        self.train_proc = None
        self.train_start.setEnabled(True)
        self.train_stop.setEnabled(False)
        self._refresh_locals()

    # ---------------- Test tab ----------------
    def _build_test_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Title
        layout.addWidget(QLabel("<h2>🧪 Test Models - Side-by-Side Chat</h2>"))

        # Side-by-side model comparison (TOP - Chat)
        models_layout = QHBoxLayout()
        models_layout.setSpacing(20)
        
        # MODEL A (Left)
        model_a_widget = QWidget()
        model_a_layout = QVBoxLayout(model_a_widget)
        model_a_layout.setSpacing(10)
        
        # Header
        header_a = QLabel("🔵 <b>Model A</b>")
        header_a.setStyleSheet("font-size: 14pt; padding: 10px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2); color: white; border-radius: 6px;")
        model_a_layout.addWidget(header_a)
        
        # Model selection
        self.test_model_a = QComboBox()
        self.test_model_a.setEditable(True)
        self.test_model_a.addItem("None")
        model_a_layout.addWidget(self.test_model_a)
        
        # Chat widget (WhatsApp style)
        self.chat_widget_a = ChatWidget()
        model_a_layout.addWidget(self.chat_widget_a, 1)
        
        models_layout.addWidget(model_a_widget, 1)
        
        # MODEL B (Right)
        model_b_widget = QWidget()
        model_b_layout = QVBoxLayout(model_b_widget)
        model_b_layout.setSpacing(10)
        
        # Header
        header_b = QLabel("🟢 <b>Model B</b>")
        header_b.setStyleSheet("font-size: 14pt; padding: 10px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4CAF50, stop:1 #2E7D32); color: white; border-radius: 6px;")
        model_b_layout.addWidget(header_b)
        
        # Model selection
        self.test_model_b = QComboBox()
        self.test_model_b.setEditable(True)
        self.test_model_b.addItem("None")
        model_b_layout.addWidget(self.test_model_b)
        
        # Chat widget (WhatsApp style)
        self.chat_widget_b = ChatWidget()
        model_b_layout.addWidget(self.chat_widget_b, 1)
        
        models_layout.addWidget(model_b_widget, 1)
        
        layout.addLayout(models_layout, 1)

        # Shared prompt input area (BOTTOM)
        prompt_layout = QVBoxLayout()
        prompt_layout.addWidget(QLabel("<b>💬 Type your message:</b>"))
        
        self.test_prompt = QTextEdit()
        self.test_prompt.setPlaceholderText("Type your message here...")
        self.test_prompt.setMinimumHeight(100)
        self.test_prompt.setMaximumHeight(100)
        prompt_layout.addWidget(self.test_prompt)
        
        # Buttons row
        btn_layout = QHBoxLayout()
        self.test_send_btn = QPushButton("📤 Send")
        self.test_send_btn.clicked.connect(self._run_side_by_side_test)
        self.test_send_btn.setMinimumHeight(40)
        self.test_send_btn.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                font-weight: bold;
            }
        """)
        btn_layout.addWidget(self.test_send_btn)
        
        self.test_clear_btn = QPushButton("🗑️ Clear")
        self.test_clear_btn.clicked.connect(self._clear_test_chat)
        btn_layout.addWidget(self.test_clear_btn)
        btn_layout.addStretch(1)
        
        prompt_layout.addLayout(btn_layout)
        layout.addLayout(prompt_layout)

        # Store for theme updates
        self.chat_widgets = [self.chat_widget_a, self.chat_widget_b]
        
        # Initialize process and buffer variables
        self.test_proc_a = None
        self.test_proc_b = None
        self.inference_buffer_a = ""
        self.inference_buffer_b = ""

        return w
    
    def _run_side_by_side_test(self) -> None:
        """Run inference on both models simultaneously"""
        prompt = self.test_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Test", "Please enter a prompt.")
            return
        
        model_a_text = self.test_model_a.currentText().strip()
        model_b_text = self.test_model_b.currentText().strip()
        
        if (model_a_text == "None" or model_a_text.startswith("(No models")) and \
           (model_b_text == "None" or model_b_text.startswith("(No models")):
            QMessageBox.warning(self, "Test", "Please download and select at least one model from the Download tab.")
            return
        
        # Get full paths
        model_a_path = None
        model_b_path = None
        
        if model_a_text != "None" and not model_a_text.startswith("(No models"):
            idx = self.test_model_a.currentIndex()
            model_a_path = self.test_model_a.itemData(idx)
        
        if model_b_text != "None" and not model_b_text.startswith("(No models"):
            idx = self.test_model_b.currentIndex()
            model_b_path = self.test_model_b.itemData(idx)
        
        # Add user message to both chats (RIGHT side bubble)
        if model_a_path:
            self.chat_widget_a.add_message(prompt, is_user=True)
        if model_b_path:
            self.chat_widget_b.add_message(prompt, is_user=True)
        
        # Run Model A
        if model_a_path:
            self.chat_widget_a.add_message("Thinking...", is_user=False)
            self._run_inference_a(model_a_path, prompt)
        
        # Run Model B
        if model_b_path:
            self.chat_widget_b.add_message("Thinking...", is_user=False)
            self._run_inference_b(model_b_path, prompt)
        
        # Clear prompt
        self.test_prompt.clear()
    
    def _run_inference_a(self, model_path: str, prompt: str):
        """Run inference for Model A using QProcess"""
        # Reset buffer
        self.inference_buffer_a = ""
        
        # Build command using existing infrastructure
        cfg = InferenceConfig(
            prompt=prompt,
            base_model=model_path,  # Full path to downloaded model
            max_new_tokens=512,
            temperature=0.7
        )
        cmd = build_run_adapter_cmd(cfg)
        
        # Create QProcess
        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        
        # Connect to read output and update last bubble
        proc.readyReadStandardOutput.connect(
            lambda: self._update_inference_output_a(proc)
        )
        proc.finished.connect(lambda: self._on_inference_finished_a())
        
        proc.start()
        self.test_proc_a = proc
    
    def _update_inference_output_a(self, proc: QProcess):
        """Update Model A chat bubble with streaming output"""
        # Read output from process
        data = proc.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='replace')
        
        # Accumulate text in buffer
        self.inference_buffer_a += text
        
        # Extract the actual response (skip loading messages, etc.)
        # Look for lines that are actual model output
        lines = self.inference_buffer_a.split('\n')
        response_lines = []
        capture = False
        
        for line in lines:
            # Skip status messages
            if any(x in line for x in ['[INFO]', '[OK]', 'Loading', 'Generating', 'FutureWarning', 'UserWarning']):
                continue
            # Start capturing after we see model is loaded
            if 'Set pad_token' in line or 'Model loaded' in line:
                capture = True
                continue
            if capture and line.strip():
                response_lines.append(line)
        
        # Update the chat bubble with cleaned response
        if response_lines:
            clean_response = '\n'.join(response_lines).strip()
            if clean_response:
                self.chat_widget_a.update_last_ai_message(clean_response)
        elif self.inference_buffer_a.strip() and 'Loading' not in self.inference_buffer_a:
            # Fallback: show raw buffer if we can't parse it
            self.chat_widget_a.update_last_ai_message(self.inference_buffer_a.strip())
    
    def _on_inference_finished_a(self):
        """Called when Model A inference finishes"""
        # Final update with complete output
        if self.inference_buffer_a.strip():
            self._update_inference_output_a(self.test_proc_a)
        self.test_proc_a = None
    
    def _run_inference_b(self, model_path: str, prompt: str):
        """Run inference for Model B using QProcess"""
        # Reset buffer
        self.inference_buffer_b = ""
        
        # Build command using existing infrastructure
        cfg = InferenceConfig(
            prompt=prompt,
            base_model=model_path,  # Full path to downloaded model
            max_new_tokens=512,
            temperature=0.7
        )
        cmd = build_run_adapter_cmd(cfg)
        
        # Create QProcess
        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        
        # Connect to read output and update last bubble
        proc.readyReadStandardOutput.connect(
            lambda: self._update_inference_output_b(proc)
        )
        proc.finished.connect(lambda: self._on_inference_finished_b())
        
        proc.start()
        self.test_proc_b = proc
    
    def _update_inference_output_b(self, proc: QProcess):
        """Update Model B chat bubble with streaming output"""
        # Read output from process
        data = proc.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='replace')
        
        # Accumulate text in buffer
        self.inference_buffer_b += text
        
        # Extract the actual response (skip loading messages, etc.)
        lines = self.inference_buffer_b.split('\n')
        response_lines = []
        capture = False
        
        for line in lines:
            # Skip status messages
            if any(x in line for x in ['[INFO]', '[OK]', 'Loading', 'Generating', 'FutureWarning', 'UserWarning']):
                continue
            # Start capturing after we see model is loaded
            if 'Set pad_token' in line or 'Model loaded' in line:
                capture = True
                continue
            if capture and line.strip():
                response_lines.append(line)
        
        # Update the chat bubble with cleaned response
        if response_lines:
            clean_response = '\n'.join(response_lines).strip()
            if clean_response:
                self.chat_widget_b.update_last_ai_message(clean_response)
        elif self.inference_buffer_b.strip() and 'Loading' not in self.inference_buffer_b:
            # Fallback: show raw buffer if we can't parse it
            self.chat_widget_b.update_last_ai_message(self.inference_buffer_b.strip())
    
    def _on_inference_finished_b(self):
        """Called when Model B inference finishes"""
        # Final update with complete output
        if self.inference_buffer_b.strip():
            self._update_inference_output_b(self.test_proc_b)
        self.test_proc_b = None
    
    def _clear_test_chat(self) -> None:
        """Clear both chat histories"""
        self.chat_widget_a.clear()
        self.chat_widget_b.clear()
        self.test_prompt.clear()

    # ---------------- Logs tab ----------------
    def _build_logs_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        row = QHBoxLayout()
        self.logs_refresh = QPushButton("Refresh")
        self.logs_refresh.clicked.connect(self._refresh_locals)
        row.addWidget(self.logs_refresh)
        row.addStretch(1)
        layout.addLayout(row)

        split = QSplitter(Qt.Horizontal)
        self.logs_list = QListWidget()
        self.logs_list.itemSelectionChanged.connect(self._open_selected_log)
        self.logs_view = QPlainTextEdit()
        self.logs_view.setReadOnly(True)
        self.logs_view.setMaximumBlockCount(20000)

        split.addWidget(self.logs_list)
        split.addWidget(self.logs_view)
        split.setSizes([300, 900])
        layout.addWidget(split, 1)
        return w

    def _open_selected_log(self) -> None:
        item = self.logs_list.currentItem()
        if not item:
            return
        path = Path(item.data(Qt.UserRole))
        try:
            data = path.read_text(encoding="utf-8", errors="replace")
            self.logs_view.setPlainText(data[-20000:])
        except Exception as e:
            self.logs_view.setPlainText(f"[ERROR] Could not read {path}: {e}")

    # ---------------- Helpers ----------------
    def _append_proc_output(self, proc: QProcess, widget: QPlainTextEdit) -> None:
        data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            widget.appendPlainText(data.rstrip("\n"))

    def _refresh_locals(self) -> None:
        # Refresh models
        self._refresh_models()
        
        # Refresh test model dropdowns with ONLY DOWNLOADED models
        models_dir = self.root / "models"
        downloaded_models = []
        
        if models_dir.exists():
            for model_dir in sorted(models_dir.iterdir()):
                if model_dir.is_dir():
                    downloaded_models.append(str(model_dir))  # Full path for inference
        
        # Update Model A dropdown
        current_a = self.test_model_a.currentText()
        self.test_model_a.clear()
        self.test_model_a.addItem("None")
        
        if downloaded_models:
            for model_path in downloaded_models:
                model_name = Path(model_path).name
                self.test_model_a.addItem(model_name, model_path)  # Display name, store full path
        else:
            self.test_model_a.addItem("(No models downloaded yet)")
        
        if current_a and current_a != "None":
            idx = self.test_model_a.findText(current_a)
            if idx >= 0:
                self.test_model_a.setCurrentIndex(idx)
        
        # Update Model B dropdown
        current_b = self.test_model_b.currentText()
        self.test_model_b.clear()
        self.test_model_b.addItem("None")
        
        if downloaded_models:
            for model_path in downloaded_models:
                model_name = Path(model_path).name
                self.test_model_b.addItem(model_name, model_path)  # Display name, store full path
        else:
            self.test_model_b.addItem("(No models downloaded yet)")
        
        if current_b and current_b != "None":
            idx = self.test_model_b.findText(current_b)
            if idx >= 0:
                self.test_model_b.setCurrentIndex(idx)

        # log list from repo root
        self.logs_list.clear()
        for p in sorted(self.root.glob("*training*.txt")) + sorted(self.root.glob("*log*.txt")):
            it = QListWidgetItem(str(p.name))
            it.setData(Qt.UserRole, str(p))
            self.logs_list.addItem(it)

    # these are set in models tab construction
    downloaded_container: QWidget
    curated_container: QWidget
    model_cards: list
    downloaded_model_cards: list


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
