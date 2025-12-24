from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer, QThread, Signal, QProcessEnvironment, QRect, QSize
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QTextEdit, QPlainTextEdit,
    QSpinBox, QDoubleSpinBox, QMessageBox, QListWidget, QListWidgetItem, QSplitter, QToolBar, QScrollArea, QGridLayout, QFrame, QProgressBar, QSizePolicy, QTabBar, QStyleOptionTab, QStyle
)
from PySide6.QtGui import QAction, QIcon, QFont

from desktop_app.model_card_widget import ModelCard, DownloadedModelCard
from desktop_app.training_widgets import MetricCard
from desktop_app.chat_widget import ChatWidget

from system_detector import SystemDetector
from smart_installer import SmartInstaller
from setup_state import SetupStateManager
from core.models import (DEFAULT_BASE_MODELS, search_hf_models, download_hf_model, list_local_adapters, 
                         list_local_downloads, get_app_root, detect_model_capabilities, get_capability_icons, get_model_size)
from core.training import TrainingConfig, default_output_dir, build_finetune_cmd
from core.inference import InferenceConfig, build_run_adapter_cmd


APP_TITLE = "🤖 LLM Fine-tuning Studio"


class InstallerThread(QThread):
    """Thread for running smart installer without freezing UI"""
    log_output = Signal(str)
    finished_signal = Signal(bool)  # True if successful
    
    def __init__(self, install_type: str):  # "pytorch", "dependencies", or "all"
        super().__init__()
        self.install_type = install_type
    
    def run(self):
        try:
            installer = SmartInstaller()
            
            # Redirect installer logs to GUI
            original_log = installer.log
            def gui_log(message):
                self.log_output.emit(message)
                original_log(message)
            installer.log = gui_log
            
            # Run detection
            installer.run_detection()
            
            # Install based on type
            if self.install_type == "pytorch":
                success = installer.install_pytorch()
            elif self.install_type == "dependencies":
                success = installer.install_dependencies()
            else:  # "all"
                success = installer.install()
            
            self.finished_signal.emit(success)
        except Exception as e:
            self.log_output.emit(f"[ERROR] Installation failed: {e}")
            self.finished_signal.emit(False)


class DownloadThread(QThread):
    """Thread for downloading models without freezing UI"""
    progress = Signal(int)  # 0-100
    finished = Signal(str)  # destination path
    error = Signal(str)     # error message
    
    def __init__(self, model_id: str, target_dir: Path):
        super().__init__()
        self.model_id = model_id
        self.target_dir = target_dir
    
    def run(self):
        try:
            # Import here to avoid import errors if not installed
            from huggingface_hub import snapshot_download
            
            # Convert model ID to directory name (e.g., unsloth/model -> unsloth__model)
            model_slug = self.model_id.replace("/", "__")
            dest = self.target_dir / model_slug
            
            # Emit initial progress
            self.progress.emit(5)
            
            # Download
            result = snapshot_download(
                repo_id=self.model_id,
                local_dir=str(dest),
                local_dir_use_symlinks=False
            )
            
            self.progress.emit(100)
            self.finished.emit(str(dest))  # Return the actual destination path
            
        except Exception as e:
            self.error.emit(str(e))


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
QFrame {
    border: none;
    background-color: #1a1d23;
}
QFrame[frameShape="4"] {
    border: none;
    border-radius: 8px;
    background-color: #1a1d23;
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
        self.setMinimumSize(800, 600)  # Allow resizing with reasonable minimum

        self.root = get_app_root()
        self.dark_mode = True  # Start in dark mode
        
        # Detect real system info
        self.system_info = SystemDetector().detect_all()

        # Create a beautiful unified header
        header_widget = QWidget()
        header_widget.setMinimumHeight(80)
        header_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #f093fb);
                border: none;
            }
        """)
        
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 10, 20, 10)
        header_layout.setSpacing(20)
        
        # Left: Theme toggle button
        theme_btn = QPushButton("🌙 Dark Mode")
        theme_btn.setMinimumHeight(50)
        theme_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 25px;
                padding: 10px 20px;
                font-size: 14pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.5);
            }
        """)
        theme_btn.clicked.connect(self._toggle_theme)
        self.theme_btn = theme_btn
        header_layout.addWidget(theme_btn)
        
        # Center: App title (transparent background)
        title_label = QLabel(APP_TITLE)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            background: transparent;
            color: white; 
            font-size: 24pt; 
            font-weight: bold;
            border: none;
        """)
        header_layout.addWidget(title_label, 1)
        
        # Right: System info (compact)
        sys_info_widget = QWidget()
        sys_info_widget.setStyleSheet("background: transparent; border: none;")
        sys_info_layout = QVBoxLayout(sys_info_widget)
        sys_info_layout.setContentsMargins(0, 0, 0, 0)
        sys_info_layout.setSpacing(4)
        
        # Get real system info
        python_info = self.system_info.get("python", {})
        pytorch_info = self.system_info.get("pytorch", {})
        hardware_info = self.system_info.get("hardware", {})
        
        python_ver = python_info.get("version", "Unknown")
        python_label = QLabel(f"🐍 Python {python_ver}")
        python_label.setStyleSheet("color: white; font-size: 11pt; font-weight: bold; background: transparent;")
        sys_info_layout.addWidget(python_label)
        
        # PyTorch info
        if pytorch_info.get("found"):
            pytorch_ver = pytorch_info.get("version", "Unknown")
            if pytorch_info.get("cuda_available"):
                cuda_ver = pytorch_info.get("cuda_version", "Unknown")
                pytorch_label = QLabel(f"🔥 PyTorch {pytorch_ver} (CUDA {cuda_ver})")
            else:
                pytorch_label = QLabel(f"🔥 PyTorch {pytorch_ver} (CPU)")
        else:
            pytorch_label = QLabel("🔥 PyTorch: Not found")
        pytorch_label.setStyleSheet("color: white; font-size: 11pt; font-weight: bold; background: transparent;")
        sys_info_layout.addWidget(pytorch_label)
        
        # RAM info (round to nearest power of 2 for cleaner display)
        ram_gb = hardware_info.get("ram_gb", 0)
        # Round to nearest common RAM size (e.g., 63.8 -> 64)
        if ram_gb > 60:
            ram_display = 64
        elif ram_gb > 30:
            ram_display = 32
        elif ram_gb > 14:
            ram_display = 16
        elif ram_gb > 6:
            ram_display = 8
        else:
            ram_display = round(ram_gb)
        
        ram_label = QLabel(f"💾 RAM: {ram_display} GB")
        ram_label.setStyleSheet("color: white; font-size: 11pt; font-weight: bold; background: transparent;")
        sys_info_layout.addWidget(ram_label)
        
        header_layout.addWidget(sys_info_widget)
        
        # Create main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        main_layout.addWidget(header_widget)

        # Create custom navbar with buttons
        navbar = QWidget()
        navbar_layout = QHBoxLayout(navbar)
        navbar_layout.setContentsMargins(0, 0, 0, 0)
        navbar_layout.setSpacing(2)
        
        # Tab buttons
        self.home_btn = QPushButton("🏠 Home")
        self.train_btn = QPushButton("🎯 Train")
        self.download_btn = QPushButton("📥 Download")
        self.test_btn = QPushButton("🧪 Test")
        self.logs_btn = QPushButton("📊 Logs")
        self.info_btn = QPushButton("ℹ️ Info")
        
        # Style buttons to look like tabs
        tab_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14pt;
                font-weight: bold;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #764ba2, stop:1 #667eea);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c8ef5, stop:1 #8a5cb8);
            }
        """
        
        for btn in [self.home_btn, self.train_btn, self.download_btn, self.test_btn, self.logs_btn, self.info_btn]:
            btn.setCheckable(True)
            btn.setStyleSheet(tab_button_style)
        
        # Add left-side buttons
        navbar_layout.addWidget(self.home_btn)
        navbar_layout.addWidget(self.train_btn)
        navbar_layout.addWidget(self.download_btn)
        navbar_layout.addWidget(self.test_btn)
        navbar_layout.addWidget(self.logs_btn)
        
        # Add stretch to consume remaining space
        navbar_layout.addStretch(1)
        
        # Add Info button on far right
        navbar_layout.addWidget(self.info_btn)
        
        main_layout.addWidget(navbar)

        # Create tab widget for content (hide the tab bar since we're using custom buttons)
        tabs = QTabWidget()
        tabs.tabBar().setVisible(False)
        
        tabs.addTab(self._build_home_tab(), "Home")
        tabs.addTab(self._build_train_tab(), "Train")
        tabs.addTab(self._build_models_tab(), "Download")
        tabs.addTab(self._build_test_tab(), "Test")
        tabs.addTab(self._build_logs_tab(), "Logs")
        tabs.addTab(self._build_info_tab(), "Info")
        
        # Connect buttons to tab switching
        self.home_btn.clicked.connect(lambda: self._switch_tab(tabs, 0))
        self.train_btn.clicked.connect(lambda: self._switch_tab(tabs, 1))
        self.download_btn.clicked.connect(lambda: self._switch_tab(tabs, 2))
        self.test_btn.clicked.connect(lambda: self._switch_tab(tabs, 3))
        self.logs_btn.clicked.connect(lambda: self._switch_tab(tabs, 4))
        self.info_btn.clicked.connect(lambda: self._switch_tab(tabs, 5))
        
        # Set Home as default
        self.home_btn.setChecked(True)
        tabs.setCurrentIndex(0)

        main_layout.addWidget(tabs)
        self.setCentralWidget(main_widget)

        self.train_proc: QProcess | None = None
        
        # Initialize card lists
        self.model_cards = []
        
        # Auto-run system diagnostics on startup (delayed to allow UI to render first)
        QTimer.singleShot(500, self._auto_check_system)
        self.downloaded_model_cards = []
        self.metric_cards = []
        
        self._refresh_locals()
        self._apply_theme()

    def _create_status_row(self, label: str, is_ok: bool, detail: str) -> QHBoxLayout:
        """Create a status indicator row"""
        row = QHBoxLayout()
        
        status_icon = "✅" if is_ok else "❌"
        color = "#4CAF50" if is_ok else "#f44336"
        
        main_label = QLabel(f"{status_icon} <b>{label}</b>")
        main_label.setStyleSheet(f"color: {color};")
        row.addWidget(main_label)
        
        row.addStretch(1)
        
        detail_label = QLabel(detail)
        detail_label.setStyleSheet("color: #888; font-size: 10pt;")
        row.addWidget(detail_label)
        
        return row
    
    def _create_status_widget(self, label: str, is_ok: bool, detail: str) -> QWidget:
        """Create a status indicator widget"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        status_icon = "✅" if is_ok else "❌"
        color = "#4CAF50" if is_ok else "#f44336"
        
        main_label = QLabel(f"{status_icon} <b>{label}</b>")
        main_label.setStyleSheet(f"color: {color};")
        layout.addWidget(main_label)
        
        layout.addStretch(1)
        
        detail_label = QLabel(detail)
        detail_label.setStyleSheet("color: #888; font-size: 10pt;")
        layout.addWidget(detail_label)
        
        return widget
    
    def _auto_check_system(self):
        """Auto-run system diagnostics on startup"""
        # Re-detect system info
        self.system_info = SystemDetector().detect_all()
        
        # Log the diagnostics
        print("=== System Diagnostics ===")
        print(f"Python: {self.system_info.get('python', {}).get('version', 'Not found')}")
        pytorch_info = self.system_info.get('pytorch', {})
        if pytorch_info.get('found'):
            print(f"PyTorch: {pytorch_info.get('version', 'N/A')} (CUDA: {pytorch_info.get('cuda_available', False)})")
        else:
            print("PyTorch: Not installed")
        
        cuda_info = self.system_info.get('cuda', {})
        if cuda_info.get('found'):
            print(f"CUDA: {cuda_info.get('cuda_version', 'N/A')} (Driver: {cuda_info.get('driver_version', 'N/A')})")
            gpus = cuda_info.get('gpus', [])
            for idx, gpu in enumerate(gpus):
                print(f"  GPU {idx}: {gpu.get('name', 'Unknown')} ({gpu.get('memory', 'Unknown')})")
        else:
            print("CUDA: Not found")
        
        print("=========================\n")
    
    def _switch_tab(self, tab_widget: QTabWidget, index: int):
        """Switch to a tab and update button states"""
        tab_widget.setCurrentIndex(index)
        
        # Update button checked states
        buttons = [self.home_btn, self.train_btn, self.download_btn, self.test_btn, self.logs_btn, self.info_btn]
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)
    
    def _install_pytorch(self):
        """Install PyTorch with CUDA"""
        reply = QMessageBox.question(
            self,
            "Install PyTorch with CUDA",
            "This will install PyTorch with CUDA support for GPU acceleration.\n\n"
            "Download size: ~2.5 GB\n"
            "Time: 5-10 minutes\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Show log area
        self.install_log.setVisible(True)
        self.install_log.clear()
        self.install_log.appendPlainText("=== Installing PyTorch with CUDA ===\n")
        
        # Disable button
        if hasattr(self, 'install_pytorch_btn'):
            self.install_pytorch_btn.setEnabled(False)
            self.install_pytorch_btn.setText("Installing...")
        
        # Start installer thread
        self.installer_thread = InstallerThread("pytorch")
        self.installer_thread.log_output.connect(lambda msg: self.install_log.appendPlainText(msg))
        self.installer_thread.finished_signal.connect(self._on_install_complete)
        self.installer_thread.start()
    
    def _install_dependencies(self):
        """Install application dependencies"""
        reply = QMessageBox.question(
            self,
            "Install Dependencies",
            "This will install all required dependencies including unsloth.\n\n"
            "Time: 5-10 minutes\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Show log area
        self.install_log.setVisible(True)
        self.install_log.clear()
        self.install_log.appendPlainText("=== Installing Dependencies ===\n")
        
        # Start installer thread
        self.installer_thread = InstallerThread("dependencies")
        self.installer_thread.log_output.connect(lambda msg: self.install_log.appendPlainText(msg))
        self.installer_thread.finished_signal.connect(self._on_install_complete)
        self.installer_thread.start()
    
    def _on_install_complete(self, success: bool):
        """Handle installation completion"""
        if success:
            self.install_log.appendPlainText("\n✅ Installation completed successfully!")
            QMessageBox.information(
                self,
                "Installation Complete",
                "✅ Installation completed successfully!\n\n"
                "Please restart the application for changes to take effect."
            )
        else:
            self.install_log.appendPlainText("\n❌ Installation failed!")
            QMessageBox.critical(
                self,
                "Installation Failed",
                "❌ Installation failed.\n\n"
                "Please check the logs above for details."
            )
        
        # Re-enable button
        if hasattr(self, 'install_pytorch_btn'):
            self.install_pytorch_btn.setEnabled(True)
            self.install_pytorch_btn.setText("Install CUDA Version")
    
    def _toggle_theme(self) -> None:
        """Toggle between dark and light themes"""
        self.dark_mode = not self.dark_mode
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the current theme"""
        if self.dark_mode:
            self.setStyleSheet(DARK_THEME)
            self.theme_btn.setText("🌙 Dark Mode")
        else:
            self.setStyleSheet(LIGHT_THEME)
            self.theme_btn.setText("☀️ Light Mode")
        
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
        
        # Create 2-column layout with SPLITTER for true equal sizing
        splitter = QSplitter(Qt.Horizontal)
        
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
        
        # Add left to splitter
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)
        
        # RIGHT: System Status (scrollable to prevent clipping when fonts are larger)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)  # Tighter spacing
        right_layout.setContentsMargins(5, 5, 5, 5)  # Tighter margins

        right_layout.addWidget(QLabel("<h2>📊 System Status</h2>"))

        # System info cards
        sys_frame = QFrame()
        sys_frame.setFrameShape(QFrame.StyledPanel)
        sys_layout = QVBoxLayout(sys_frame)
        sys_layout.setSpacing(6)  # Tighter spacing
        sys_layout.setContentsMargins(10, 8, 10, 8)  # Tighter margins

        refresh_btn = QPushButton("🔄 Refresh GPU Detection")
        refresh_btn.setMaximumWidth(200)  # Compact button
        sys_layout.addWidget(refresh_btn)

        # Get real GPU info
        cuda_info = self.system_info.get("cuda", {})
        gpus = cuda_info.get("gpus", [])
        
        if gpus:
            gpu_status = QLabel(f"✅ <b>{len(gpus)} GPU{'s' if len(gpus) > 1 else ''} detected</b>")
            gpu_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            sys_layout.addWidget(gpu_status)
            
            # Display each GPU
            for idx, gpu in enumerate(gpus):
                gpu_row = QHBoxLayout()
                gpu_name = gpu.get("name", "Unknown GPU")
                gpu_mem = gpu.get("memory", "Unknown")
                
                gpu_label = QLabel(f"<b>GPU {idx}:</b> {gpu_name}")
                gpu_row.addWidget(gpu_label)
                gpu_row.addStretch(1)
                gpu_mem_label = QLabel(f"💾 <b>{gpu_mem}</b>")
                gpu_row.addWidget(gpu_mem_label)
                sys_layout.addLayout(gpu_row)
        else:
            gpu_status = QLabel("⚠️ <b>No GPUs detected</b>")
            gpu_status.setStyleSheet("color: #FF9800; font-weight: bold;")
            sys_layout.addWidget(gpu_status)
            sys_layout.addWidget(QLabel("Training will use CPU (slower)"))

        # Models stats - compact 2-column grid
        stats_grid = QGridLayout()
        stats_grid.setSpacing(8)
        stats_grid.setContentsMargins(0, 0, 0, 0)
        
        # Row 1: Models Trained | Models Downloaded
        stats_grid.addWidget(QLabel("<b>Models Trained</b>"), 0, 0)
        stats_grid.addWidget(QLabel("<b>Models Downloaded</b>"), 0, 1)
        
        # Row 2: Count values
        models_count = QLabel("<span style='font-size: 24pt; font-weight: bold;'>7</span>")
        models_count.setAlignment(Qt.AlignCenter)
        stats_grid.addWidget(models_count, 1, 0)
        
        downloads_count = QLabel("<span style='font-size: 24pt; font-weight: bold;'>5</span>")
        downloads_count.setAlignment(Qt.AlignCenter)
        stats_grid.addWidget(downloads_count, 1, 1)
        
        sys_layout.addLayout(stats_grid)

        sys_layout.addWidget(QLabel("<hr>"))

        # Status - compact single line
        status_row = QHBoxLayout()
        status_label = QLabel("<b>Status:</b>")
        status_row.addWidget(status_label)
        status_val = QLabel("<span style='font-size: 16pt; font-weight: bold; color: #4CAF50;'>Ready</span>")
        status_row.addWidget(status_val)
        status_row.addStretch(1)
        sys_layout.addLayout(status_row)

        right_layout.addWidget(sys_frame)

        # Tips section
        # Software Requirements & Auto-Installer Card (replaces Tips)
        right_layout.addWidget(QLabel("<h2>⚙️ Software Requirements & Setup</h2>"))

        setup_frame = QFrame()
        setup_frame.setFrameShape(QFrame.StyledPanel)
        setup_layout = QVBoxLayout(setup_frame)
        setup_layout.setSpacing(8)
        setup_layout.setContentsMargins(10, 10, 10, 10)
        
        # Check each requirement
        python_info = self.system_info.get("python", {})
        pytorch_info = self.system_info.get("pytorch", {})
        cuda_info = self.system_info.get("cuda", {})
        
        # Python status
        python_status = self._create_status_row(
            "🐍 Python 3.8+",
            python_info.get("found", False),
            f"Version {python_info.get('version', 'N/A')}" if python_info.get("found") else "Not found"
        )
        setup_layout.addLayout(python_status)
        
        # PyTorch status with install button
        pytorch_ok = pytorch_info.get("found", False) and pytorch_info.get("cuda_available", False)
        pytorch_row = QHBoxLayout()
        
        if pytorch_ok:
            pytorch_status_widget = self._create_status_widget(
                "🔥 PyTorch (CUDA)",
                True,
                f"Version {pytorch_info.get('version', 'N/A')}"
            )
        elif pytorch_info.get("found"):
            pytorch_status_widget = self._create_status_widget(
                "🔥 PyTorch (CUDA)",
                False,
                "CPU-only version installed"
            )
        else:
            pytorch_status_widget = self._create_status_widget(
                "🔥 PyTorch (CUDA)",
                False,
                "Not installed"
            )
        
        pytorch_row.addWidget(pytorch_status_widget, 1)
        
        # Add install button if needed
        if not pytorch_ok:
            self.install_pytorch_btn = QPushButton("Install CUDA Version")
            self.install_pytorch_btn.setMaximumWidth(160)
            self.install_pytorch_btn.clicked.connect(self._install_pytorch)
            pytorch_row.addWidget(self.install_pytorch_btn)
        
        setup_layout.addLayout(pytorch_row)
        
        # CUDA drivers status
        cuda_status = self._create_status_row(
            "🎮 CUDA Drivers",
            cuda_info.get("found", False),
            f"Version {cuda_info.get('driver_version', 'N/A')}" if cuda_info.get("found") else "Not found"
        )
        setup_layout.addLayout(cuda_status)
        
        # Dependencies status with install button
        deps_row = QHBoxLayout()
        
        # Check if key packages are installed (check package existence, not import)
        deps_ok = True
        missing_packages = []
        error_message = None
        
        # Use importlib.metadata instead of deprecated pkg_resources
        try:
            from importlib.metadata import version, PackageNotFoundError
        except ImportError:
            # Fallback for Python < 3.8
            from importlib_metadata import version, PackageNotFoundError
        
        required_packages = ['unsloth', 'transformers', 'accelerate', 'peft', 'datasets', 'Pillow']
        
        for pkg in required_packages:
            try:
                version(pkg)
            except PackageNotFoundError:
                deps_ok = False
                missing_packages.append(pkg)
        
        # Try to import unsloth to check for runtime errors
        if deps_ok:
            try:
                import unsloth
                deps_msg = "✅ All packages installed"
            except Exception as e:
                # Packages are installed but there are compatibility issues
                error_str = str(e)
                if "triton" in error_str.lower() or "attrsDescriptor" in error_str:
                    deps_msg = "⚠️ Installed (minor triton warning, will work)"
                    deps_ok = True  # Don't block - this is a known non-critical warning
                else:
                    deps_msg = f"❌ Import error: {error_str[:50]}..."
                    deps_ok = False
        else:
            deps_msg = f"Missing: {', '.join(missing_packages)}"
        
        deps_status_widget = self._create_status_widget(
            "📦 Dependencies",
            deps_ok,
            deps_msg
        )
        deps_row.addWidget(deps_status_widget, 1)
        
        if not deps_ok:
            install_deps_btn = QPushButton("Install Dependencies")
            install_deps_btn.setMaximumWidth(160)
            install_deps_btn.clicked.connect(self._install_dependencies)
            deps_row.addWidget(install_deps_btn)
        
        setup_layout.addLayout(deps_row)
        
        # Installation log area (hidden by default)
        self.install_log = QPlainTextEdit()
        self.install_log.setReadOnly(True)
        self.install_log.setMaximumHeight(150)
        self.install_log.setVisible(False)
        self.install_log.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        setup_layout.addWidget(self.install_log)

        right_layout.addWidget(setup_frame)
        right_layout.addStretch(1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)
        
        # Set equal stretch factors (50/50 split)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        # Apply exact 50/50 split after window is shown
        def _apply_equal_split():
            w = splitter.width() or 1200
            half = w // 2
            splitter.setSizes([half, half])
        
        QTimer.singleShot(0, _apply_equal_split)
        
        layout.addWidget(splitter)
        
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
        # Increased max height to accommodate larger fonts and more content
        downloaded_scroll.setMaximumHeight(600)
        
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
        self.hf_results.setMaximumHeight(350)
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
        self.models_status.setMaximumHeight(220)
        right_layout.addWidget(self.models_status)
        
        # Refresh button at bottom
        refresh_btn = QPushButton("🔄 Refresh Models")
        refresh_btn.setMaximumWidth(200)
        refresh_btn.clicked.connect(self._refresh_models)
        right_layout.addWidget(refresh_btn)
        
        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # Right column = EXACTLY 1/4 width, Left = 3/4
        right_widget.setMaximumWidth(400)  # Cap the right side
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        # Apply actual 3:1 ratio after widget is shown
        def _apply_download_split():
            w = splitter.width() or 1400
            right_w = int(w * 0.25)  # Exactly 1/4
            left_w = w - right_w      # The rest (3/4)
            splitter.setSizes([left_w, right_w])
        
        QTimer.singleShot(0, _apply_download_split)
        
        main_layout.addWidget(splitter)
        
        # Store model cards for theme updates
        self.model_cards = []
        self.downloaded_model_cards = []
        
        # Track active downloads
        self.active_downloads = {}  # model_id -> (thread, card)
        
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
        """Download a curated model in background thread with progress"""
        # Check if already downloading
        if model_id in self.active_downloads:
            self._log_models(f"⚠ {model_id} is already downloading...")
            return
        
        # Find the card
        card = None
        for c in self.model_cards:
            if c.model_id == model_id:
                card = c
                break
        
        if not card:
            return
        
        # IMMEDIATELY disable button to prevent double-clicks
        card.download_btn.setEnabled(False)
        card.download_btn.setText("⏳ Starting...")
        
        self._log_models(f"📥 Downloading {model_id}...")
        target = Path(self.hf_target_dir.text().strip())
        
        # Hide button after a moment (let user see "Starting...")
        QTimer.singleShot(500, lambda: card.download_btn.setVisible(False))
        
        # Create progress bar
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        progress_bar.setFormat("Downloading... %p%")
        progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 4px;
                text-align: center;
                background-color: #262730;
                color: white;
                font-weight: bold;
                height: 30px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 4px;
            }
        """)
        
        # Add progress bar to card layout
        card.layout().addWidget(progress_bar)
        
        # Create download thread
        thread = DownloadThread(model_id, target)
        
        # Mark as active BEFORE starting
        self.active_downloads[model_id] = (thread, card)
        
        # Connect signals
        thread.progress.connect(lambda p: progress_bar.setValue(p))
        thread.finished.connect(lambda dest: self._on_download_complete(model_id, dest, card, progress_bar))
        thread.error.connect(lambda err: self._on_download_error(model_id, err, card, progress_bar))
        
        # Start download
        thread.start()
    
    def _on_download_complete(self, model_id: str, dest: str, card, progress_bar):
        """Handle successful download"""
        self._log_models(f"✓ Downloaded to: {dest}")
        self._log_models(f"DEBUG: Checking if path exists: {Path(dest).exists()}")
        self._log_models(f"DEBUG: Directory name: {Path(dest).name}")
        
        # Clean up thread first
        if model_id in self.active_downloads:
            thread, _ = self.active_downloads[model_id]
            if thread.isRunning():
                thread.quit()
                thread.wait()
            del self.active_downloads[model_id]
        
        # Remove progress bar
        if progress_bar:
            progress_bar.setVisible(False)
            progress_bar.deleteLater()
        
        # Don't hide the card - let _refresh_models handle showing it as downloaded
        
        # Refresh both downloaded models list AND train dropdown
        self._log_models(f"DEBUG: Refreshing models list...")
        self._refresh_models()
        self._refresh_locals()  # This updates the Train tab dropdown
        self._log_models(f"DEBUG: Refresh complete!")
    
    def _on_download_error(self, model_id: str, error: str, card, progress_bar):
        """Handle download error"""
        self._log_models(f"✗ Error downloading {model_id}: {error}")
        
        # Clean up thread first
        if model_id in self.active_downloads:
            thread, _ = self.active_downloads[model_id]
            if thread.isRunning():
                thread.quit()
                thread.wait()
            del self.active_downloads[model_id]
        
        # Remove progress bar
        if progress_bar:
            progress_bar.setVisible(False)
            progress_bar.deleteLater()
        
        # Restore button
        card.download_btn.setVisible(True)
        card.download_btn.setEnabled(True)
        card.download_btn.setText("❌ Failed - Retry")

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
        self.train_base_model.currentTextChanged.connect(self._auto_generate_model_name)  # Auto-generate name
        config_layout.addWidget(self.train_base_model)
        
        # Model info label
        self.model_info_label = QLabel("Select a model to see details")
        self.model_info_label.setWordWrap(True)
        model_info_font = QFont()
        model_info_font.setPointSize(13)
        self.model_info_label.setFont(model_info_font)
        self.model_info_label.setStyleSheet("color: #888;")
        self.model_info_label.setMinimumHeight(50)
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
        self.train_data_path.textChanged.connect(self._auto_generate_model_name)  # Auto-generate name when dataset changes
        dataset_layout.addWidget(self.train_data_path)
        
        browse_btn = QPushButton("📁 Browse Files")
        browse_btn.clicked.connect(self._browse_train_data)
        dataset_layout.addWidget(browse_btn)
        
        # Dataset validation status
        self.dataset_status_label = QLabel("")
        self.dataset_status_label.setWordWrap(True)
        self.dataset_status_label.setMinimumHeight(30)
        dataset_layout.addWidget(self.dataset_status_label)
        
        # Total examples count
        self.examples_label = QLabel("Total Examples: --")
        examples_font = QFont()
        examples_font.setPointSize(15)
        examples_font.setBold(True)
        self.examples_label.setFont(examples_font)
        self.examples_label.setMinimumHeight(40)
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
        
        # Output directory (moved from Advanced Settings)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output dir:"))
        self.train_out_dir = QLineEdit(str(default_output_dir()))
        out_row.addWidget(self.train_out_dir, 2)
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._browse_train_out)
        out_row.addWidget(out_browse)
        params_layout.addLayout(out_row)
        
        # Batch size kept for internal use but not displayed
        self.train_batch = QSpinBox()
        self.train_batch.setRange(1, 512)
        self.train_batch.setValue(2)
        self.train_batch.setVisible(False)  # Hidden, controlled by optimal batch size checkbox
        
        left_layout.addWidget(params_frame)
        
        # GPU Selection Section
        left_layout.addWidget(QLabel("<h2>💻 Select GPU(s) for Training</h2>"))
        
        gpu_frame = QFrame()
        gpu_frame.setFrameShape(QFrame.StyledPanel)
        gpu_layout = QVBoxLayout(gpu_frame)
        
        # GPU status using REAL system detection
        cuda_info = self.system_info.get("cuda", {})
        gpus = cuda_info.get("gpus", [])
        
        if gpus:
            gpu_count = len(gpus)
            self.gpu_status_label = QLabel(f"✅ {gpu_count} GPU{'s' if gpu_count > 1 else ''} detected")
            self.gpu_status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        else:
            self.gpu_status_label = QLabel("⚠️ No GPUs detected")
            self.gpu_status_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        
        gpu_layout.addWidget(self.gpu_status_label)
        
        # GPU selection dropdown
        self.gpu_select = QComboBox()
        
        if gpus:
            for idx, gpu in enumerate(gpus):
                gpu_name = gpu.get("name", f"GPU {idx}")
                self.gpu_select.addItem(f"GPU {idx}: {gpu_name}")
            self.training_info_label = QLabel(f"⚡ Training will use: {self.gpu_select.currentText()}")
        else:
            self.gpu_select.addItem("No GPUs available - CPU mode")
            self.gpu_select.setEnabled(False)
            self.training_info_label = QLabel("⚠️ Training will use CPU (slower)")
        
        # Connect GPU selection change to update label
        self.gpu_select.currentIndexChanged.connect(
            lambda idx: self.training_info_label.setText(
                f"⚡ Training will use: {self.gpu_select.currentText()}"
            )
        )
            
        gpu_layout.addWidget(self.gpu_select)
        
        # Training info
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
                font-size: 16pt;
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
        
        # RIGHT COLUMN: Training Visualization - FUTURISTIC REDESIGN
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(0)  # NO GAPS - containers touch each other
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Training Dashboard Header
        dashboard_header = QLabel("📊 TRAINING DASHBOARD")
        dashboard_header.setAlignment(Qt.AlignCenter)
        dashboard_header.setMinimumHeight(50)
        dashboard_header.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 rgba(102, 126, 234, 0.8), stop:1 rgba(118, 75, 162, 0.8));
                color: white;
                font-size: 16pt;
                font-weight: bold;
                border: none;
                border-radius: 12px;
            }
        """)
        right_layout.addWidget(dashboard_header)
        
        # REMOVED: Status banner (no longer needed)
        
        # Single row with all 4 metrics: EPOCH, STEPS, LOSS, ETA
        metrics_row = QWidget()
        metrics_row.setMinimumHeight(90)
        metrics_row.setMaximumHeight(90)
        metrics_layout = QHBoxLayout(metrics_row)
        metrics_layout.setSpacing(8)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        
        self.epoch_card = self._create_3d_metric_card("EPOCH", "📚", "0/0", "rgba(26, 31, 53, 0.7)")
        self.steps_card = self._create_3d_metric_card("STEPS", "🔥", "0/0", "rgba(31, 26, 53, 0.7)")
        self.loss_card = self._create_3d_metric_card("LOSS", "📉", "--·----", "rgba(53, 34, 26, 0.7)")
        self.eta_card = self._create_3d_metric_card("ETA", "⏱", "--m --s", "rgba(26, 53, 34, 0.7)")
        
        metrics_layout.addWidget(self.epoch_card)
        metrics_layout.addWidget(self.steps_card)
        metrics_layout.addWidget(self.loss_card)
        metrics_layout.addWidget(self.eta_card)
        right_layout.addWidget(metrics_row)
        
        # Second row with LR, SPEED, GPU (same height as first row)
        extra_row = QWidget()
        extra_row.setMinimumHeight(90)
        extra_row.setMaximumHeight(90)
        extra_layout = QHBoxLayout(extra_row)
        extra_layout.setSpacing(8)
        extra_layout.setContentsMargins(0, 0, 0, 0)
        
        self.learning_rate_card = self._create_3d_metric_card("LR", "📊", "--e-0", "rgba(42, 26, 53, 0.7)")
        self.speed_card = self._create_3d_metric_card("SPEED", "🚀", "-- s/s", "rgba(26, 37, 53, 0.7)")
        self.gpu_mem_card = self._create_3d_metric_card("GPU", "💾", "-- GB", "rgba(53, 41, 26, 0.7)")
        
        extra_layout.addWidget(self.learning_rate_card)
        extra_layout.addWidget(self.speed_card)
        extra_layout.addWidget(self.gpu_mem_card)
        right_layout.addWidget(extra_row)
        
        # Loss Over Time Section (60px taller = 240px)
        loss_section = QWidget()
        loss_section.setMinimumHeight(240)
        loss_section.setMaximumHeight(240)
        loss_section_layout = QVBoxLayout(loss_section)
        loss_section_layout.setContentsMargins(15, 10, 15, 10)
        loss_section_layout.setSpacing(5)
        
        loss_title = QLabel("<b>📉 Loss Over Time</b>")
        loss_title.setStyleSheet("color: white; font-size: 12pt;")
        loss_section_layout.addWidget(loss_title)
        
        self.loss_chart_label = QLabel("Loss chart will appear here once training starts...")
        self.loss_chart_label.setAlignment(Qt.AlignCenter)
        self.loss_chart_label.setStyleSheet("color: #888;")
        loss_section_layout.addWidget(self.loss_chart_label, 1)
        
        loss_section.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(15, 15, 26, 0.6), stop:1 rgba(25, 25, 36, 0.8));
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 12px;
            }
        """)
        right_layout.addWidget(loss_section)
        
        # Training Logs (takes remaining space with stretch)
        logs_section = QWidget()
        logs_section_layout = QVBoxLayout(logs_section)
        logs_section_layout.setContentsMargins(15, 10, 15, 10)
        logs_section_layout.setSpacing(5)
        
        logs_header = QHBoxLayout()
        logs_title = QLabel("<b>📋 Training Logs</b>")
        logs_title.setStyleSheet("color: white; font-size: 12pt;")
        logs_header.addWidget(logs_title)
        logs_header.addStretch(1)
        
        self.logs_expand_btn = QPushButton("▼ Show Logs")
        self.logs_expand_btn.setCheckable(True)
        self.logs_expand_btn.setChecked(True)  # Start expanded
        self.logs_expand_btn.clicked.connect(self._toggle_logs)
        self.logs_expand_btn.setStyleSheet("""
            QPushButton {
                background: rgba(102, 126, 234, 0.3);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 5px 15px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background: rgba(102, 126, 234, 0.5);
            }
        """)
        logs_header.addWidget(self.logs_expand_btn)
        logs_section_layout.addLayout(logs_header)
        
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumBlockCount(10000)
        # NO maximum height - let it expand to fill remaining space!
        self.train_log.setMinimumHeight(200)  # Just a minimum
        self.train_log.setVisible(True)  # Start visible
        self.train_log.setStyleSheet("""
            QPlainTextEdit {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(10, 10, 15, 0.9), stop:1 rgba(20, 20, 25, 0.9));
                color: #00ff00;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11pt;
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 8px;
                padding: 10px;
            }
        """)
        logs_section_layout.addWidget(self.train_log, 1)  # Stretch=1 to fill all remaining space!
        
        logs_section.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(15, 15, 26, 0.6), stop:1 rgba(25, 25, 36, 0.8));
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 12px;
            }
        """)
        right_layout.addWidget(logs_section, 1)  # Stretch=1 to expand with window!
        
        # NO addStretch here - let logs_section take all remaining space!
        
        # Add to splitter
        # Left column is long; make it scrollable so larger fonts don't get clipped.
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left_widget)

        # Give each side a sensible minimum so the dashboard can't get crushed.
        left_scroll.setMinimumWidth(520)
        right_widget.setMinimumWidth(400)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)

        # Dashboard = 1/3 of width, Config = 2/3
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        # Important: apply the initial size AFTER the widget is shown, otherwise Qt may override it.
        def _apply_initial_split():
            w = splitter.width() or 1400
            left_w = int(w * 0.67)  # Config = 2/3
            right_w = w - left_w    # Dashboard = 1/3
            splitter.setSizes([left_w, right_w])

        QTimer.singleShot(0, _apply_initial_split)
        
        main_layout.addWidget(splitter)
        
        # Store metric cards for updates (they're now just QWidgets with value_label and set_value)
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
    
    def _auto_generate_model_name(self):
        """Auto-generate model name based on base model and dataset"""
        from datetime import datetime
        
        # Get base model name (last part after / or __)
        base_model = self.train_base_model.currentText().strip()
        if not base_model or base_model == "(No models downloaded yet)":
            return
        
        # Handle different formats: nvidia__Model or unsloth/model or unsloth__model
        if "__" in base_model:
            model_short = base_model.split("__")[-1]
        elif "/" in base_model:
            model_short = base_model.split('/')[-1]
        else:
            model_short = base_model
        
        # Clean up model name
        model_short = model_short.replace('-bnb-4bit', '').replace('unsloth-', '').replace('_', '-')
        
        # Get dataset name (filename without extension)
        dataset_path = self.train_data_path.text().strip()
        if dataset_path:
            dataset_short = Path(dataset_path).stem
        else:
            dataset_short = "dataset"
        
        # Generate: YYMMDD_modelname_dataset_HHMM
        now = datetime.now()
        auto_name = f"{now.strftime('%y%m%d')}_{model_short}_{dataset_short}_{now.strftime('%H%M')}"
        
        # Always update the name when model changes
        self.train_model_name.setText(auto_name)
    
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
    
    def _create_3d_metric_card(self, title: str, icon: str, value: str, bg_color: str):
        """Create a compact 3D metric card with semi-transparent gradients"""
        card = QWidget()
        card.setMinimumHeight(90)
        card.setMaximumHeight(90)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)
        
        # Title + Icon
        header = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 12pt; background: transparent; border: none;")
        header.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 9pt; font-weight: bold; color: #aaa; background: transparent; border: none;")
        header.addWidget(title_label)
        header.addStretch(1)
        layout.addLayout(header)
        
        layout.addStretch(1)
        
        # Value
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: white; background: transparent; border: none;")
        layout.addWidget(value_label)
        
        layout.addStretch(1)
        
        card.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {bg_color}, stop:1 rgba(20, 20, 30, 0.5));
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 12px;
            }}
        """)
        
        # Store value_label for updates
        card.value_label = value_label
        card.set_value = lambda v: value_label.setText(v)
        
        return card
    
    def _create_compact_metric_card(self, title: str, icon: str, value: str, bg_color: str):
        """Create a futuristic compact metric card with fixed height"""
        card = QWidget()
        card.setMinimumHeight(140)
        card.setMaximumHeight(140)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Title + Icon
        header = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 14pt;")
        header.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #888;")
        header.addWidget(title_label)
        header.addStretch(1)
        layout.addLayout(header)
        
        layout.addStretch(1)
        
        # Value
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: white;")
        layout.addWidget(value_label)
        
        layout.addStretch(1)
        
        card.setStyleSheet(f"""
            QWidget {{
                background: {bg_color};
                border: none;
                border-radius: 12px;
            }}
        """)
        
        # Store value_label for updates
        card.value_label = value_label
        card.set_value = lambda v: value_label.setText(v)
        
        return card
    
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

        # Get model name and convert to proper format for training
        model_name = self.train_base_model.currentText().strip()
        
        # Convert hf_models folder format to HuggingFace ID:
        # meta-llama__Llama-3.2-1B -> meta-llama/Llama-3.2-1B
        # Double underscore __ is used in folder names, but HF needs single slash /
        if "__" in model_name:
            model_name_hf = model_name.replace("__", "/")
        # Handle old single underscore format: nvidia_Llama -> nvidia/Llama
        elif "/" not in model_name and "_" in model_name:
            parts = model_name.split("_", 1)
            if len(parts) == 2:
                model_name_hf = f"{parts[0]}/{parts[1]}"
            else:
                model_name_hf = model_name
        else:
            model_name_hf = model_name
        
        cfg = TrainingConfig(
            base_model=model_name_hf,
            data_path=Path(data_path),
            output_dir=Path(self.train_out_dir.text().strip()),
            epochs=int(self.train_epochs.value()),
            batch_size=int(self.train_batch.value()),
            learning_rate=float(self.train_lr.value()),
        )

        cmd = build_finetune_cmd(cfg)

        # Show logs immediately
        if not self.train_log.isVisible():
            self.logs_expand_btn.setChecked(True)
            self.train_log.setVisible(True)
            self.logs_expand_btn.setText("▲ Hide Logs")

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        
        # Set UTF-8 encoding for Windows to handle emojis in transformers/unsloth output
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        proc.setProcessEnvironment(env)
        
        proc.readyReadStandardOutput.connect(lambda: self._append_proc_output(proc, self.train_log))
        proc.errorOccurred.connect(lambda err: self._on_training_error(proc, err))
        proc.finished.connect(self._train_finished)

        self.train_log.clear()
        self.train_log.appendPlainText("=== Starting Training ===")
        self.train_log.appendPlainText(f"Command: {' '.join(cmd)}")
        self.train_log.appendPlainText("=" * 50)
        
        # Enable/disable buttons
        self.train_start.setEnabled(False)
        self.train_stop.setEnabled(True)
        
        # Start process
        proc.start()
        
        if not proc.waitForStarted(3000):
            self.train_log.appendPlainText("\n[ERROR] Failed to start training process!")
            self.train_start.setEnabled(True)
            self.train_stop.setEnabled(False)
            return
            
        self.train_proc = proc
        self.train_log.appendPlainText("[INFO] Training process started successfully")
        if not proc.waitForStarted(5000):
            QMessageBox.critical(self, "Training", "Failed to start training process.")
            return

        self.train_proc = proc
        self.train_start.setEnabled(False)
        self.train_stop.setEnabled(True)

    def _on_training_error(self, proc, error):
        """Handle training process errors"""
        error_msgs = {
            QProcess.FailedToStart: "Failed to start (check if Python is installed)",
            QProcess.Crashed: "Process crashed",
            QProcess.Timedout: "Process timed out",
            QProcess.WriteError: "Write error",
            QProcess.ReadError: "Read error",
            QProcess.UnknownError: "Unknown error"
        }
        error_msg = error_msgs.get(error, f"Error code: {error}")
        self.train_log.appendPlainText(f"\n[ERROR] {error_msg}")
        self.train_start.setEnabled(True)
        self.train_stop.setEnabled(False)

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
        header_a.setStyleSheet("font-size: 16pt; padding: 10px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2); color: white; border-radius: 6px;")
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
        header_b.setStyleSheet("font-size: 16pt; padding: 10px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4CAF50, stop:1 #2E7D32); color: white; border-radius: 6px;")
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
        self.test_prompt.setMinimumHeight(120)
        self.test_prompt.setMaximumHeight(120)
        prompt_layout.addWidget(self.test_prompt)
        
        # Buttons row
        btn_layout = QHBoxLayout()
        self.test_send_btn = QPushButton("📤 Send")
        self.test_send_btn.clicked.connect(self._run_side_by_side_test)
        self.test_send_btn.setMinimumHeight(50)
        self.test_send_btn.setStyleSheet("""
            QPushButton {
                font-size: 14pt;
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

    # ---------------- Info/About tab ----------------
    def _build_info_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("<h1>ℹ️ About LLM Fine-tuning Studio</h1>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Two-column layout using QSplitter for fixed 50/50 split
        splitter = QSplitter(Qt.Horizontal)
        
        # LEFT COLUMN: Credits
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        
        credits_widget = QWidget()
        credits_layout = QVBoxLayout(credits_widget)
        credits_layout.setSpacing(15)
        
        credits_frame = QFrame()
        credits_frame.setFrameShape(QFrame.StyledPanel)
        credits_inner = QVBoxLayout(credits_frame)
        credits_inner.setSpacing(12)
        
        credits_title = QLabel("<h2>💝 Credits</h2>")
        credits_inner.addWidget(credits_title)
        
        credits_text = QLabel("""
<p style="line-height: 1.4;">
<b>LLM Fine-tuning Studio</b> - A user-friendly desktop application for fine-tuning Large Language Models<br><br>

<b>Development:</b><br>
• Built with modern Python technologies and AI-assisted development<br>
• Designed for researchers, developers, and AI enthusiasts<br><br>

<b>Special Thanks:</b><br>
• <b>Unsloth AI</b> - For the incredible Unsloth library that makes training 2x faster<br>
• <b>Hugging Face</b> - For transformers, datasets, and the model hub<br>
• <b>Meta AI</b> - For the Llama model family<br>
• <b>NVIDIA</b> - For CUDA and GPU acceleration technologies<br>
• <b>The Open Source Community</b> - For all the amazing tools and libraries
</p>
        """)
        credits_text.setWordWrap(True)
        credits_text.setTextFormat(Qt.RichText)
        credits_inner.addWidget(credits_text)
        credits_inner.addStretch(1)
        
        credits_layout.addWidget(credits_frame)
        left_scroll.setWidget(credits_widget)
        splitter.addWidget(left_scroll)
        
        # RIGHT COLUMN: Licenses
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        
        license_widget = QWidget()
        license_layout = QVBoxLayout(license_widget)
        license_layout.setSpacing(15)
        
        license_frame = QFrame()
        license_frame.setFrameShape(QFrame.StyledPanel)
        license_inner = QVBoxLayout(license_frame)
        license_inner.setSpacing(12)
        
        license_title = QLabel("<h2>📜 License Information</h2>")
        license_inner.addWidget(license_title)
        
        license_text = QLabel("""
<p style="line-height: 1.3; font-size: 12pt;">
This application uses the following open-source libraries and tools:
</p>

<h3 style="color: #667eea; margin-top: 12px;">Core Libraries</h3>
<ul style="line-height: 1.4;">
<li><b>Python 3.10+</b> - PSF License</li>
<li><b>PySide6 (Qt for Python)</b> - LGPL v3<br>
    <span style="color: #888;">GUI framework</span></li>
<li><b>PyTorch</b> - BSD-3-Clause<br>
    <span style="color: #888;">Deep learning framework</span></li>
<li><b>Transformers</b> - Apache 2.0 (HuggingFace)<br>
    <span style="color: #888;">NLP models and tokenizers</span></li>
<li><b>Unsloth</b> - Apache 2.0<br>
    <span style="color: #888;">Fast LLM fine-tuning</span></li>
</ul>

<h3 style="color: #667eea; margin-top: 12px;">Training & Data</h3>
<ul style="line-height: 1.4;">
<li><b>TRL</b> - Apache 2.0<br>
    <span style="color: #888;">SFTTrainer for supervised fine-tuning</span></li>
<li><b>Datasets</b> - Apache 2.0<br>
    <span style="color: #888;">Dataset loading and processing</span></li>
<li><b>PEFT</b> - Apache 2.0<br>
    <span style="color: #888;">LoRA and efficient training</span></li>
<li><b>BitsAndBytes</b> - MIT<br>
    <span style="color: #888;">4-bit/8-bit quantization</span></li>
</ul>

<h3 style="color: #667eea; margin-top: 12px;">Acceleration</h3>
<ul style="line-height: 1.4;">
<li><b>xFormers</b> - BSD (Meta)<br>
    <span style="color: #888;">Memory-efficient attention</span></li>
<li><b>CUDA Toolkit 12.4</b> - NVIDIA EULA<br>
    <span style="color: #888;">GPU acceleration</span></li>
<li><b>Triton</b> - MIT (OpenAI)<br>
    <span style="color: #888;">GPU programming</span></li>
</ul>

<h3 style="color: #667eea; margin-top: 12px;">Utilities</h3>
<ul style="line-height: 1.4;">
<li><b>huggingface_hub</b> - Apache 2.0</li>
<li><b>psutil</b> - BSD-3-Clause</li>
<li><b>pandas</b> - BSD-3-Clause</li>
<li><b>numpy</b> - BSD-3-Clause</li>
</ul>

<h3 style="color: #667eea; margin-top: 12px;">Models</h3>
<ul style="line-height: 1.4;">
<li><b>Llama Models</b> - Llama Community License (Meta)<br>
    <span style="color: #888;">Commercial use with restrictions</span></li>
<li><b>Other Models</b> - Various licenses<br>
    <span style="color: #888;">Check model card on Hugging Face</span></li>
</ul>

<h3 style="color: #667eea; margin-top: 12px;">Important Notes</h3>
<p style="line-height: 1.4; background: #2a2a2a; padding: 12px; border-radius: 8px; border-left: 4px solid #667eea;">
⚠️ <b>Disclaimer:</b> This application is provided "AS IS" without warranty. 
Users are responsible for complying with all applicable licenses.<br><br>

📖 <b>Full Licenses:</b> Complete license texts can be found in their 
respective package directories or official repositories.
</p>
        """)
        license_text.setWordWrap(True)
        license_text.setTextFormat(Qt.RichText)
        license_text.setOpenExternalLinks(True)
        license_inner.addWidget(license_text)
        license_inner.addStretch(1)
        
        license_layout.addWidget(license_frame)
        right_scroll.setWidget(license_widget)
        splitter.addWidget(right_scroll)
        
        # Set 50/50 split
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        # Apply exact 50/50 split after window is shown
        def _apply_equal_split():
            w = splitter.width() or 1200
            half = w // 2
            splitter.setSizes([half, half])
        
        QTimer.singleShot(0, _apply_equal_split)
        
        layout.addWidget(splitter)
        
        return w
    
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
        # Refresh downloaded models display
        self._refresh_models()
        
        # Get downloaded models from the models directory (where downloads actually go)
        models_dir = self.root / "models"
        downloaded_models = []
        
        if models_dir.exists():
            for model_dir in sorted(models_dir.iterdir()):
                if model_dir.is_dir():
                    # Check if it has actual model weights (not just config files)
                    has_weights = any(
                        (model_dir / f).exists() 
                        for f in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]
                    )
                    if has_weights or len(list(model_dir.glob("*.safetensors"))) > 0:
                        downloaded_models.append(model_dir.name)
        
        # Update Train tab: Select Base Model dropdown with downloaded models
        current_train = self.train_base_model.currentText()
        self.train_base_model.clear()
        
        if downloaded_models:
            for model_name in downloaded_models:
                self.train_base_model.addItem(model_name)
        else:
            self.train_base_model.addItem("(No models downloaded yet)")
        
        if current_train:
            idx = self.train_base_model.findText(current_train)
            if idx >= 0:
                self.train_base_model.setCurrentIndex(idx)
        
        # Update Test tab Model A dropdown with downloaded models + trained adapters
        current_a = self.test_model_a.currentText()
        self.test_model_a.clear()
        self.test_model_a.addItem("None")
        
        # Add base models
        if downloaded_models:
            for model_name in downloaded_models:
                self.test_model_a.addItem(f"📦 {model_name}", str(models_dir / model_name))
        
        # Add trained adapters
        adapter_dir = self.root / "fine_tuned_adapter"
        if adapter_dir.exists():
            trained_adapters = sorted([d for d in adapter_dir.iterdir() if d.is_dir() and (d / "adapter_config.json").exists()])
            if trained_adapters:
                for adapter_path in trained_adapters:
                    adapter_name = adapter_path.name
                    self.test_model_a.addItem(f"🎯 {adapter_name}", str(adapter_path))
        
        if not downloaded_models and (not adapter_dir.exists() or not trained_adapters):
            self.test_model_a.addItem("(No models available)")
        
        if current_a and current_a != "None":
            idx = self.test_model_a.findText(current_a)
            if idx >= 0:
                self.test_model_a.setCurrentIndex(idx)
        
        # Update Model B dropdown
        current_b = self.test_model_b.currentText()
        self.test_model_b.clear()
        self.test_model_b.addItem("None")
        
        # Add base models
        if downloaded_models:
            for model_name in downloaded_models:
                self.test_model_b.addItem(f"📦 {model_name}", str(models_dir / model_name))
        
        # Add trained adapters
        if adapter_dir.exists():
            trained_adapters = sorted([d for d in adapter_dir.iterdir() if d.is_dir() and (d / "adapter_config.json").exists()])
            if trained_adapters:
                for adapter_path in trained_adapters:
                    adapter_name = adapter_path.name
                    self.test_model_b.addItem(f"🎯 {adapter_name}", str(adapter_path))
        
        if not downloaded_models and (not adapter_dir.exists() or not trained_adapters):
            self.test_model_b.addItem("(No models available)")
        
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
    # Set base font size for the entire application - INCREASED to 16pt
    app_font = QFont()
    # Keep this modest; very large fonts require scroll areas (added above).
    app_font.setPointSize(14)
    app.setFont(app_font)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
