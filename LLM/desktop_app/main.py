from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer, QThread, Signal, QProcessEnvironment, QRect, QSize
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QTextEdit, QPlainTextEdit,
    QSpinBox, QDoubleSpinBox, QMessageBox, QListWidget, QListWidgetItem, QSplitter, QToolBar, QScrollArea, QGridLayout, QFrame, QProgressBar, QSizePolicy, QTabBar, QStyleOptionTab, QStyle, QStackedWidget
)
from PySide6.QtGui import QAction, QIcon, QFont

from desktop_app.model_card_widget import ModelCard, DownloadedModelCard
from desktop_app.training_widgets import MetricCard
from desktop_app.chat_widget import ChatWidget

from system_detector import SystemDetector
from smart_installer import SmartInstaller
from setup_state import SetupStateManager
from model_integrity_checker import ModelIntegrityChecker
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
            self.log_output.emit("Detecting system configuration...")
            installer.run_detection()
            
            # Install based on type
            if self.install_type == "pytorch":
                success = installer.install_pytorch()
            elif self.install_type == "dependencies":
                success = installer.install_dependencies()
            elif self.install_type == "repair":
                self.log_output.emit("Starting repair process...")
                success = installer.repair_all()
                self.log_output.emit(f"Repair completed with result: {success}")
            else:  # "all"
                success = installer.install()
            
            self.finished_signal.emit(success)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log_output.emit(f"\n[ERROR] Installation failed with exception:")
            self.log_output.emit(str(e))
            self.log_output.emit("\nFull traceback:")
            self.log_output.emit(error_details)
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
        
        # Model integrity checker
        self.model_checker = ModelIntegrityChecker()
        
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
        self.requirements_btn = QPushButton("🔧")  # Tool icon only
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
        
        for btn in [self.home_btn, self.train_btn, self.download_btn, self.test_btn, self.logs_btn, self.requirements_btn, self.info_btn]:
            btn.setCheckable(True)
            btn.setStyleSheet(tab_button_style)
        
        # Special styling for requirements button (icon only, smaller)
        self.requirements_btn.setMaximumWidth(60)
        
        # Add left-side buttons
        navbar_layout.addWidget(self.home_btn)
        navbar_layout.addWidget(self.train_btn)
        navbar_layout.addWidget(self.download_btn)
        navbar_layout.addWidget(self.test_btn)
        navbar_layout.addWidget(self.logs_btn)
        
        # Add stretch to consume remaining space
        navbar_layout.addStretch(1)
        
        # Add Requirements and Info buttons on far right
        navbar_layout.addWidget(self.requirements_btn)
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
        tabs.addTab(self._build_requirements_tab(), "Requirements")
        tabs.addTab(self._build_info_tab(), "Info")
        
        # Connect buttons to tab switching
        self.home_btn.clicked.connect(lambda: self._switch_tab(tabs, 0))
        self.train_btn.clicked.connect(lambda: self._switch_tab(tabs, 1))
        self.download_btn.clicked.connect(lambda: self._switch_tab(tabs, 2))
        self.test_btn.clicked.connect(lambda: self._switch_tab(tabs, 3))
        self.logs_btn.clicked.connect(lambda: self._switch_tab(tabs, 4))
        self.requirements_btn.clicked.connect(lambda: self._switch_tab(tabs, 5))
        self.info_btn.clicked.connect(lambda: self._switch_tab(tabs, 6))
        
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
        buttons = [self.home_btn, self.train_btn, self.download_btn, self.test_btn, self.logs_btn, self.requirements_btn, self.info_btn]
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
        """Fix Issues (repair mode): deterministically repair PyTorch + deps"""
        reply = QMessageBox.question(
            self,
            "Fix Issues",
            "This will automatically fix the environment:\n"
            "• Force reinstall PyTorch with the correct CUDA build\n"
            "• Fix common corruption (missing metadata / partial installs)\n"
            "• Install correct Triton for Windows\n"
            "• Install all required dependencies (unsloth, transformers, etc.)\n"
            "• Remove xformers if it would break torch compatibility\n\n"
            "Time: 10-15 minutes\n"
            "Download size: ~2.5GB\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Show log area
        self.install_log.setVisible(True)
        self.install_log.clear()
        self.install_log.appendPlainText("=== Fix Issues (Repair Mode) ===\n")
        self.install_log.appendPlainText("This will repair PyTorch CUDA + dependencies and verify the environment.\n\n")
        
        # Start installer thread with "repair" (deterministic self-heal)
        self.installer_thread = InstallerThread("repair")
        
        def append_log(msg):
            self.install_log.appendPlainText(msg)
            # Auto-scroll to bottom
            cursor = self.install_log.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.install_log.setTextCursor(cursor)
        
        self.installer_thread.log_output.connect(append_log)
        self.installer_thread.finished_signal.connect(self._on_install_complete)
        self.installer_thread.start()
    
    def _on_install_complete(self, success: bool):
        """Handle installation completion"""
        if success:
            self.install_log.appendPlainText("\n✅ Installation completed successfully!")
            reply = QMessageBox.information(
                self,
                "Installation Complete",
                "✅ Installation completed successfully!\n\n"
                "The application will restart automatically when you click OK.",
                QMessageBox.Ok
            )
            
            # Restart application after user clicks OK
            if reply == QMessageBox.Ok:
                self._restart_application()
        else:
            self.install_log.appendPlainText("\n❌ Installation failed!")
            
            # Write error to log files so it appears in Logs tab
            from pathlib import Path
            logs_dir = Path(__file__).parent.parent / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            app_log_path = logs_dir / "app.log"
            repair_log_path = logs_dir / "auto_repair.log"
            
            # Get install log content
            log_content = self.install_log.toPlainText()
            
            # Append to app.log
            try:
                with open(app_log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n=== Fix Issues Failed ===\n")
                    f.write(log_content)
                    f.write("\n=====================================\n")
            except Exception as e:
                print(f"Failed to write to app.log: {e}")
            
            # Write to auto_repair.log
            try:
                with open(repair_log_path, "w", encoding="utf-8") as f:
                    f.write(log_content)
            except Exception as e:
                print(f"Failed to write to auto_repair.log: {e}")
            
            # Refresh Logs tab to show new files
            if hasattr(self, 'logs_list'):
                self._refresh_locals()
            
            QMessageBox.critical(
                self,
                "Installation Failed",
                "❌ Installation failed.\n\n"
                "The error has been written to logs\\app.log and logs\\auto_repair.log\n"
                "Check the Logs tab for details."
            )
        
        # Re-enable button
        if hasattr(self, 'install_pytorch_btn'):
            self.install_pytorch_btn.setEnabled(True)
            self.install_pytorch_btn.setText("Install CUDA Version")
    
    def _restart_application(self):
        """Restart the application"""
        import sys
        import os
        from pathlib import Path
        
        # Get the path to launcher.exe
        app_dir = Path(__file__).parent.parent
        launcher_exe = app_dir / "launcher.exe"
        
        if launcher_exe.exists():
            # Launch using launcher.exe
            import subprocess
            subprocess.Popen([str(launcher_exe)], cwd=str(app_dir))
        else:
            # Fallback: restart with python
            python = sys.executable
            subprocess.Popen([python, "-m", "desktop_app.main"], cwd=str(app_dir))
        
        # Close current instance
        QApplication.quit()
    
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
        
        # Welcome title with 12px top and bottom margin
        title = QLabel("<h1>Welcome to LLM Fine-tuning Studio</h1>")
        title.setAlignment(Qt.AlignCenter)
        title.setContentsMargins(0, 12, 0, 12)
        layout.addWidget(title)
        
        # Create 2-column layout with FIXED 40/60 ratio (not resizable)
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(20)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        
        # LEFT COLUMN: Features + Quick Start Guide (40% width)
        left_container = QFrame()
        left_container.setFrameShape(QFrame.StyledPanel)
        left_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(60, 60, 80, 0.4), stop:1 rgba(40, 40, 60, 0.4));
                border: 2px solid #667eea;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        # Features section
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
        features_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        left_layout.addWidget(features_text)
        
        # Quick Start Guide section
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
        guide_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        left_layout.addWidget(guide_text)
        
        left_layout.addStretch(1)
        
        # Add left container with 40% stretch (2 parts out of 5 total = 40%)
        columns_layout.addWidget(left_container, 2)
        
        # RIGHT COLUMN: System Status + Software Requirements (60% width)
        right_container = QFrame()
        right_container.setFrameShape(QFrame.StyledPanel)
        right_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(60, 60, 80, 0.4), stop:1 rgba(40, 40, 60, 0.4));
                border: 2px solid #667eea;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        right_layout = QVBoxLayout(right_container)
        right_layout.setSpacing(15)
        right_layout.setContentsMargins(15, 15, 15, 15)

        # System Status section
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
        
        # Software Requirements section
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
        
        # We intentionally do NOT show a separate PyTorch button here.
        # Use the single "Fix Issues" button (below) to repair everything deterministically.
        
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
        
        required_packages = ['unsloth', 'transformers', 'accelerate', 'peft', 'datasets', 'Pillow', 'numpy', 'bitsandbytes', 'tokenizers']
        
        # Version requirements for critical packages - mirrors requirements.txt
        version_requirements = {
            'numpy': {'operator': '<', 'version': '2.0.0', 'reason': 'Torch/NumPy 2.x incompatibility'},
            'transformers': {'operator': '==', 'version': '4.51.3', 'reason': 'Unsloth compatibility'},
            'tokenizers': {'operator': '>=', 'min_version': '0.21', 'operator2': '<', 'max_version': '0.22', 'reason': 'Transformers 4.51.3 requirement'},
            'datasets': {'operator': '<', 'version': '4.4.0', 'reason': 'Unsloth compatibility'},
            'accelerate': {'operator': '>=', 'version': '0.18.0', 'reason': 'Training requirement'},
            'peft': {'operator': '>=', 'version': '0.3.0', 'reason': 'LoRA training'},
            'bitsandbytes': {'operator': '>=', 'version': '0.39.0', 'reason': '4-bit quantization'},
        }
        
        for pkg in required_packages:
            try:
                installed_version = version(pkg)
                
                # Check version compatibility for packages with requirements
                if pkg in version_requirements:
                    req = version_requirements[pkg]
                    try:
                        from packaging import version as pkg_version
                        installed = pkg_version.parse(installed_version)
                        
                        # Handle different operator types
                        if req['operator'] == '==':
                            target = pkg_version.parse(req['version'])
                            if installed != target:
                                deps_ok = False
                                missing_packages.append(f"{pkg} (need =={req['version']}, have {installed_version})")
                        
                        elif req['operator'] == '<':
                            target = pkg_version.parse(req['version'])
                            if installed >= target:
                                deps_ok = False
                                missing_packages.append(f"{pkg} (need <{req['version']}, have {installed_version})")
                        
                        elif req['operator'] == '>=':
                            min_ver = pkg_version.parse(req['version'])
                            if installed < min_ver:
                                deps_ok = False
                                missing_packages.append(f"{pkg} (need >={req['version']}, have {installed_version})")
                            
                            # Check max version if specified (for range checks like tokenizers)
                            if 'operator2' in req and req['operator2'] == '<':
                                max_ver = pkg_version.parse(req['max_version'])
                                if installed >= max_ver:
                                    deps_ok = False
                                    missing_packages.append(f"{pkg} (need <{req['max_version']}, have {installed_version})")
                    
                    except Exception as e:
                        # If version parsing fails, log but don't block
                        print(f"Version check error for {pkg}: {e}")
                
            except PackageNotFoundError:
                deps_ok = False
                missing_packages.append(pkg)
        
        # Runtime import test for transformers to detect broken installations
        # (e.g., missing transformers.modeling_layers submodule)
        if deps_ok and 'transformers' in required_packages:
            try:
                import transformers.modeling_layers
                # If import succeeds, transformers is properly installed
            except ImportError as e:
                deps_ok = False
                missing_packages.append(f"transformers (corrupted: {str(e)[:50]})")
            except Exception:
                # Other errors (e.g., CUDA issues) are not dependency problems
                pass
        
        # IMPORTANT:
        # Do NOT import unsloth (or other heavy deps) inside the GUI process.
        # A broken native extension (e.g. xformers) can crash pythonw.exe with a Windows loader dialog
        # before the user can click "Fix Issues". We only check package metadata here.
        if deps_ok:
            deps_msg = "✅ Packages installed (runtime validated by Fix Issues)"
        else:
            deps_msg = f"Missing: {', '.join(missing_packages)}"
        
        deps_status_widget = self._create_status_widget(
            "📦 Dependencies",
            deps_ok,
            deps_msg
        )
        deps_row.addWidget(deps_status_widget, 1)
        
        # Single Fix Issues button (repair mode)
        if (not pytorch_ok) or (not deps_ok):
            fix_btn = QPushButton("🛠️ Fix Issues (Recommended)")
            fix_btn.setMinimumHeight(42)
            fix_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #f093fb, stop:1 #f5576c);
                    color: white;
                    font-size: 13pt;
                    font-weight: bold;
                    border-radius: 8px;
                    padding: 10px 18px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #f5a3ff, stop:1 #ff6a7e);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #d083db, stop:1 #d5475c);
                }
            """)
            fix_btn.clicked.connect(self._install_dependencies)
            setup_layout.addWidget(fix_btn)
        
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
        
        # Add right container with 60% stretch (3 parts out of 5 total = 60%)
        columns_layout.addWidget(right_container, 3)
        
        # Add columns to main layout
        layout.addLayout(columns_layout)
        
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
        
        # Check model integrity and show warnings
        incomplete_models = self.model_checker.get_incomplete_models()
        if incomplete_models:
            self._log_models(f"⚠️ Warning: Found {len(incomplete_models)} incomplete model(s) - missing weights/config files")
            for model_status in incomplete_models:
                self._log_models(f"   ✗ {model_status.model_name} - Missing: {', '.join(model_status.missing_files)}")
        
        # Downloaded models (vertical list on right)
        models_dir = self.root / "models"
        if models_dir.exists():
            for model_dir in sorted(models_dir.iterdir()):
                if model_dir.is_dir():
                    model_name = model_dir.name
                    
                    # Check if model is complete
                    status = self.model_checker.check_model(model_dir)
                    if not status.is_complete:
                        # Show as incomplete with warning
                        size = "⚠️ INCOMPLETE"
                        icons = "❌"
                    else:
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
        
        # TOP ROW: Model and Dataset in 2 columns
        top_row_header = QLabel("<h2>🎯 Model & Dataset Configuration</h2>")
        left_layout.addWidget(top_row_header)
        
        top_row_widget = QWidget()
        top_row_layout = QHBoxLayout(top_row_widget)
        top_row_layout.setSpacing(20)
        top_row_layout.setContentsMargins(0, 0, 0, 0)
        
        # LEFT SUB-COLUMN: Model Configuration
        model_frame = QFrame()
        model_frame.setFrameShape(QFrame.StyledPanel)
        model_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(60, 60, 80, 0.4), stop:1 rgba(40, 40, 60, 0.4));
                border: 2px solid #667eea;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        model_layout = QVBoxLayout(model_frame)
        model_layout.setSpacing(12)
        
        model_header = QLabel("🤖 <b>Select Base Model</b>")
        model_header.setStyleSheet("font-size: 14pt; color: #667eea; border: none; padding: 0;")
        model_layout.addWidget(model_header)
        
        self.train_base_model = QComboBox()
        self.train_base_model.setEditable(True)
        self.train_base_model.addItems(DEFAULT_BASE_MODELS)
        self.train_base_model.currentTextChanged.connect(self._on_model_selected_for_training)
        self.train_base_model.currentTextChanged.connect(self._auto_generate_model_name)
        model_layout.addWidget(self.train_base_model)
        
        # Model info label
        self.model_info_label = QLabel("Select a model to see details")
        self.model_info_label.setWordWrap(True)
        model_info_font = QFont()
        model_info_font.setPointSize(13)
        self.model_info_label.setFont(model_info_font)
        self.model_info_label.setStyleSheet("color: #888;")
        self.model_info_label.setMinimumHeight(50)
        model_layout.addWidget(self.model_info_label)
        
        top_row_layout.addWidget(model_frame, 1)
        
        # RIGHT SUB-COLUMN: Dataset Upload
        dataset_frame = QFrame()
        dataset_frame.setFrameShape(QFrame.StyledPanel)
        dataset_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(60, 80, 60, 0.4), stop:1 rgba(40, 60, 40, 0.4));
                border: 2px solid #4CAF50;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        dataset_layout = QVBoxLayout(dataset_frame)
        dataset_layout.setSpacing(10)
        
        dataset_header = QLabel("📊 <b>Upload Training Dataset</b>")
        dataset_header.setStyleSheet("font-size: 14pt; color: #4CAF50; border: none; padding: 0;")
        dataset_layout.addWidget(dataset_header)
        
        self.train_data_path = QLineEdit()
        self.train_data_path.setPlaceholderText("Drag and drop file or browse...")
        self.train_data_path.textChanged.connect(self._validate_dataset)
        self.train_data_path.textChanged.connect(self._auto_generate_model_name)
        dataset_layout.addWidget(self.train_data_path)
        
        dataset_btn_row = QHBoxLayout()
        browse_btn = QPushButton("📁 Browse")
        browse_btn.clicked.connect(self._browse_train_data)
        dataset_btn_row.addWidget(browse_btn)
        
        check_btn = QPushButton("🔍 Check Dataset")
        check_btn.clicked.connect(self._check_dataset)
        dataset_btn_row.addWidget(check_btn)
        dataset_layout.addLayout(dataset_btn_row)
        
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
        
        top_row_layout.addWidget(dataset_frame, 1)
        
        left_layout.addWidget(top_row_widget)
        
        # Training Parameters Section
        left_layout.addWidget(QLabel("<h2>⚙️ Training Parameters</h2>"))
        
        params_frame = QFrame()
        params_frame.setFrameShape(QFrame.StyledPanel)
        params_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(70, 60, 90, 0.4), stop:1 rgba(50, 40, 70, 0.4));
                border: 2px solid #9c27b0;
                border-radius: 12px;
                padding: 15px;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 8px;
                font-size: 13pt;
                font-weight: bold;
                min-width: 100px;
                max-width: 140px;
                border: 2px solid #555;
                border-radius: 6px;
                background: rgba(50, 50, 70, 0.6);
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: right;
                width: 24px;
                height: 22px;
                border-left: 1px solid #666;
                background: rgba(80, 80, 100, 0.8);
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: left;
                width: 24px;
                height: 22px;
                border-right: 1px solid #666;
                background: rgba(80, 80, 100, 0.8);
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-bottom: 8px solid #4CAF50;
                width: 0px;
                height: 0px;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 8px solid #f44336;
                width: 0px;
                height: 0px;
            }
            QLabel {
                font-size: 11pt;
            }
        """)
        params_layout = QVBoxLayout(params_frame)
        params_layout.setSpacing(15)
        
        # Use recommended settings button
        self.use_recommended_btn = QPushButton("✨ Use Recommended Settings")
        self.use_recommended_btn.setMinimumHeight(40)
        self.use_recommended_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                font-size: 13pt;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7b8ff0, stop:1 #8a5ab8);
            }
        """)
        self.use_recommended_btn.clicked.connect(lambda: (self._use_recommended_settings(), self._switch_to_dashboard()))
        params_layout.addWidget(self.use_recommended_btn)
        
        # Model Name (auto-generated)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("<b>Model Name:</b>"))
        self.train_model_name = QLineEdit()
        self.train_model_name.setPlaceholderText("Auto-generated: YYMMDD_modelname_dataset_HHMM")
        self.train_model_name.setMinimumHeight(35)
        name_row.addWidget(self.train_model_name, 1)
        params_layout.addLayout(name_row)
        
        # Parameters in compact grid
        params_grid = QGridLayout()
        params_grid.setSpacing(12)
        params_grid.setColumnStretch(1, 1)
        params_grid.setColumnStretch(3, 1)
        
        # Row 0: Epochs + LoRA R
        params_grid.addWidget(QLabel("<b>Epochs:</b>"), 0, 0)
        self.train_epochs = QSpinBox()
        self.train_epochs.setRange(1, 1000)
        self.train_epochs.setValue(1)
        self.train_epochs.setMinimumHeight(40)
        params_grid.addWidget(self.train_epochs, 0, 1)
        
        params_grid.addWidget(QLabel("<b>LoRA R:</b>"), 0, 2)
        self.train_lora_r = QSpinBox()
        self.train_lora_r.setRange(8, 256)
        self.train_lora_r.setValue(16)
        self.train_lora_r.setMinimumHeight(40)
        params_grid.addWidget(self.train_lora_r, 0, 3)
        
        # Row 1: Learning Rate + LoRA Alpha (auto-calculated, display only)
        params_grid.addWidget(QLabel("<b>Learning Rate:</b>"), 1, 0)
        self.train_lr = QDoubleSpinBox()
        self.train_lr.setDecimals(6)
        self.train_lr.setRange(1e-6, 1.0)
        self.train_lr.setValue(2e-4)
        self.train_lr.setSingleStep(1e-5)
        self.train_lr.setMinimumHeight(40)
        params_grid.addWidget(self.train_lr, 1, 1)
        
        params_grid.addWidget(QLabel("<b>LoRA Alpha:</b>"), 1, 2)
        self.train_lora_alpha_label = QLabel("32")
        self.train_lora_alpha_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #4CAF50; padding: 8px;")
        self.train_lora_alpha_label.setMinimumHeight(40)
        self.train_lora_alpha_label.setAlignment(Qt.AlignCenter)
        params_grid.addWidget(self.train_lora_alpha_label, 1, 3)
        self.train_lora_r.valueChanged.connect(lambda v: self.train_lora_alpha_label.setText(str(v * 2)))
        
        # Row 2: Max Seq Length + Batch Size toggle
        params_grid.addWidget(QLabel("<b>Max Seq Length:</b>"), 2, 0)
        self.train_max_seq = QSpinBox()
        self.train_max_seq.setRange(128, 8192)
        self.train_max_seq.setValue(2048)
        self.train_max_seq.setSingleStep(128)
        self.train_max_seq.setMinimumHeight(40)
        params_grid.addWidget(self.train_max_seq, 2, 1)
        
        self.batch_size_auto = QPushButton("✅ Optimal batch size")
        self.batch_size_auto.setCheckable(True)
        self.batch_size_auto.setChecked(True)
        self.batch_size_auto.setMinimumHeight(40)
        self.batch_size_auto.setStyleSheet("""
            QPushButton {
                background: rgba(76, 175, 80, 0.3);
                border: 2px solid #4CAF50;
                border-radius: 6px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:checked {
                background: rgba(76, 175, 80, 0.6);
            }
        """)
        self.batch_size_auto.clicked.connect(self._toggle_batch_size)
        params_grid.addWidget(self.batch_size_auto, 2, 2, 1, 2)
        
        params_layout.addLayout(params_grid)
        
        # Output directory - show base path + generated folder name
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("<b>Output:</b>"))
        self.train_out_dir = QLineEdit(str(default_output_dir()))
        self.train_out_dir.setMinimumHeight(35)
        self.train_out_dir.setReadOnly(True)  # Read-only, user browses to change
        self.train_out_dir.setStyleSheet("background: rgba(50, 50, 60, 0.5);")
        output_row.addWidget(self.train_out_dir, 1)
        out_browse = QPushButton("📁 Browse")
        out_browse.setMinimumHeight(35)
        out_browse.setMinimumWidth(100)
        out_browse.clicked.connect(self._browse_train_out)
        output_row.addWidget(out_browse)
        params_layout.addLayout(output_row)
        
        # Batch size kept for internal use but not displayed
        self.train_batch = QSpinBox()
        self.train_batch.setRange(1, 512)
        self.train_batch.setValue(2)
        self.train_batch.setVisible(False)
        
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
        
        # Connect GPU selection change to update label and switch to dashboard
        self.gpu_select.currentIndexChanged.connect(
            lambda idx: (
                self.training_info_label.setText(f"⚡ Training will use: {self.gpu_select.currentText()}"),
                self._switch_to_dashboard()
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
        self.train_start.clicked.connect(lambda: (self._start_training(), self._switch_to_dashboard()))
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
        
        # RIGHT COLUMN: Stacked Widget (Dashboard OR Dataset Viewer)
        self.train_right_stack = QStackedWidget()
        
        # Page 0: Training Dashboard
        dashboard_widget = self._build_training_dashboard()
        self.train_right_stack.addWidget(dashboard_widget)
        
        # Page 1: Dataset Viewer
        dataset_viewer_widget = self._build_dataset_viewer()
        self.train_right_stack.addWidget(dataset_viewer_widget)
        
        # Start with dashboard
        self.train_right_stack.setCurrentIndex(0)
        
        # Add to splitter
        # Left column is long; make it scrollable so larger fonts don't get clipped.
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left_widget)

        # Give each side a sensible minimum so the dashboard can't get crushed.
        left_scroll.setMinimumWidth(520)
        self.train_right_stack.setMinimumWidth(400)

        splitter.addWidget(left_scroll)
        splitter.addWidget(self.train_right_stack)

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
    
    def _check_dataset(self):
        """Check dataset format and switch to dataset viewer"""
        path = self.train_data_path.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "No Dataset", "Please select a dataset file first.")
            return
        
        # Switch to dataset viewer page
        self.train_right_stack.setCurrentIndex(1)
        
        # Load and display dataset preview
        self._load_dataset_preview(path)
    
    def _switch_to_dashboard(self):
        """Switch right panel back to training dashboard"""
        self.train_right_stack.setCurrentIndex(0)
    
    def _load_dataset_preview(self, path: str):
        """Load first few entries from dataset and update viewer"""
        try:
            import json
            examples = []
            
            # Try loading the file (handle both JSONL and JSON array formats)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Try as JSONL first (one object per line)
            try:
                for i, line in enumerate(content.split('\n')[:10]):
                    if line.strip():
                        entry = json.loads(line.strip())
                        examples.append(entry)
            except:
                # Try as JSON array
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        examples = data[:10]
                    elif isinstance(data, dict):
                        # Wrapped format like {"data": [...]} or {"entries": [...]}
                        for key in ['data', 'entries', 'examples', 'dataset', 'items']:
                            if key in data and isinstance(data[key], list):
                                examples = data[key][:10]
                                break
                        # If still empty, treat the dict itself as a single example
                        if not examples:
                            examples = [data]
                except:
                    pass
            
            # Update dataset viewer with loaded examples
            if hasattr(self, 'dataset_preview_text'):
                preview_text = ""
                for i, entry in enumerate(examples, 1):
                    preview_text += f"=== Entry {i} ===\n"
                    preview_text += json.dumps(entry, indent=2, ensure_ascii=False)
                    preview_text += "\n\n"
                
                self.dataset_preview_text.setPlainText(preview_text)
                
                # Validate format - comprehensive field detection
                if examples:
                    first = examples[0]
                    
                    # All possible field names
                    instruction_fields = ['instruction', 'prompt', 'input', 'text', 'question', 'query', 'user', 'human']
                    output_fields = ['output', 'response', 'completion', 'answer', 'assistant', 'reply', 'bot', 'gpt']
                    
                    found_instruction = [k for k in instruction_fields if k in first]
                    found_output = [k for k in output_fields if k in first]
                    has_messages = 'messages' in first
                    
                    # Show detailed format info
                    format_info = f"📊 Dataset has {len(examples)} entries shown\n\n"
                    format_info += f"🔍 Detected fields: {', '.join(first.keys())}\n\n"
                    
                    if found_instruction and found_output:
                        format_info += f"✅ Format: Instruction/Output\n"
                        format_info += f"   - Instruction field: {found_instruction[0]}\n"
                        format_info += f"   - Output field: {found_output[0]}\n"
                        format_info += f"   Status: Compatible ✓"
                        self.dataset_format_label.setText(format_info)
                        self.dataset_format_label.setStyleSheet("color: #4CAF50; font-size: 11pt;")
                    elif has_messages:
                        format_info += f"✅ Format: Chat/Messages\n"
                        format_info += f"   Status: Compatible ✓"
                        self.dataset_format_label.setText(format_info)
                        self.dataset_format_label.setStyleSheet("color: #4CAF50; font-size: 11pt;")
                    else:
                        format_info += f"⚠️ Format: Custom\n"
                        format_info += f"   No standard instruction/output fields found\n"
                        format_info += f"   Available fields: {', '.join(first.keys())}\n"
                        format_info += f"   May require conversion"
                        self.dataset_format_label.setText(format_info)
                        self.dataset_format_label.setStyleSheet("color: #ff9800; font-size: 11pt;")
                else:
                    self.dataset_format_label.setText("⚠️ No entries found in dataset")
                    self.dataset_format_label.setStyleSheet("color: #ff9800; font-size: 11pt;")
        except Exception as e:
            if hasattr(self, 'dataset_preview_text'):
                self.dataset_preview_text.setPlainText(f"Error loading dataset: {str(e)}")
                self.dataset_format_label.setText(f"❌ Error: {str(e)}")
                self.dataset_format_label.setStyleSheet("color: #f44336; font-size: 11pt;")
    
    def _build_training_dashboard(self) -> QWidget:
        """Build the training dashboard widget (right column page 0)"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(0)
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
        
        # Second row with LR, SPEED, GPU
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
        
        # Loss Over Time Section
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
        
        # Training Logs
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
        self.logs_expand_btn.setChecked(True)
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
        self.train_log.setMinimumHeight(200)
        self.train_log.setVisible(True)
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
        logs_section_layout.addWidget(self.train_log, 1)
        
        logs_section.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(15, 15, 26, 0.6), stop:1 rgba(25, 25, 36, 0.8));
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 12px;
            }
        """)
        right_layout.addWidget(logs_section, 1)
        
        return right_widget
    
    def _build_dataset_viewer(self) -> QWidget:
        """Build the dataset viewer widget (right column page 1)"""
        viewer_widget = QWidget()
        viewer_layout = QVBoxLayout(viewer_widget)
        viewer_layout.setSpacing(15)
        viewer_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header
        header = QLabel("<h2>🔍 Dataset Viewer</h2>")
        viewer_layout.addWidget(header)
        
        # Format validation status
        self.dataset_format_label = QLabel("Select a dataset and click 'Check Dataset' to view")
        self.dataset_format_label.setWordWrap(True)
        self.dataset_format_label.setStyleSheet("font-size: 12pt; padding: 10px;")
        viewer_layout.addWidget(self.dataset_format_label)
        
        # Preview text
        preview_label = QLabel("<b>Dataset Preview (first 10 entries):</b>")
        viewer_layout.addWidget(preview_label)
        
        self.dataset_preview_text = QPlainTextEdit()
        self.dataset_preview_text.setReadOnly(True)
        self.dataset_preview_text.setPlaceholderText("Dataset preview will appear here...")
        viewer_layout.addWidget(self.dataset_preview_text, 1)
        
        # Back button
        back_btn = QPushButton("← Back to Dashboard")
        back_btn.clicked.connect(self._switch_to_dashboard)
        viewer_layout.addWidget(back_btn)
        
        return viewer_widget
    
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
    def _build_requirements_tab(self) -> QWidget:
        """Build Requirements tab with hardcoded package list"""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("<h1>🔧 Required Packages & Versions</h1>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Info text
        info = QLabel("These are the exact package versions used by this application:")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #888; font-size: 11pt; margin-bottom: 20px;")
        layout.addWidget(info)
        
        # Package list in scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(15)
        
        # Hardcoded requirements from requirements.txt
        requirements = {
            "Core ML Libraries": [
                ("numpy", "<2", "Pinned for Windows/PyTorch compatibility"),
                ("transformers", "==4.51.3", "Required by unsloth 2025.12.9 (min version: >=4.51.3)"),
                ("tokenizers", ">=0.21,<0.22", "Required by transformers 4.51.3"),
                ("accelerate", ">=0.18.0", "Distributed training"),
                ("datasets", ">=2.11.0,<4.4.0", "Dataset loading (upper bound per unsloth requirements)"),
                ("peft", ">=0.3.0", "LoRA and efficient training"),
                ("safetensors", ">=0.3.0", "Safe model serialization"),
            ],
            "Tokenization": [
                ("sentencepiece", ">=0.1.98", "Text tokenization"),
            ],
            "Quantization & Optimization": [
                ("bitsandbytes", ">=0.39.0", "4-bit/8-bit quantization (Python 3.8+)"),
            ],
            "Utilities": [
                ("evaluate", ">=0.4.0", "Model evaluation metrics"),
                ("filelock", ">=3.0.0", "File locking"),
                ("tqdm", ">=4.60.0", "Progress bars"),
            ],
            "GUI & Visualization": [
                ("streamlit", ">=1.28.0", "Web interface (optional)"),
                ("pandas", ">=2.0.0", "Data manipulation"),
                ("PySide6", "==6.8.1", "Qt desktop GUI (ALL components must match)"),
                ("PySide6-Essentials", "==6.8.1", "PySide6 essentials (must match PySide6 version)"),
                ("PySide6-Addons", "==6.8.1", "PySide6 addons (must match PySide6 version)"),
                ("shiboken6", "==6.8.1", "PySide6 binding generator (must match PySide6 version)"),
            ],
            "Hugging Face Integration": [
                ("huggingface_hub", ">=0.36.0", "Model download/upload"),
            ],
            "Vision Model Support": [
                ("psutil", ">=5.9.0", "System monitoring"),
                ("timm", ">=0.9.0", "Vision models"),
                ("einops", ">=0.6.0", "Tensor operations"),
                ("open-clip-torch", ">=2.20.0", "CLIP models"),
                ("Pillow", "", "Image processing"),
            ],
            "PyTorch Ecosystem (SmartInstaller)": [
                ("torch", "==2.5.1+cu118", "Deep learning framework (CUDA 11.8)"),
                ("torchvision", "==0.20.1+cu118", "Computer vision"),
                ("torchaudio", "==2.5.1+cu118", "Audio processing"),
                ("triton-windows", "==3.5.1.post22", "GPU programming (Windows)"),
            ],
            "Fast Fine-tuning": [
                ("unsloth", "2025.12.9", "2x faster LLM fine-tuning"),
                ("unsloth_zoo", "2025.12.7", "Unsloth model patches"),
            ],
        }
        
        for category, packages in requirements.items():
            # Category header
            cat_label = QLabel(f"<h3 style='color: #667eea;'>{category}</h3>")
            content_layout.addWidget(cat_label)
            
            # Packages in this category
            for pkg_name, version, description in packages:
                pkg_frame = QFrame()
                pkg_frame.setFrameShape(QFrame.StyledPanel)
                pkg_frame.setStyleSheet("""
                    QFrame {
                        background: rgba(60, 60, 80, 0.3);
                        border: 1px solid #555;
                        border-radius: 6px;
                        padding: 8px;
                    }
                """)
                pkg_layout = QVBoxLayout(pkg_frame)
                pkg_layout.setSpacing(4)
                
                # Package name and version
                name_ver = QLabel(f"<b>{pkg_name}</b> {version}")
                name_ver.setStyleSheet("font-size: 12pt; color: #4CAF50;")
                pkg_layout.addWidget(name_ver)
                
                # Description
                if description:
                    desc = QLabel(description)
                    desc.setStyleSheet("font-size: 10pt; color: #aaa;")
                    desc.setWordWrap(True)
                    pkg_layout.addWidget(desc)
                
                content_layout.addWidget(pkg_frame)
        
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return w
    
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
<p style="line-height: 1.3; font-size: 10pt;">
This application uses the following open-source libraries and tools:
</p>

<h3 style="color: #667eea; margin-top: 12px; font-size: 12pt;">Core Libraries</h3>
<ul style="line-height: 1.4; font-size: 10pt;">
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

<h3 style="color: #667eea; margin-top: 12px; font-size: 12pt;">Training & Data</h3>
<ul style="line-height: 1.4; font-size: 10pt;">
<li><b>TRL</b> - Apache 2.0<br>
    <span style="color: #888;">SFTTrainer for supervised fine-tuning</span></li>
<li><b>Datasets</b> - Apache 2.0<br>
    <span style="color: #888;">Dataset loading and processing</span></li>
<li><b>PEFT</b> - Apache 2.0<br>
    <span style="color: #888;">LoRA and efficient training</span></li>
<li><b>BitsAndBytes</b> - MIT<br>
    <span style="color: #888;">4-bit/8-bit quantization</span></li>
</ul>

<h3 style="color: #667eea; margin-top: 12px; font-size: 12pt;">Acceleration</h3>
<ul style="line-height: 1.4; font-size: 10pt;">
<li><b>xFormers</b> - BSD (Meta)<br>
    <span style="color: #888;">Memory-efficient attention</span></li>
<li><b>CUDA Toolkit 12.4</b> - NVIDIA EULA<br>
    <span style="color: #888;">GPU acceleration</span></li>
<li><b>Triton</b> - MIT (OpenAI)<br>
    <span style="color: #888;">GPU programming</span></li>
</ul>

<h3 style="color: #667eea; margin-top: 12px; font-size: 12pt;">Utilities</h3>
<ul style="line-height: 1.4; font-size: 10pt;">
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

        # log list from repo root and logs directory
        self.logs_list.clear()
        # Add logs from root directory
        for p in sorted(self.root.glob("*training*.txt")) + sorted(self.root.glob("*log*.txt")):
            it = QListWidgetItem(str(p.name))
            it.setData(Qt.UserRole, str(p))
            self.logs_list.addItem(it)
        # Add logs from logs directory (app.log, auto_repair.log, etc.)
        logs_dir = self.root / "logs"
        if logs_dir.exists():
            for p in sorted(logs_dir.glob("*.log")) + sorted(logs_dir.glob("*.txt")):
                it = QListWidgetItem(f"logs/{p.name}")
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
