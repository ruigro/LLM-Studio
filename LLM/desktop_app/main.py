from __future__ import annotations

import sys
import os
import shutil
import json
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QProcess, QTimer, QThread, Signal, QProcessEnvironment, QRect, QSize, QEvent, QObject, QPoint
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QTextEdit, QPlainTextEdit,
    QSpinBox, QDoubleSpinBox, QMessageBox, QListWidget, QListWidgetItem, QSplitter, QToolBar, QScrollArea, QGridLayout, QFrame, QProgressBar, QSizePolicy, QTabBar, QStyleOptionTab, QStyle, QStackedWidget, QGroupBox, QInputDialog, QCheckBox
)
from PySide6.QtGui import QAction, QIcon, QFont, QMouseEvent, QCursor, QPixmap

# Feature flag for hybrid frame wrapper (enabled by default)
# To disable: set USE_HYBRID_FRAME=0 before running
USE_HYBRID_FRAME = os.getenv("USE_HYBRID_FRAME", "1") == "1"

from desktop_app.model_card_widget import ModelCard, DownloadedModelCard
from desktop_app.training_widgets import MetricCard
from desktop_app.splash_screen import SplashScreen
from desktop_app.pages.server_page import ServerPage
from desktop_app.pages.mcp_page import MCPPage

from system_detector import SystemDetector
from smart_installer import SmartInstaller
from setup_state import SetupStateManager
from model_integrity_checker import ModelIntegrityChecker
from core.models import (DEFAULT_BASE_MODELS, search_hf_models, download_hf_model, list_local_adapters, 
                         list_local_downloads, get_app_root, detect_model_capabilities, get_capability_icons, get_model_size)
from core.training import TrainingConfig, default_output_dir, build_finetune_cmd
from core.inference import InferenceConfig, build_run_adapter_cmd


_APP_BUILD = datetime.fromtimestamp(Path(__file__).stat().st_mtime).strftime("%y%m%d-%H%M%S")
APP_TITLE = f"🤖 LLM Fine-tuning Studio [{_APP_BUILD}]"


class InstallerThread(QThread):
    """Thread for running smart installer without freezing UI"""
    log_output = Signal(str)
    finished_signal = Signal(bool)  # True if successful
    
    def __init__(self, install_type: str):  # "pytorch", "dependencies", or "all"
        super().__init__()
        self.install_type = install_type
    
    def run(self):
        try:
            # For repair, use InstallerV2 for targeted repair
            if self.install_type == "repair":
                self.log_output.emit("Starting targeted repair process...")
                self.log_output.emit("This will only fix broken/missing packages without reinstalling everything")
                
                # Import and run InstallerV2 repair
                from installer_v2 import InstallerV2
                from pathlib import Path
                
                # Explicitly target LLM/.venv (not bootstrap/.venv)
                llm_root = Path(__file__).parent.parent
                target_venv = llm_root / ".venv"
                
                self.log_output.emit(f"Target environment: {target_venv}")
                
                installer_v2 = InstallerV2(root_dir=llm_root)
                
                # Redirect logs
                original_log = installer_v2.log
                def gui_log(message):
                    # Ensure message is safe for the log output
                    try:
                        self.log_output.emit(message)
                    except Exception:
                        pass
                    
                    # Original log already has safety for print() now
                    original_log(message)
                
                installer_v2.log = gui_log
                
                success = installer_v2.repair()
                self.log_output.emit(f"Repair completed with result: {success}")
                self.finished_signal.emit(success)
                return
            
            # For all other operations, use SmartInstaller
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
            elif self.install_type == "uninstall":
                self.log_output.emit("Starting uninstallation process...")
                python_exe = sys.executable
                self.log_output.emit(f"Using Python: {python_exe}")
                success = installer.uninstall_all(python_executable=python_exe)
                self.log_output.emit(f"Uninstallation completed with result: {success}")
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


class PipPackageThread(QThread):
    """Thread for running a single pip install/uninstall without freezing UI."""
    log_output = Signal(str)
    finished_signal = Signal(bool)

    def __init__(self, action: str, package_spec: str, python_exe: str | None = None):
        super().__init__()
        self.action = action  # "install" | "uninstall"
        self.package_spec = package_spec
        self.python_exe = python_exe or sys.executable

    def run(self):
        import subprocess
        try:
            if self.action == "uninstall":
                cmd = [self.python_exe, "-m", "pip", "uninstall", "-y", self.package_spec]
            else:
                # Install/update a single package only; do NOT resolve deps (to avoid cascading reinstalls).
                cmd = [self.python_exe, "-m", "pip", "install", "--no-deps", "--no-cache-dir", self.package_spec]

            self.log_output.emit(f"Running: {' '.join(cmd)}")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self.log_output.emit(line)
            rc = proc.wait()
            self.finished_signal.emit(rc == 0)
        except Exception as e:
            import traceback
            self.log_output.emit(f"[ERROR] pip task failed: {e}")
            self.log_output.emit(traceback.format_exc())
            self.finished_signal.emit(False)


class SystemDetectThread(QThread):
    """Thread for running system detection without freezing UI."""
    detected = Signal(dict)        # system_info dict
    error = Signal(str)            # error string

    def run(self):
        try:
            detector = SystemDetector()
            python_info = detector.detect_python()
            cuda_info = detector.detect_cuda()
            pytorch_info = detector.detect_pytorch()
            hardware_info = detector.detect_hardware()
            
            system_info = {
                "python": python_info,
                "cuda": cuda_info,
                "pytorch": pytorch_info,
                "hardware": hardware_info,
            }
            self.detected.emit(system_info)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


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


class PackageCard(QFrame):
    """Modern, polished card widget for displaying a package with version info and status."""
    clicked = Signal(str)  # Emits package name when clicked
    
    def __init__(self, pkg_name, required_version, installed_version, is_installed, status, status_color, status_text, description, parent=None):
        super().__init__(parent)
        self.pkg_name = pkg_name
        self.required_version = required_version
        self.installed_version = installed_version
        self.is_installed = is_installed
        self.status_text = status_text
        self.status_color = status_color
        self.description = description
        
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumHeight(140)
        self.setMaximumHeight(250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setCursor(Qt.PointingHandCursor)
        self._setup_style(False)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(18, 16, 18, 16)
        
        # Header: Name + Status badge
        header = QHBoxLayout()
        header.setSpacing(12)
        
        name_label = QLabel(pkg_name)
        name_label.setWordWrap(True)
        name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        name_label.setStyleSheet("""
            font-size: 15pt; 
            font-weight: 700; 
            color: #ffffff; 
            background: transparent;
            letter-spacing: 0.5px;
        """)
        header.addWidget(name_label, 1)
        header.addStretch()
        
        # Status badge with rounded background
        status_badge = QLabel(status_text)
        status_badge.setStyleSheet(f"""
            background: {status_color}33;
            color: {status_color};
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 9pt;
            font-weight: 600;
            border: 1px solid {status_color}66;
        """)
        header.addWidget(status_badge)
        layout.addLayout(header)
        
        # Version info - vertical layout for proper wrapping
        if installed_version:
            inst_label = QLabel(f"Installed: {installed_version}")
            inst_label.setWordWrap(True)
            inst_label.setStyleSheet(f"color: {status_color}; font-size: 10pt; font-weight: 500; background: transparent;")
            layout.addWidget(inst_label)
        else:
            inst_label = QLabel("Not installed")
            inst_label.setWordWrap(True)
            inst_label.setStyleSheet("color: #888; font-size: 10pt; background: transparent;")
            layout.addWidget(inst_label)
        
        if required_version:
            req_label = QLabel(f"Required: {required_version}")
            req_label.setWordWrap(True)
            req_label.setStyleSheet("color: #aaa; font-size: 9pt; background: transparent;")
            layout.addWidget(req_label)
        
        # Description
        if description:
            desc = QLabel(description)
            desc.setWordWrap(True)
            desc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            desc.setStyleSheet("""
                color: #b0b0b0; 
                font-size: 10pt; 
                background: transparent;
                line-height: 1.4;
                margin-top: 4px;
            """)
            layout.addWidget(desc)
        
        layout.addStretch()

    def _setup_style(self, selected=False):
        if selected:
            self.setStyleSheet(f"""
                PackageCard {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 rgba(102, 126, 234, 0.15), stop:1 rgba(118, 75, 162, 0.15));
                    border: 2px solid {self.status_color};
                    border-radius: 12px;
                    padding: 0px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                PackageCard {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 rgba(40, 40, 55, 0.6), stop:1 rgba(25, 25, 35, 0.6));
                    border: 1px solid rgba(102, 126, 234, 0.2);
                    border-radius: 12px;
                    padding: 0px;
                }}
                PackageCard:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 rgba(60, 60, 80, 0.8), stop:1 rgba(40, 40, 55, 0.8));
                    border: 1px solid rgba(102, 126, 234, 0.5);
                    transform: translateY(-2px);
                }}
            """)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.pkg_name)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool):
        self._setup_style(selected)


# Dark theme stylesheet with gradient accents
DARK_THEME = """
QMainWindow {
    background-color: #0e1117;
    color: #fafafa;
    border: 2px solid #667eea;
    border-radius: 12px;
}
QWidget {
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
    text-decoration: none;
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
QMainWindow {
    background-color: #ffffff;
    color: #262730;
    border: 2px solid #667eea;
    border-radius: 12px;
}
QWidget {
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
    text-decoration: none;
}
QToolBar {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
    border: none;
    spacing: 10px;
}
"""

# Color theme definitions - each color has dark and light variants
COLOR_THEMES = {
    "purple": {
        "dark": {
            "primary": "#667eea",
            "secondary": "#764ba2",
            "accent": "#7c8ef5"
        },
        "light": {
            "primary": "#667eea",
            "secondary": "#764ba2",
            "accent": "#7c8ef5"
        }
    },
    "yellow": {
        "dark": {
            "primary": "#fbbf24",
            "secondary": "#f59e0b",
            "accent": "#fcd34d"
        },
        "light": {
            "primary": "#d97706",
            "secondary": "#f59e0b",
            "accent": "#fbbf24"
        }
    },
    "red": {
        "dark": {
            "primary": "#ef4444",
            "secondary": "#dc2626",
            "accent": "#f87171"
        },
        "light": {
            "primary": "#dc2626",
            "secondary": "#b91c1c",
            "accent": "#ef4444"
        }
    },
    "navy": {
        "dark": {
            "primary": "#3b82f6",
            "secondary": "#1e40af",
            "accent": "#60a5fa"
        },
        "light": {
            "primary": "#1e40af",
            "secondary": "#1e3a8a",
            "accent": "#3b82f6"
        }
    },
    "green": {
        "dark": {
            "primary": "#10b981",
            "secondary": "#059669",
            "accent": "#34d399"
        },
        "light": {
            "primary": "#059669",
            "secondary": "#047857",
            "accent": "#10b981"
        }
    },
    "gray": {
        "dark": {
            "primary": "#6b7280",
            "secondary": "#4b5563",
            "accent": "#9ca3af"
        },
        "light": {
            "primary": "#4b5563",
            "secondary": "#374151",
            "accent": "#6b7280"
        }
    }
}


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to rgba string"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

def get_theme_stylesheet(dark_mode: bool, color_theme: str) -> str:
    """Generate theme stylesheet with specified color theme"""
    colors = COLOR_THEMES[color_theme]["dark" if dark_mode else "light"]
    primary = colors["primary"]
    secondary = colors["secondary"]
    accent = colors["accent"]
    
    # Convert to rgba with 40% transparency (60% opacity) for transparent elements
    primary_rgba = hex_to_rgba(primary, 0.6)
    secondary_rgba = hex_to_rgba(secondary, 0.6)
    accent_rgba = hex_to_rgba(accent, 0.65)
    
    if dark_mode:
        return f"""
QMainWindow {{
    background-color: #0e1117;
    color: #fafafa;
    border: 2px solid {primary};
    border-radius: 12px;
}}
/* Remove all underlines from HTML content in QLabel - comprehensive rules */
QLabel {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel * {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
/* Remove underlines from all HTML elements in QLabel */
QLabel a {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel b {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel strong {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel h1, QLabel h2, QLabel h3, QLabel h4, QLabel h5, QLabel h6 {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel p {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel span {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel li {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel div {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
/* Window border container with transparency */
QFrame#windowBorderContainer {{
    background: transparent;
    border: 2px solid {primary};
    border-radius: 12px;
}}
QWidget {{
    background-color: rgba(14, 17, 23, 0.2);
    color: #fafafa;
}}
/* Header styling with transparency */
/* 
 * IMPORTANT: NEVER ADD border-bottom TO QFrame ELEMENTS!
 * 
 * This was removed to fix double-line issues at the bottom of pages and the app window.
 * Adding border-bottom to QFrame creates unwanted bottom borders on all pages.
 * If you need a border, use a specific object name or class, not the global QFrame rule.
 */
QFrame {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(60, 60, 80, 0.2), stop:0.5 rgba(40, 40, 60, 0.2), stop:1 rgba(60, 60, 80, 0.2));
    border: none;
    border-bottom: none;
    border-radius: 0px;
}}
/* Navigation buttons with transparency */
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {primary_rgba}, stop:1 {secondary_rgba});
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 14pt;
    font-weight: bold;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QPushButton:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {secondary_rgba}, stop:1 {primary_rgba});
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent_rgba}, stop:1 {primary_rgba});
}}
QTabWidget::pane {{
    border: 1px solid #262730;
    background-color: #0e1117;
}}
QTabBar::tab {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {primary}, stop:1 {secondary});
    color: white;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {secondary}, stop:1 {primary});
}}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: #262730;
    color: #fafafa;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 4px;
}}
/* Training dashboard buttons */
QPushButton#train_start, QPushButton#train_stop {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {primary_rgba}, stop:1 {secondary_rgba});
    color: white;
    border: 2px solid {primary};
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 16pt;
    font-weight: bold;
    min-height: 50px;
}}
QPushButton#train_start:hover, QPushButton#train_stop:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {secondary_rgba}, stop:1 {primary_rgba});
    border: 2px solid {accent};
}}
/* Regular buttons (not navigation) with transparency */
QPushButton[class!="nav-button"] {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {primary_rgba}, stop:1 {secondary_rgba});
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}}
QPushButton[class!="nav-button"]:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {secondary_rgba}, stop:1 {primary_rgba});
}}
QPushButton:disabled {{
    background-color: rgba(58, 58, 58, 0.2);
    color: #808080;
}}
QListWidget {{
    background-color: #262730;
    color: #fafafa;
    border: 1px solid #3a3a3a;
}}
QLabel {{
    color: #fafafa;
    text-decoration: none;
}}
QFrame {{
    border: none;
    background-color: #1a1d23;
}}
QFrame[frameShape="4"] {{
    border: none;
    border-radius: 8px;
    background-color: #1a1d23;
}}
QToolBar {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {primary}, stop:1 {secondary});
    border: none;
    spacing: 10px;
}}
"""
    else:
        return f"""
QMainWindow {{
    background-color: #ffffff;
    color: #262730;
    border: 2px solid {primary};
    border-radius: 12px;
}}
/* Remove all underlines from HTML content in QLabel - comprehensive rules */
QLabel {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel * {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
/* Remove underlines from all HTML elements in QLabel */
QLabel a {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel b {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel strong {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel h1, QLabel h2, QLabel h3, QLabel h4, QLabel h5, QLabel h6 {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel p {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel span {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel li {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
QLabel div {{
    text-decoration: none !important;
    border: none !important;
    border-bottom: none !important;
}}
/* Window border container with transparency */
QFrame#windowBorderContainer {{
    background: transparent;
    border: 2px solid {primary};
    border-radius: 12px;
}}
QWidget {{
    background-color: rgba(255, 255, 255, 0.2);
    color: #262730;
}}
/* Header styling with transparency */
/* 
 * IMPORTANT: NEVER ADD border-bottom TO QFrame ELEMENTS!
 * 
 * This was removed to fix double-line issues at the bottom of pages and the app window.
 * Adding border-bottom to QFrame creates unwanted bottom borders on all pages.
 * If you need a border, use a specific object name or class, not the global QFrame rule.
 */
QFrame {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(240, 240, 250, 0.2), stop:0.5 rgba(220, 220, 240, 0.2), stop:1 rgba(200, 200, 230, 0.2));
    border: none;
    border-bottom: none;
    border-radius: 0px;
}}
/* Navigation buttons with transparency */
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {primary_rgba}, stop:1 {secondary_rgba});
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 14pt;
    font-weight: bold;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QPushButton:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {secondary_rgba}, stop:1 {primary_rgba});
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent_rgba}, stop:1 {primary_rgba});
}}
QTabWidget::pane {{
    border: 1px solid #e0e0e0;
    background-color: #ffffff;
}}
QTabBar::tab {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {primary}, stop:1 {secondary});
    color: white;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {secondary}, stop:1 {primary});
}}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: #f5f5f5;
    color: #262730;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    padding: 4px;
}}
/* Training dashboard buttons */
QPushButton#train_start, QPushButton#train_stop {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {primary_rgba}, stop:1 {secondary_rgba});
    color: white;
    border: 2px solid {primary};
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 16pt;
    font-weight: bold;
    min-height: 50px;
}}
QPushButton#train_start:hover, QPushButton#train_stop:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {secondary_rgba}, stop:1 {primary_rgba});
    border: 2px solid {accent};
}}
/* Regular buttons (not navigation) with transparency */
QPushButton[class!="nav-button"] {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {primary_rgba}, stop:1 {secondary_rgba});
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}}
QPushButton[class!="nav-button"]:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {secondary_rgba}, stop:1 {primary_rgba});
}}
QPushButton:disabled {{
    background-color: rgba(224, 224, 224, 0.2);
    color: #a0a0a0;
}}
QListWidget {{
    background-color: #f5f5f5;
    color: #262730;
    border: 1px solid #d0d0d0;
}}
QLabel {{
    color: #262730;
    text-decoration: none;
}}
QToolBar {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {primary}, stop:1 {secondary});
    border: none;
    spacing: 10px;
}}
"""


class MainWindow(QMainWindow):
    def __init__(self, splash: SplashScreen = None) -> None:
        super().__init__()
        # Remove native Windows title bar but keep window resizable
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window | Qt.WindowMinMaxButtonsHint)
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 900)
        self.setMinimumSize(800, 600)  # Allow resizing with reasonable minimum

        self.root = get_app_root()
        self.dark_mode = True  # Start in dark mode
        self.color_theme = "purple"  # Default color theme
        
        # Store references to widgets with hardcoded colors for theme updates
        self.themed_widgets = {
            "frames": [],
            "labels": [],
            "buttons": [],
            "containers": []
        }
        
        # Window dragging and resizing support
        self.drag_position = None
        self.resize_edge = None  # Track which edge is being resized
        self.resize_start_pos = None
        self.resize_start_geometry = None
        self.edge_margin = 8  # Pixels from edge to trigger resize (increased for easier detection)
        self.cursor_override_active = False  # Track if we have an override cursor active
        self.current_cursor_shape = None  # Track current cursor shape to avoid unnecessary changes
        self._drag_disabled = False  # Flag to disable drag when wrapped in HybridFrameWindow
        
        # Model integrity checker
        if splash:
            splash.update_progress(5, "Initializing model checker", "")
        self.model_checker = ModelIntegrityChecker()

        # App shutdown coordination (stop server first, then exit)
        self._shutdown_in_progress = False
        self._shutdown_force_timer = None

        # IMPORTANT:
        # - If splash is provided (slow path): do full detection synchronously with UI updates.
        # - If splash is None (fast path): do NOT run detection here (it can take minutes and freeze at 50%).
        #   Instead, use lightweight placeholders and refresh after `_background_system_check()` completes.
        if splash:
            splash.update_progress(10, "Detecting system", "")
            splash.set_checking("Python")

            detector = SystemDetector()
            self.system_info = {}

            # Detect Python
            splash.set_checking("Python")
            python_info = detector.detect_python()
            self.system_info["python"] = python_info
            ver = python_info.get("version", "Unknown")
            splash.set_result("Python", f"{ver}", python_info.get("found", False))
            splash.update_progress(20, "Python detected", "")

            # Detect CUDA/GPUs
            splash.set_checking("CUDA & GPUs")
            cuda_info = detector.detect_cuda()
            self.system_info["cuda"] = cuda_info
            if cuda_info.get("found"):
                gpus = cuda_info.get("gpus", [])
                for idx, gpu in enumerate(gpus):
                    gpu_name = gpu.get("name", f"GPU {idx}")
                    gpu_mem = gpu.get("memory", "N/A")
                    splash.set_result(f"GPU {idx}", f"{gpu_name} ({gpu_mem})", True)
                splash.update_progress(40, f"CUDA v{cuda_info.get('version', 'N/A')}", "")
            else:
                splash.set_result("CUDA & GPUs", "Not detected (CPU mode)", False)
                splash.update_progress(40, "No CUDA", "")

            # Detect PyTorch
            splash.set_checking("PyTorch")
            pytorch_info = detector.detect_pytorch()
            self.system_info["pytorch"] = pytorch_info
            if pytorch_info.get("found"):
                ver = pytorch_info.get("version", "Unknown")
                cuda = " + CUDA" if pytorch_info.get("cuda_available") else " (CPU only)"
                splash.set_result("PyTorch", f"v{ver}{cuda}", True)
            else:
                splash.set_result("PyTorch", "Not installed", False)
            splash.update_progress(60, "PyTorch detected", "")

            # Detect Hardware
            splash.set_checking("Hardware (CPU/RAM)")
            hardware_info = detector.detect_hardware()
            self.system_info["hardware"] = hardware_info
            cpu_name = hardware_info.get("cpu_name", "Unknown CPU")
            if len(cpu_name) > 40:
                cpu_name = cpu_name[:37] + "..."
            cpu_cores = hardware_info.get("cpu", {}).get("cores", "?")
            ram = hardware_info.get("ram_gb", 0)
            splash.set_result("CPU", f"{cpu_name} ({cpu_cores} cores)", True)
            splash.set_result("RAM", f"{ram:.1f} GB", True)
            splash.update_progress(80, "Hardware detected", "")

            # Store SmartInstaller instance for reuse
            splash.set_checking("Installer state")
            self.installer = SmartInstaller()
            splash.set_result("Installer", "Ready", True)
            splash.update_progress(90, "Building UI", "")
        else:
            # Fast init: placeholders only (real detection happens in `_background_system_check`)
            pyver = ".".join(map(str, sys.version_info[:3]))
            self.system_info = {
                "python": {"found": True, "version": pyver},
                "cuda": {"found": False, "gpus": [], "driver_version": None, "version": None},
                "pytorch": {"found": False, "cuda_available": False, "version": None, "cuda_version": None},
                "hardware": {"cpu_name": "Unknown", "cpu": {"cores": 0}, "ram_gb": 0},
            }
            self.installer = SmartInstaller()

        # Background detection state (fast path uses this to populate real values)
        self._bg_detect_thread: SystemDetectThread | None = None
        self._bg_detect_started: bool = False
        self._home_tab_needs_update: bool = False
        self._rebuilding_home_tab: bool = False

        # Create a beautiful unified header
        header_widget = QFrame()
        header_widget.setFrameShape(QFrame.StyledPanel)
        header_widget.setMinimumHeight(80)
        # Header styling will be applied by theme system
        # Store header reference for dragging
        self.header_widget = header_widget
        # Make header draggable
        header_widget.mousePressEvent = lambda e: self._header_mouse_press(e)
        header_widget.mouseMoveEvent = lambda e: self._header_mouse_move(e)
        # Install event filter on header to catch top edge resize events
        header_widget.installEventFilter(self)
        
        header_layout = QHBoxLayout(header_widget)
        # Increased right margin to 50px to account for frame extension (75px) + padding
        header_layout.setContentsMargins(20, 10, 50, 10)
        header_layout.setSpacing(6)
        
        # Left: Theme controls container
        theme_container = QWidget()
        theme_container.setStyleSheet("background: transparent;")
        theme_layout = QHBoxLayout(theme_container)
        theme_layout.setContentsMargins(0, 0, 0, 0)
        theme_layout.setSpacing(0)  # No spacing - attach theme button to color selector
        
        # Dark/Light mode toggle button (same size as color selector, icon on top, text below)
        theme_btn_container = QWidget()
        theme_btn_container.setFixedSize(70, 50)
        theme_btn_container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(60, 60, 80, 0.8), stop:1 rgba(40, 40, 60, 0.8));
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 6px;
            }
        """)
        theme_btn_layout = QVBoxLayout(theme_btn_container)
        theme_btn_layout.setContentsMargins(4, 4, 4, 4)
        theme_btn_layout.setSpacing(2)
        theme_btn_layout.setAlignment(Qt.AlignCenter)
        
        # Icon label (bigger)
        theme_icon = QLabel("🌙")
        theme_icon.setAlignment(Qt.AlignCenter)
        theme_icon.setStyleSheet("""
            QLabel {
                background: transparent;
                color: white;
                font-size: 20pt;
                border: none;
            }
        """)
        theme_btn_layout.addWidget(theme_icon)
        
        # Text label
        theme_text = QLabel("Dark")
        theme_text.setAlignment(Qt.AlignCenter)
        theme_text.setStyleSheet("""
            QLabel {
                background: transparent;
                color: white;
                font-size: 9pt;
                font-weight: bold;
                border: none;
            }
        """)
        theme_btn_layout.addWidget(theme_text)
        
        # Make the container clickable
        theme_btn_container.mousePressEvent = lambda e: self._toggle_theme()
        theme_btn_container.setCursor(QCursor(Qt.PointingHandCursor))
        
        # Store references for updates
        self.theme_btn_container = theme_btn_container
        self.theme_icon = theme_icon
        self.theme_text = theme_text
        theme_layout.addWidget(theme_btn_container)
        
        # Color theme selector (2 rows x 3 columns)
        color_selector = QWidget()
        color_selector.setFixedSize(70, 50)
        color_selector.setStyleSheet("background: rgba(40, 40, 60, 0.4); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 6px;")
        color_layout = QVBoxLayout(color_selector)
        color_layout.setContentsMargins(4, 4, 4, 4)
        color_layout.setSpacing(3)
        
        # Row 1: Purple, Yellow, Red
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(3)
        
        # Row 2: Navy, Green, Gray
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(3)
        
        color_themes = ["purple", "yellow", "red", "navy", "green", "gray"]
        color_colors = {
            "purple": "#667eea",
            "yellow": "#fbbf24",
            "red": "#ef4444",
            "navy": "#3b82f6",
            "green": "#10b981",
            "gray": "#6b7280"
        }
        
        self.color_buttons = {}
        for i, theme_name in enumerate(color_themes):
            color_btn = QPushButton()
            color_btn.setFixedSize(18, 18)
            color_btn.setCheckable(True)
            color_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color_colors[theme_name]};
                    border: 2px solid rgba(255, 255, 255, 0.4);
                    border-radius: 3px;
                }}
                QPushButton:checked {{
                    border: 3px solid white;
                }}
                QPushButton:hover {{
                    border: 2px solid rgba(255, 255, 255, 0.8);
                }}
            """)
            color_btn.clicked.connect(lambda checked, t=theme_name: self._set_color_theme(t))
            if theme_name == "purple":
                color_btn.setChecked(True)
            self.color_buttons[theme_name] = color_btn
            
            # Add to appropriate row
            if i < 3:
                row1.addWidget(color_btn)
            else:
                row2.addWidget(color_btn)
        
        color_layout.addLayout(row1)
        color_layout.addLayout(row2)
        
        theme_layout.addWidget(color_selector)
        header_layout.addWidget(theme_container)
        
        # Center: App title (transparent background)
        title_container = QWidget()
        title_container.setStyleSheet("background: transparent;")
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(12)
        title_layout.setAlignment(Qt.AlignCenter)
        
        # Title text only (without emoji or icon)
        title_text = APP_TITLE.replace("🤖 ", "").replace("🤖", "")  # Remove robot emoji
        title_label = QLabel(title_text)
        title_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        title_label.setStyleSheet("""
            QLabel {
                background: transparent;
                color: white; 
                font-size: 24pt; 
                font-weight: bold;
                border: none;
                padding: 0px;
            }
        """)
        title_layout.addWidget(title_label)
        
        # Add stretch on both sides to center the content
        title_layout.insertStretch(0, 1)
        title_layout.addStretch(1)
        
        header_layout.addWidget(title_container, 1)
        
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
        self.header_python_label = python_label
        
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
        self.header_pytorch_label = pytorch_label
        
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
        self.header_ram_label = ram_label
        
        header_layout.addWidget(sys_info_widget)
        
        # Fullscreen button (⛶) in top right
        fullscreen_btn = QPushButton("⛶")
        fullscreen_btn.setFixedSize(30, 30)
        fullscreen_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #ffffff;
                border: none;
                font-size: 20pt;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 4px;
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.3);
            }
        """)
        fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        header_layout.addWidget(fullscreen_btn)
        self.fullscreen_btn = fullscreen_btn
        
        # Close button (X) in top right
        close_btn = QPushButton("❌")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #f44336;
                border: none;
                font-size: 16pt;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background: rgba(244, 67, 54, 0.2);
                border-radius: 4px;
            }
            QPushButton:pressed {
                background: rgba(244, 67, 54, 0.3);
            }
        """)
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)
        self.close_btn = close_btn  # Store reference for frame integration
        
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
        self.server_btn = QPushButton("🖧 Server")
        self.mcp_btn = QPushButton("🧩 MCP")
        self.requirements_btn = QPushButton("🔧")  # Tool icon only
        self.info_btn = QPushButton("ℹ️ Info")
        
        # Navigation buttons will be styled by theme system
        
        for btn in [self.home_btn, self.train_btn, self.download_btn, self.test_btn, self.logs_btn, self.server_btn, self.mcp_btn, self.requirements_btn, self.info_btn]:
            btn.setCheckable(True)
            # Navigation buttons will be styled by theme system
        
        # Special styling for requirements button (icon only, smaller)
        self.requirements_btn.setMaximumWidth(60)
        
        # Add left-side buttons
        navbar_layout.addWidget(self.home_btn)
        navbar_layout.addWidget(self.train_btn)
        navbar_layout.addWidget(self.download_btn)
        navbar_layout.addWidget(self.test_btn)
        navbar_layout.addWidget(self.server_btn)
        navbar_layout.addWidget(self.mcp_btn)
        
        # Add stretch to consume remaining space
        navbar_layout.addStretch(1)
        
        # Add Requirements, Logs and Info buttons on far right
        navbar_layout.addWidget(self.requirements_btn)
        navbar_layout.addWidget(self.logs_btn)
        navbar_layout.addWidget(self.info_btn)
        
        main_layout.addWidget(navbar)

        # Create tab widget for content (hide the tab bar since we're using custom buttons)
        tabs = QTabWidget()
        tabs.tabBar().setVisible(False)
        # IMPORTANT: store on self for other callbacks (GPU refresh, background detection, etc.)
        self.tabs = tabs
        
        tabs.addTab(self._build_home_tab(), "Home")
        tabs.addTab(self._build_train_tab(), "Train")
        tabs.addTab(self._build_models_tab(), "Download")
        tabs.addTab(self._build_test_tab(), "Test")
        tabs.addTab(self._build_logs_tab(), "Logs")
        self.server_page = ServerPage(self)
        tabs.addTab(self.server_page, "Server")
        tabs.addTab(MCPPage(self), "MCP")
        tabs.addTab(self._build_requirements_tab(), "Requirements")
        tabs.addTab(self._build_info_tab(), "Info")
        
        # Connect buttons to tab switching
        self.home_btn.clicked.connect(lambda: self._switch_tab(tabs, 0))
        self.train_btn.clicked.connect(lambda: self._switch_tab(tabs, 1))
        self.download_btn.clicked.connect(lambda: self._switch_tab(tabs, 2))
        self.test_btn.clicked.connect(lambda: self._switch_tab(tabs, 3))
        self.logs_btn.clicked.connect(lambda: self._switch_tab(tabs, 4))
        self.server_btn.clicked.connect(lambda: self._switch_tab(tabs, 5))
        self.mcp_btn.clicked.connect(lambda: self._switch_tab(tabs, 6))
        self.requirements_btn.clicked.connect(lambda: self._switch_tab(tabs, 7))
        self.info_btn.clicked.connect(lambda: self._switch_tab(tabs, 8))
        
        # Also connect to tab widget's currentChanged signal to handle programmatic changes
        tabs.currentChanged.connect(self._update_frame_corner_br)
        tabs.currentChanged.connect(self._on_tab_changed)
        
        # Set Home as default
        self.home_btn.setChecked(True)
        tabs.setCurrentIndex(0)

        main_layout.addWidget(tabs)
        
        # Wrap main widget in a bordered container to create window border effect
        border_container = QFrame()
        border_container.setFrameShape(QFrame.NoFrame)
        border_container.setObjectName("windowBorderContainer")
        # Border styling will be handled by theme system
        border_layout = QVBoxLayout(border_container)
        border_layout.setContentsMargins(2, 2, 2, 2)  # Small margin for border visibility
        border_layout.setSpacing(0)
        border_layout.addWidget(main_widget)
        
        self.setCentralWidget(border_container)
        
        # Install event filter on central widget and main widget to catch mouse events for resizing
        # This allows us to intercept mouse events at the window edges even when child widgets would normally consume them
        border_container.installEventFilter(self)
        main_widget.installEventFilter(self)

        self.train_proc: QProcess | None = None
        
        # Initialize card lists
        self.model_cards = []
        
        # Auto-run system diagnostics on startup (delayed to allow UI to render first)
        QTimer.singleShot(500, self._auto_check_system)
        self.downloaded_model_cards = []
        self.metric_cards = []
        
        self._refresh_locals()
        self._apply_theme()

    def _get_text_color(self) -> str:
        """Get appropriate text color based on theme"""
        return "#262730" if not self.dark_mode else "white"
    
    def _get_status_color(self, is_ok: bool) -> str:
        """Get appropriate status color based on theme and status"""
        if is_ok:
            return "#555555" if not self.dark_mode else "#4CAF50"  # Dark gray in light mode, green in dark mode
        else:
            return "#f44336"  # Red for errors (same in both modes)
    
    def _create_status_row(self, label: str, is_ok: bool, detail: str) -> QHBoxLayout:
        """Create a status indicator row"""
        row = QHBoxLayout()
        
        status_icon = "✅" if is_ok else "❌"
        color = self._get_status_color(is_ok)
        
        main_label = QLabel(f"{status_icon} <b>{label}</b>")
        main_label.setObjectName(f"homeStatusRow_{label.replace(' ', '_')}")
        main_label.setStyleSheet(f"background: transparent; color: {color};")
        self.themed_widgets["labels"].append(main_label)
        row.addWidget(main_label)
        
        row.addStretch(1)
        
        detail_label = QLabel(detail)
        detail_label.setObjectName(f"homeStatusDetail_{label.replace(' ', '_')}")
        detail_color = "#666666" if not self.dark_mode else "#888888"
        detail_label.setStyleSheet(f"background: transparent; color: {detail_color}; font-size: 10pt;")
        self.themed_widgets["labels"].append(detail_label)
        row.addWidget(detail_label)
        
        return row
    
    def _get_target_venv_python(self) -> str:
        """Get the target venv Python executable path"""
        import sys
        from pathlib import Path
        
        # Try to find LLM/.venv Python
        llm_venv = self.root / ".venv"
        if sys.platform == "win32":
            venv_python = llm_venv / "Scripts" / "python.exe"
        else:
            venv_python = llm_venv / "bin" / "python"
        
        if venv_python.exists():
            return str(venv_python)
        
        # Fallback to current Python
        return sys.executable
    
    def _create_status_widget(self, label: str, is_ok: bool, detail: str) -> QWidget:
        """Create a status indicator widget"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        status_icon = "✅" if is_ok else "❌"
        color = self._get_status_color(is_ok)
        
        main_label = QLabel(f"{status_icon} <b>{label}</b>")
        main_label.setObjectName(f"homeStatusWidget_{label.replace(' ', '_')}")
        main_label.setStyleSheet(f"background: transparent; color: {color};")
        self.themed_widgets["labels"].append(main_label)
        layout.addWidget(main_label)
        
        layout.addStretch(1)
        
        detail_label = QLabel(detail)
        detail_label.setObjectName(f"homeStatusWidgetDetail_{label.replace(' ', '_')}")
        detail_color = "#666666" if not self.dark_mode else "#888888"
        detail_label.setStyleSheet(f"background: transparent; color: {detail_color}; font-size: 10pt;")
        self.themed_widgets["labels"].append(detail_label)
        layout.addWidget(detail_label)
        
        return widget
    
    def _auto_check_system(self):
        """Auto-run system diagnostics on startup with automatic retry for CUDA"""
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
            print(f"CUDA: {cuda_info.get('version', 'N/A')} (Driver: {cuda_info.get('driver_version', 'N/A')})")
            gpus = cuda_info.get('gpus', [])
            for idx, gpu in enumerate(gpus):
                print(f"  GPU {idx}: {gpu.get('name', 'Unknown')} ({gpu.get('memory', 'Unknown')})")
        else:
            print("CUDA: Not found")
            # If CUDA detection failed, retry once after 2 seconds
            print("CUDA detection failed on first attempt, retrying in 2 seconds...")
            QTimer.singleShot(2000, self._retry_cuda_detection)
        
        print("=========================\n")

    def _sort_gpus_by_memory(self, gpus: list) -> list:
        """Sort GPU list by reported memory (descending) while preserving original index."""
        import re

        enriched = []
        for idx, gpu in enumerate(gpus or []):
            gpu_copy = dict(gpu) if isinstance(gpu, dict) else {"name": str(gpu)}
            gpu_copy["_orig_index"] = idx
            enriched.append(gpu_copy)

        def mem_gb(mem_str: str) -> float:
            match = re.search(r"(\d+(?:\.\d+)?)", str(mem_str or ""))
            if not match:
                return 0.0
            return float(match.group(1)) / 1024.0 if "MiB" in str(mem_str) else float(match.group(1))

        return sorted(enriched, key=lambda g: mem_gb(g.get("memory")), reverse=True)
    
    def _retry_cuda_detection(self):
        """Retry CUDA detection after initial failure"""
        print("=== CUDA Detection Retry ===")
        detector = SystemDetector()
        cuda_result = detector.detect_cuda()
        
        # Update system_info with new CUDA detection
        self.system_info["cuda"] = cuda_result
        
        if cuda_result.get('found'):
            print(f"✅ CUDA detected on retry: {cuda_result.get('version', 'N/A')}")
            gpus = cuda_result.get('gpus', [])
            if gpus:
                for idx, gpu in enumerate(gpus):
                    print(f"  GPU {idx}: {gpu.get('name', 'Unknown')} ({gpu.get('memory', 'Unknown')})")
            # Rebuild Home tab to show updated status
            current_index = self.tabs.currentIndex()
            self.tabs.removeTab(0)
            self.tabs.insertTab(0, self._build_home_tab(), "🏠 Home")
            if current_index == 0:
                self.tabs.setCurrentIndex(0)
        else:
            print("❌ CUDA still not detected after retry")
            if cuda_result.get('error'):
                print(f"   Error: {cuda_result['error']}")
            if cuda_result.get('warnings'):
                for warning in cuda_result['warnings']:
                    print(f"   Warning: {warning}")
        
        print("===========================\n")
    
    def _refresh_gpu_detection(self):
        """Refresh GPU/CUDA detection and update the Home tab"""
        # Get the sender button if available
        sender = self.sender()
        if isinstance(sender, QPushButton):
            sender.setEnabled(False)
            sender.setText("🔄 Refreshing...")
            sender.repaint()  # Force UI update
        
        # Re-detect system info
        try:
            self.system_info = SystemDetector().detect_all()
            cuda_info = self.system_info.get("cuda", {})
            
            # Show success message
            if cuda_info.get("found"):
                gpus = cuda_info.get("gpus", [])
                if gpus:
                    msg = f"✅ {len(gpus)} GPU(s) detected"
                else:
                    msg = "⚠️ CUDA toolkit found but no GPUs detected"
            else:
                msg = "❌ CUDA not detected"
            
            # Rebuild Home tab to show updated info
            current_index = self.tabs.currentIndex()
            self.tabs.removeTab(0)
            self.tabs.insertTab(0, self._build_home_tab(), "🏠 Home")
            if current_index == 0:
                self.tabs.setCurrentIndex(0)
            
            # Show brief status message
            print(f"GPU Detection Refresh: {msg}")
            
        except Exception as e:
            print(f"Error refreshing GPU detection: {e}")
            # Button will be recreated when tab is rebuilt, so no need to restore state
    
    def _get_resize_edge(self, pos) -> str:
        """Determine which edge the mouse is near for resizing"""
        rect = self.rect()
        x, y = pos.x(), pos.y()
        w, h = rect.width(), rect.height()
        
        # Check corners first (for diagonal resize)
        if x <= self.edge_margin and y <= self.edge_margin:
            return "top-left"
        elif x >= w - self.edge_margin and y <= self.edge_margin:
            return "top-right"
        elif x <= self.edge_margin and y >= h - self.edge_margin:
            return "bottom-left"
        elif x >= w - self.edge_margin and y >= h - self.edge_margin:
            return "bottom-right"
        # Check edges
        elif x <= self.edge_margin:
            return "left"
        elif x >= w - self.edge_margin:
            return "right"
        elif y <= self.edge_margin:
            return "top"
        elif y >= h - self.edge_margin:
            return "bottom"
        return None
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for window resizing"""
        if event is None or self._drag_disabled:
            if self._drag_disabled:
                return super().mousePressEvent(event)
            return
        
        try:
            if event.button() == Qt.LeftButton:
                try:
                    pos = event.position().toPoint()
                    global_pos = event.globalPosition().toPoint()
                except (RuntimeError, AttributeError) as e:
                    print(f"Error getting positions in mousePressEvent: {e}")
                    super().mousePressEvent(event)
                    return
                
                edge = self._get_resize_edge(pos)
                if edge:
                    try:
                        geom = self.geometry()
                        if not geom.isValid():
                            super().mousePressEvent(event)
                            return
                        
                        self.resize_edge = edge
                        self.resize_start_pos = global_pos
                        self.resize_start_geometry = geom
                        event.accept()
                        return
                    except (RuntimeError, AttributeError) as e:
                        print(f"Error setting resize state in mousePressEvent: {e}")
                        super().mousePressEvent(event)
                        return
        except Exception as e:
            print(f"Unexpected error in mousePressEvent: {e}")
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for window resizing and cursor changes"""
        if event is None or self._drag_disabled:
            if self._drag_disabled:
                return super().mouseMoveEvent(event)
            return
        
        try:
            try:
                pos = event.position().toPoint()
                global_pos = event.globalPosition().toPoint()
            except (RuntimeError, AttributeError) as e:
                print(f"Error getting positions in mouseMoveEvent: {e}")
                super().mouseMoveEvent(event)
                return
            
            # If resizing, handle resize using global coordinates
            if self.resize_edge and event.buttons() == Qt.LeftButton:
                if self.resize_start_pos and self.resize_start_geometry:
                    try:
                        # Use global position delta for accurate resize
                        delta = global_pos - self.resize_start_pos
                        geom = QRect(self.resize_start_geometry)
                        
                        if "left" in self.resize_edge:
                            new_x = geom.x() + delta.x()
                            new_width = geom.width() - delta.x()
                            if new_width >= self.minimumWidth():
                                geom.setX(new_x)
                                geom.setWidth(new_width)
                        if "right" in self.resize_edge:
                            new_width = geom.width() + delta.x()
                            if new_width >= self.minimumWidth():
                                geom.setWidth(new_width)
                        if "top" in self.resize_edge:
                            new_y = geom.y() + delta.y()
                            new_height = geom.height() - delta.y()
                            if new_height >= self.minimumHeight():
                                geom.setY(new_y)
                                geom.setHeight(new_height)
                        if "bottom" in self.resize_edge:
                            new_height = geom.height() + delta.y()
                            if new_height >= self.minimumHeight():
                                geom.setHeight(new_height)
                        
                        # Validate geometry before setting
                        if geom.isValid() and geom.width() >= self.minimumWidth() and geom.height() >= self.minimumHeight():
                            self.setGeometry(geom)
                            # Update start position for next move to prevent accumulation
                            self.resize_start_pos = global_pos
                            self.resize_start_geometry = geom
                            event.accept()
                            return
                    except (RuntimeError, AttributeError) as e:
                        print(f"Error during resize in mouseMoveEvent: {e}")
                        # Cancel resize on error
                        self.resize_edge = None
                        self.resize_start_pos = None
                        self.resize_start_geometry = None
            
            # Always check cursor on mouse move (even when not resizing)
            try:
                edge = self._get_resize_edge(pos)
                if edge:
                    cursor_shape = None
                    if edge in ["top-left", "bottom-right"]:
                        cursor_shape = Qt.SizeFDiagCursor
                    elif edge in ["top-right", "bottom-left"]:
                        cursor_shape = Qt.SizeBDiagCursor
                    elif edge in ["left", "right"]:
                        cursor_shape = Qt.SizeHorCursor
                    elif edge in ["top", "bottom"]:
                        cursor_shape = Qt.SizeVerCursor
                    
                    # Only change cursor if it's different from current
                    if cursor_shape and cursor_shape != self.current_cursor_shape:
                        try:
                            if self.cursor_override_active:
                                # Change existing override instead of restoring and setting new one
                                QApplication.changeOverrideCursor(QCursor(cursor_shape))
                            else:
                                # Set new override
                                QApplication.setOverrideCursor(QCursor(cursor_shape))
                            # Also set on widget as fallback
                            self.setCursor(QCursor(cursor_shape))
                            self.cursor_override_active = True
                            self.current_cursor_shape = cursor_shape
                        except Exception as e:
                            print(f"Error setting cursor in mouseMoveEvent: {e}")
                else:
                    # Not on edge - restore normal cursor
                    if self.cursor_override_active:
                        try:
                            QApplication.restoreOverrideCursor()
                            self.cursor_override_active = False
                            self.current_cursor_shape = None
                        except Exception as e:
                            print(f"Error restoring cursor in mouseMoveEvent: {e}")
                            self.cursor_override_active = False
                            self.current_cursor_shape = None
                    # Also restore widget cursor
                    try:
                        self.unsetCursor()
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error checking cursor in mouseMoveEvent: {e}")
        except Exception as e:
            print(f"Unexpected error in mouseMoveEvent: {e}")
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release to stop resizing"""
        if event is None or self._drag_disabled:
            if self._drag_disabled:
                return super().mouseReleaseEvent(event)
            return
        
        try:
            if event.button() == Qt.LeftButton:
                self.resize_edge = None
                self.resize_start_pos = None
                self.resize_start_geometry = None
                if self.cursor_override_active:
                    try:
                        QApplication.restoreOverrideCursor()
                        self.cursor_override_active = False
                        self.current_cursor_shape = None
                    except Exception as e:
                        print(f"Error restoring cursor in mouseReleaseEvent: {e}")
                        self.cursor_override_active = False
                        self.current_cursor_shape = None
        except Exception as e:
            print(f"Unexpected error in mouseReleaseEvent: {e}")
        
        super().mouseReleaseEvent(event)

    def closeEvent(self, event):
        """
        When user clicks the window X:
        1) trigger server stop
        2) wait 2.5s
        3) close
        """
        # If we're already in the final close pass, allow it.
        if getattr(self, "_shutdown_in_progress", False):
            event.accept()
            return

        print("[DEBUG] closeEvent triggered - shutting down server...")
        server_page = getattr(self, "server_page", None)
        if server_page:
            try:
                # Trigger the same logic as the "Stop Server" button
                server_page.request_stop()
                print("[DEBUG] Server stop requested")
            except Exception as e:
                print(f"[DEBUG] Error requesting server stop: {e}")

        # Mark that we are shutting down
        self._shutdown_in_progress = True
        
        # Prevent immediate close
        event.ignore()
        
        # Wait exactly 2.5s then force close
        print("[DEBUG] Waiting 2.5s before app exit...")
        QTimer.singleShot(2500, self.close)
    
    def eventFilter(self, obj, event) -> bool:
        """Event filter to catch mouse events for window resizing from child widgets"""
        if self._drag_disabled:
            return super().eventFilter(obj, event)
        
        if not isinstance(event, QMouseEvent):
            return super().eventFilter(obj, event)
        
        try:
            # Always use global position for accurate calculations
            try:
                global_pos = event.globalPosition().toPoint()
                # Convert to window coordinates for edge detection
                window_pos = self.mapFromGlobal(global_pos)
            except (RuntimeError, AttributeError) as e:
                print(f"Error getting positions in eventFilter: {e}")
                return super().eventFilter(obj, event)
            
            # Handle mouse enter to ensure cursor updates
            if event.type() == QMouseEvent.Type.Enter:
                # When mouse enters a child widget, check if we're near an edge
                if not self.resize_edge:
                    try:
                        edge = self._get_resize_edge(window_pos)
                        if edge:
                            cursor_shape = None
                            if edge in ["top-left", "bottom-right"]:
                                cursor_shape = Qt.SizeFDiagCursor
                            elif edge in ["top-right", "bottom-left"]:
                                cursor_shape = Qt.SizeBDiagCursor
                            elif edge in ["left", "right"]:
                                cursor_shape = Qt.SizeHorCursor
                            elif edge in ["top", "bottom"]:
                                cursor_shape = Qt.SizeVerCursor
                            
                            # Only change cursor if it's different from current
                            if cursor_shape and cursor_shape != self.current_cursor_shape:
                                try:
                                    if self.cursor_override_active:
                                        # Change existing override instead of restoring and setting new one
                                        QApplication.changeOverrideCursor(QCursor(cursor_shape))
                                    else:
                                        # Set new override
                                        QApplication.setOverrideCursor(QCursor(cursor_shape))
                                    # Also set on widget as fallback
                                    self.setCursor(QCursor(cursor_shape))
                                    self.cursor_override_active = True
                                    self.current_cursor_shape = cursor_shape
                                except Exception as e:
                                    print(f"Error setting cursor in eventFilter Enter: {e}")
                    except Exception as e:
                        print(f"Error checking edge in eventFilter Enter: {e}")
                return False  # Let event propagate
            
            # Handle mouse leave to restore cursor
            if event.type() == QMouseEvent.Type.Leave:
                if self.cursor_override_active and not self.resize_edge:
                    try:
                        QApplication.restoreOverrideCursor()
                        self.cursor_override_active = False
                        self.current_cursor_shape = None
                    except Exception as e:
                        print(f"Error restoring cursor in eventFilter Leave: {e}")
                        self.cursor_override_active = False
                        self.current_cursor_shape = None
                return False  # Let event propagate
            
            # Handle mouse move for cursor changes and resizing
            if event.type() == QMouseEvent.Type.MouseMove:
                # If currently resizing, handle resize using global coordinates
                if self.resize_edge and event.buttons() == Qt.LeftButton:
                    if self.resize_start_pos and self.resize_start_geometry:
                        # Use global position delta for accurate resize
                        delta = global_pos - self.resize_start_pos
                        geom = QRect(self.resize_start_geometry)
                        
                        if "left" in self.resize_edge:
                            new_x = geom.x() + delta.x()
                            new_width = geom.width() - delta.x()
                            if new_width >= self.minimumWidth():
                                geom.setX(new_x)
                                geom.setWidth(new_width)
                        if "right" in self.resize_edge:
                            new_width = geom.width() + delta.x()
                            if new_width >= self.minimumWidth():
                                geom.setWidth(new_width)
                        if "top" in self.resize_edge:
                            new_y = geom.y() + delta.y()
                            new_height = geom.height() - delta.y()
                            if new_height >= self.minimumHeight():
                                geom.setY(new_y)
                                geom.setHeight(new_height)
                        if "bottom" in self.resize_edge:
                            new_height = geom.height() + delta.y()
                            if new_height >= self.minimumHeight():
                                geom.setHeight(new_height)
                        
                        # Validate geometry before setting
                        if geom.isValid() and geom.width() >= self.minimumWidth() and geom.height() >= self.minimumHeight():
                            try:
                                self.setGeometry(geom)
                                # Update start position for next move to prevent accumulation
                                self.resize_start_pos = global_pos
                                self.resize_start_geometry = geom
                                return True  # Consume event
                            except (RuntimeError, AttributeError) as e:
                                print(f"Error setting geometry in eventFilter: {e}")
                                # Cancel resize on error
                                self.resize_edge = None
                                self.resize_start_pos = None
                                self.resize_start_geometry = None
                                return True
                
                # Always check cursor on mouse move (even when not resizing)
                try:
                    edge = self._get_resize_edge(window_pos)
                    if edge:
                        cursor_shape = None
                        if edge in ["top-left", "bottom-right"]:
                            cursor_shape = Qt.SizeFDiagCursor
                        elif edge in ["top-right", "bottom-left"]:
                            cursor_shape = Qt.SizeBDiagCursor
                        elif edge in ["left", "right"]:
                            cursor_shape = Qt.SizeHorCursor
                        elif edge in ["top", "bottom"]:
                            cursor_shape = Qt.SizeVerCursor
                        
                        # Only change cursor if it's different from current
                        if cursor_shape and cursor_shape != self.current_cursor_shape:
                            try:
                                if self.cursor_override_active:
                                    # Change existing override instead of restoring and setting new one
                                    QApplication.changeOverrideCursor(QCursor(cursor_shape))
                                else:
                                    # Set new override
                                    QApplication.setOverrideCursor(QCursor(cursor_shape))
                                # Also set on widget as fallback
                                self.setCursor(QCursor(cursor_shape))
                                self.cursor_override_active = True
                                self.current_cursor_shape = cursor_shape
                            except Exception as e:
                                print(f"Error setting cursor in eventFilter: {e}")
                        # Return False to let event propagate, but cursor is set
                    else:
                        # Restore cursor when not on edge
                        if self.cursor_override_active:
                            try:
                                QApplication.restoreOverrideCursor()
                                self.cursor_override_active = False
                                self.current_cursor_shape = None
                            except Exception as e:
                                print(f"Error restoring cursor in eventFilter: {e}")
                                self.cursor_override_active = False
                                self.current_cursor_shape = None
                        # Also restore widget cursor
                        try:
                            self.unsetCursor()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Error checking cursor in eventFilter: {e}")
            
            # Handle mouse press for resize start
            elif event.type() == QMouseEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
                try:
                    edge = self._get_resize_edge(window_pos)
                    if edge:
                        geom = self.geometry()
                        if geom.isValid():
                            self.resize_edge = edge
                            self.resize_start_pos = global_pos
                            self.resize_start_geometry = geom
                            return True  # Consume event to start resize
                except (RuntimeError, AttributeError) as e:
                    print(f"Error in eventFilter mouse press: {e}")
                    return super().eventFilter(obj, event)
            
            # Handle mouse release
            elif event.type() == QMouseEvent.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
                try:
                    if self.resize_edge:
                        self.resize_edge = None
                        self.resize_start_pos = None
                        self.resize_start_geometry = None
                        # Check current position and set cursor accordingly
                        try:
                            edge = self._get_resize_edge(window_pos)
                            if edge:
                                cursor_shape = None
                                if edge in ["top-left", "bottom-right"]:
                                    cursor_shape = Qt.SizeFDiagCursor
                                elif edge in ["top-right", "bottom-left"]:
                                    cursor_shape = Qt.SizeBDiagCursor
                                elif edge in ["left", "right"]:
                                    cursor_shape = Qt.SizeHorCursor
                                elif edge in ["top", "bottom"]:
                                    cursor_shape = Qt.SizeVerCursor
                                
                                if cursor_shape and cursor_shape != self.current_cursor_shape:
                                    try:
                                        if self.cursor_override_active:
                                            QApplication.restoreOverrideCursor()
                                        QApplication.setOverrideCursor(QCursor(cursor_shape))
                                        self.cursor_override_active = True
                                        self.current_cursor_shape = cursor_shape
                                    except Exception as e:
                                        print(f"Error setting cursor in eventFilter release: {e}")
                            else:
                                if self.cursor_override_active:
                                    try:
                                        QApplication.restoreOverrideCursor()
                                        self.cursor_override_active = False
                                        self.current_cursor_shape = None
                                    except Exception as e:
                                        print(f"Error restoring cursor in eventFilter release: {e}")
                                        self.cursor_override_active = False
                                        self.current_cursor_shape = None
                        except Exception as e:
                            print(f"Error checking edge in eventFilter release: {e}")
                        return True  # Consume event
                except Exception as e:
                    print(f"Unexpected error in eventFilter mouse release: {e}")
                    return super().eventFilter(obj, event)
        
        except Exception as e:
            print(f"Unexpected error in eventFilter: {e}")
            return super().eventFilter(obj, event)
        
        return super().eventFilter(obj, event)
    
    def _header_mouse_press(self, event: QMouseEvent) -> None:
        """Handle mouse press on header for window dragging"""
        if event is None or self.header_widget is None or self._drag_disabled:
            return
        
        try:
            if event.button() == Qt.LeftButton:
                try:
                    # Check for top edge FIRST before allowing drag (to prioritize resize)
                    # Convert header widget position to window coordinates
                    header_pos = self.header_widget.mapTo(self, event.position().toPoint())
                    edge = self._get_resize_edge(header_pos)
                    # If on top edge (or any edge), don't set drag_position - let resize handler take over
                    if not edge:
                        global_pos = event.globalPosition().toPoint()
                        frame_geom = self.frameGeometry()
                        if frame_geom.isValid():
                            self.drag_position = global_pos - frame_geom.topLeft()
                            event.accept()
                    # If on edge, don't accept event - let event filter handle resize
                except (RuntimeError, AttributeError) as e:
                    print(f"Error in _header_mouse_press: {e}")
        except Exception as e:
            print(f"Unexpected error in _header_mouse_press: {e}")
    
    def _header_mouse_move(self, event: QMouseEvent) -> None:
        """Handle mouse move on header for window dragging"""
        if event is None or self.header_widget is None or self._drag_disabled:
            return
        
        try:
            if event.buttons() == Qt.LeftButton and self.drag_position is not None:
                # Check if we're currently resizing FIRST - if so, don't handle drag
                if not self.resize_edge:
                    try:
                        # Also check if we're on an edge now (might have moved to edge)
                        header_pos = self.header_widget.mapTo(self, event.position().toPoint())
                        edge = self._get_resize_edge(header_pos)
                        if not edge:
                            global_pos = event.globalPosition().toPoint()
                            new_pos = global_pos - self.drag_position
                            # Validate position before moving (reasonable bounds)
                            if new_pos.x() >= -10000 and new_pos.y() >= -10000:
                                self.move(new_pos)
                                event.accept()
                    except (RuntimeError, AttributeError) as e:
                        print(f"Error in _header_mouse_move: {e}")
                        # Cancel drag on error
                        self.drag_position = None
                # If resizing, don't handle drag - let resize take precedence
        except Exception as e:
            print(f"Unexpected error in _header_mouse_move: {e}")
            self.drag_position = None
    
    def _toggle_fullscreen(self) -> None:
        """Toggle window between fullscreen and normal mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def _on_tab_changed(self, index: int):
        """Update UI when tabs are switched."""
        # Handle corner image update
        self._update_frame_corner_br(index)
        
        # index 0 is Home tab
        if index == 0 and self._home_tab_needs_update:
            # Home tab is now visible and needs update
            QTimer.singleShot(100, self._rebuild_home_tab)
            
        # index 7 is Requirements tab
        elif index == 7:
            self._refresh_requirements_grid()

    def _switch_tab(self, tab_widget: QTabWidget, index: int):
        """Switch to a tab and update button states"""
        tab_widget.setCurrentIndex(index)
        
        # Update button checked states
        buttons = [self.home_btn, self.train_btn, self.download_btn, self.test_btn, self.logs_btn, self.server_btn, self.mcp_btn, self.requirements_btn, self.info_btn]
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)
        
        # Update corner_br image based on current tab
        self._update_frame_corner_br(index)

    def _rebuild_home_tab(self):
        """Safely rebuild Home tab with latest detection results."""
        try:
            if not hasattr(self, "tabs") or self.tabs is None or self.tabs.count() == 0:
                return
            
            # Prevent multiple simultaneous rebuilds
            if hasattr(self, "_rebuilding_home_tab") and self._rebuilding_home_tab:
                return
            self._rebuilding_home_tab = True
            
            current_index = self.tabs.currentIndex()
            # Rebuild Home tab (index 0)
            self.tabs.removeTab(0)
            self.tabs.insertTab(0, self._build_home_tab(), "🏠 Home")
            # Restore current tab if it wasn't Home
            if current_index != 0:
                self.tabs.setCurrentIndex(current_index)
            else:
                self.tabs.setCurrentIndex(0)
            
            self._home_tab_needs_update = False
            self._rebuilding_home_tab = False
        except Exception as e:
            self._log_to_app_log(f"[ERROR] Failed to rebuild Home tab: {e}")
            if hasattr(self, "_rebuilding_home_tab"):
                self._rebuilding_home_tab = False
    
    def _update_frame_corner_br(self, tab_index: int) -> None:
        """Update the frame's corner_br image based on the current tab."""
        if not hasattr(self, '_hybrid_frame') or self._hybrid_frame is None:
            return
        if not hasattr(self, '_get_frame_asset_path'):
            return
        
        # Map tab indices to corner_br image names
        # All corner images are sized to 150px width (height adaptable) by hybrid_frame system
        tab_to_image = {
            0: "corner_br_owl_coding",      # Home
            1: "corner_br_owl_training",     # Train
            2: "corner_br_owl_models",       # Download
            3: "corner_br_owl_chat",        # Test
            4: "corner_br_owl_logs",         # Logs
            5: "corner_br_owl_server",       # Server
            6: "corner_br_owl_MCP",          # MCP
            7: "corner_br_owl_tools",        # Requirements (using tools as fallback)
            8: "corner_br_owl_info",         # Info (changed from corner_br_owl_thanks)
        }
        
        image_name = tab_to_image.get(tab_index, "corner_br")  # Default to corner_br if not found
        image_path = self._get_frame_asset_path(image_name)
        self._hybrid_frame.set_corner_br(image_path)
    
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
        """Fix Issues: Launch installer GUI for repair"""
        import subprocess
        from pathlib import Path
        
        # Find run_installer.bat (preferred) or installer_gui.py (fallback)
        app_dir = Path(__file__).parent.parent
        run_installer_bat = app_dir / "run_installer.bat"
        installer_gui = app_dir / "installer_gui.py"
        
        # Prefer run_installer.bat which ensures bootstrap is used
        if run_installer_bat.exists():
            try:
                subprocess.Popen(
                    [str(run_installer_bat)],
                    cwd=str(app_dir),
                    shell=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                # Close main app - installer will handle everything
                self.close()
                return
            except Exception as e:
                # Fall through to installer_gui.py fallback
                pass
        
        # Fallback: launch installer_gui.py directly (it has bootstrap guard)
        if installer_gui.exists():
            python_exe = sys.executable
            try:
                subprocess.Popen(
                    [python_exe, str(installer_gui)],
                    cwd=str(app_dir),
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                # Close main app - installer will handle everything
                self.close()
                return
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Launch Failed",
                    f"Failed to launch installer GUI:\n{str(e)}\n\n"
                    "Please run run_installer.bat or installer_gui.py manually."
                )
        else:
            QMessageBox.critical(
                self,
                "Installer Not Found",
                f"Installer not found.\n\n"
                "Please ensure run_installer.bat or installer_gui.py is in the LLM directory."
            )
    
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

    def _get_theme_colors(self):
        """Get current theme colors"""
        return COLOR_THEMES[self.color_theme]["dark" if self.dark_mode else "light"]
    
    def _get_frame_border_style(self, primary_color: str = None) -> str:
        """Get frame border style with current theme color"""
        if primary_color is None:
            primary_color = self._get_theme_colors()["primary"]
        return f"border: 2px solid {primary_color};"
    
    def _get_gradient_style(self, primary_color: str = None, secondary_color: str = None) -> str:
        """Get gradient style with current theme colors"""
        colors = self._get_theme_colors()
        if primary_color is None:
            primary_color = colors["primary"]
        if secondary_color is None:
            secondary_color = colors["secondary"]
        return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {primary_color}, stop:1 {secondary_color})"
    
    def _set_color_theme(self, theme_name: str) -> None:
        """Set the color theme"""
        self.color_theme = theme_name
        # Update button checked states
        for name, btn in self.color_buttons.items():
            btn.setChecked(name == theme_name)
        # Reapply theme with new color
        self._apply_theme()
    
    def _apply_theme(self) -> None:
        """Apply the current theme"""
        # Use dynamic theme with selected color
        stylesheet = get_theme_stylesheet(self.dark_mode, self.color_theme)
        self.setStyleSheet(stylesheet)
        
        # Update theme button icon and text
        if self.dark_mode:
            self.theme_icon.setText("🌙")
            self.theme_text.setText("Dark")
        else:
            self.theme_icon.setText("☀️")
            self.theme_text.setText("Light")
        
        # Update header border with theme color
        colors = self._get_theme_colors()
        primary = colors["primary"]
        secondary = colors["secondary"]
        accent = colors["accent"]
        
        if hasattr(self, 'header_widget'):
            # IMPORTANT: Never add border-bottom here - it creates double-line issues
            self.header_widget.setStyleSheet(f"""
                QFrame {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 rgba(60, 60, 80, 0.6), stop:0.5 rgba(40, 40, 60, 0.6), stop:1 rgba(60, 60, 80, 0.6));
                    border: none;
                    border-bottom: none;
                    border-radius: 0px;
                }}
            """)
        
        # Update all themed widgets
        self._update_themed_widgets(primary, secondary, accent)
        
        # Update HybridFrameWindow content if it exists (when using hybrid frame)
        # The frame itself is just decorative - only update the widget inside it
        if hasattr(self, '_hybrid_frame_content') and self._hybrid_frame_content is not None:
            # Apply theme ONLY to the central widget - frame and container don't need it
            self._hybrid_frame_content.setStyleSheet(stylesheet)
        
        # Update frame colors if frame overlay exists
        if hasattr(self, '_hybrid_frame') and self._hybrid_frame is not None:
            # Convert hex colors to QColor with alpha for frame
            from PySide6.QtGui import QColor
            frame_color = QColor(primary)
            frame_color.setAlpha(220)  # Semi-transparent
            frame_accent = QColor(accent)
            frame_accent.setAlpha(200)  # Semi-transparent
            # Background color - darker version of primary for solid fill
            bg_color = QColor(primary)
            bg_color = bg_color.darker(300)  # Much darker for background
            self._hybrid_frame.set_frame_colors(frame_color, frame_accent, bg_color)
    
    def _update_themed_widgets(self, primary: str, secondary: str, accent: str) -> None:
        """Update all stored themed widgets with current colors"""
        # Update frames
        for frame in self.themed_widgets["frames"]:
            obj_name = frame.objectName()
            if obj_name == "titleFrame":
                frame.setStyleSheet(f"""
                    QFrame#titleFrame {{
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 rgba(60, 60, 80, 0.4), stop:1 rgba(40, 40, 60, 0.4));
                        border: 2px solid {primary};
                        border-radius: 12px;
                        padding: 15px;
                    }}
                """)
            elif obj_name == "leftColumnContainer":
                frame.setStyleSheet(f"""
                    QFrame#leftColumnContainer {{
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 rgba(60, 60, 80, 0.4), stop:1 rgba(40, 40, 60, 0.4));
                        border: 2px solid {primary};
                        border-radius: 12px;
                        padding: 15px;
                    }}
                """)
            elif obj_name == "rightColumnContainer":
                frame.setStyleSheet(f"""
                    QFrame#rightColumnContainer {{
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 rgba(60, 60, 80, 0.4), stop:1 rgba(40, 40, 60, 0.4));
                        border: 2px solid {primary};
                        border-radius: 12px;
                        padding: 15px;
                    }}
                """)
        
        # Update buttons
        for btn in self.themed_widgets["buttons"]:
            obj_name = btn.objectName()
            if obj_name == "refreshGpuBtn":
                btn.setStyleSheet(f"""
                    QPushButton#refreshGpuBtn {{
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 rgba(60, 60, 80, 0.4), stop:1 rgba(40, 40, 60, 0.4));
                        border: 2px solid {primary};
                        border-radius: 12px;
                        padding: 8px 15px;
                        color: white;
                        font-size: 11pt;
                        font-weight: bold;
                    }}
                    QPushButton#refreshGpuBtn:hover {{
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 rgba(80, 80, 100, 0.5), stop:1 rgba(60, 60, 80, 0.5));
                        border: 2px solid {accent};
                    }}
                    QPushButton#refreshGpuBtn:pressed {{
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 rgba(50, 50, 70, 0.5), stop:1 rgba(30, 30, 50, 0.5));
                    }}
                """)
        
        # Update labels
        text_color = self._get_text_color()
        status_color_ok = self._get_status_color(True)
        for label in self.themed_widgets["labels"]:
            obj_name = label.objectName()
            if obj_name == "modelAHeader":
                label.setStyleSheet(f"font-size: 16pt; padding: 10px; background: {self._get_gradient_style(primary, secondary)}; color: white; border-radius: 6px;")
            elif obj_name == "modelBHeader":
                label.setStyleSheet(f"font-size: 16pt; padding: 10px; background: {self._get_gradient_style(primary, secondary)}; color: white; border-radius: 6px;")
            elif obj_name == "modelCHeader":
                label.setStyleSheet(f"font-size: 16pt; padding: 10px; background: {self._get_gradient_style(primary, secondary)}; color: white; border-radius: 6px;")
            elif obj_name == "trainModelHeader":
                label.setStyleSheet(f"font-size: 14pt; color: {primary}; border: none; padding: 0;")
            # Home tab labels
            elif obj_name == "homeWelcomeTitle":
                label.setStyleSheet(f"color: {text_color}; background: transparent; border: none; padding: 0; font-size: 24pt; font-weight: bold; text-decoration: none;")
            elif obj_name == "homeFeaturesHeader":
                label.setStyleSheet(f"background: transparent; color: {text_color}; font-size: 18pt; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name == "homeFeaturesText":
                label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name == "homeGuideHeader":
                label.setStyleSheet(f"background: transparent; color: {text_color}; font-size: 18pt; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name == "homeGuideText":
                label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name == "homeSysStatusHeader":
                label.setStyleSheet(f"background: transparent; color: {text_color}; font-size: 18pt; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name == "homeGpuStatus":
                label.setStyleSheet(f"background: transparent; color: {status_color_ok}; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name and obj_name.startswith("homeGpuLabel"):
                label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name and obj_name.startswith("homeGpuMemLabel"):
                label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name == "homeCpuLabel":
                label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none;")
            elif obj_name == "homeCpuNameLabel":
                label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name == "homeCpuSpecsLabel":
                label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name == "homeStatusLabel":
                label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name == "homeStatusVal":
                # Update the HTML color in the label text
                current_text = label.text()
                if "Ready" in current_text:
                    label.setText(f"<span style='font-size: 16pt; font-weight: bold; color: {status_color_ok}; text-decoration: none; border: none; border-bottom: none;'>Ready</span>")
                label.setStyleSheet("background: transparent; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name == "homeRequirementsHeader":
                label.setStyleSheet(f"background: transparent; color: {text_color}; font-size: 18pt; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
            elif obj_name and obj_name.startswith("homeStatusRow_"):
                # Status row labels (software requirements)
                # Extract is_ok from the label text (✅ or ❌)
                current_text = label.text()
                is_ok = "✅" in current_text
                status_color = self._get_status_color(is_ok)
                label.setStyleSheet(f"background: transparent; color: {status_color};")
            elif obj_name and obj_name.startswith("homeStatusWidget_"):
                # Status widget labels (software requirements)
                current_text = label.text()
                is_ok = "✅" in current_text
                status_color = self._get_status_color(is_ok)
                label.setStyleSheet(f"background: transparent; color: {status_color};")
            elif obj_name and obj_name.startswith("homeStatusDetail_") or obj_name and obj_name.startswith("homeStatusWidgetDetail_"):
                # Detail labels in status rows/widgets
                detail_color = "#666666" if not self.dark_mode else "#888888"
                label.setStyleSheet(f"background: transparent; color: {detail_color}; font-size: 10pt;")
        
        # Update chat display theme
        if hasattr(self, 'chat_display'):
            self.chat_display.set_theme(self.dark_mode)
        
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
        
        # Welcome title in a styled container
        title_frame = QFrame()
        title_frame.setFrameShape(QFrame.StyledPanel)
        title_frame.setObjectName("titleFrame")
        colors = self._get_theme_colors()
        title_frame.setStyleSheet(f"""
            QFrame#titleFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(60, 60, 80, 0.4), stop:1 rgba(40, 40, 60, 0.4));
                {self._get_frame_border_style(colors["primary"])}
                border-radius: 12px;
                padding: 15px;
            }}
        """)
        self.themed_widgets["frames"].append(title_frame)
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(0, 12, 0, 12)
        title = QLabel("Welcome to LLM Fine-tuning Studio")
        title.setObjectName("homeWelcomeTitle")
        title.setAlignment(Qt.AlignCenter)
        text_color = self._get_text_color()
        title.setStyleSheet(f"color: {text_color}; background: transparent; border: none; padding: 0; font-size: 24pt; font-weight: bold; text-decoration: none;")
        self.themed_widgets["labels"].append(title)
        title_layout.addWidget(title)
        layout.addWidget(title_frame)
        
        # Create 2-column layout with FIXED 40/60 ratio (not resizable)
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(20)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        
        # LEFT COLUMN: Features + Quick Start Guide (40% width)
        left_container = QFrame()
        left_container.setFrameShape(QFrame.StyledPanel)
        left_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_container.setObjectName("leftColumnContainer")  # Give it a unique name
        colors = self._get_theme_colors()
        left_container.setStyleSheet(f"""
            QFrame#leftColumnContainer {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(60, 60, 80, 0.4), stop:1 rgba(40, 40, 60, 0.4));
                {self._get_frame_border_style(colors["primary"])}
                border-radius: 12px;
                padding: 15px;
            }}
        """)
        self.themed_widgets["frames"].append(left_container)
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        # Features section
        text_color = self._get_text_color()
        features_header = QLabel("🚀 Features")
        features_header.setObjectName("homeFeaturesHeader")
        features_header.setStyleSheet(f"background: transparent; color: {text_color}; font-size: 18pt; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
        font = features_header.font()
        font.setUnderline(False)
        features_header.setFont(font)
        self.themed_widgets["labels"].append(features_header)
        left_layout.addWidget(features_header)
        features_text = QLabel("""
<p style="text-decoration: none; border: none; border-bottom: none;"><span style="text-decoration: none; border: none; border-bottom: none;">This application provides a beautiful, user-friendly interface to:</span></p>
<ul style="line-height: 1.8; text-decoration: none; border: none; border-bottom: none;">
<li style="text-decoration: none; border: none; border-bottom: none;"><span style="font-weight: bold; text-decoration: none; border: none; border-bottom: none;">🎯 Train Models:</span> <span style="text-decoration: none; border: none; border-bottom: none;">Select from popular pre-trained models and fine-tune them with your data</span></li>
<li style="text-decoration: none; border: none; border-bottom: none;"><span style="font-weight: bold; text-decoration: none; border: none; border-bottom: none;">📥 Upload Datasets:</span> <span style="text-decoration: none; border: none; border-bottom: none;">Easy drag-and-drop for JSONL format datasets</span></li>
<li style="text-decoration: none; border: none; border-bottom: none;"><span style="font-weight: bold; text-decoration: none; border: none; border-bottom: none;">🧪 Test Models:</span> <span style="text-decoration: none; border: none; border-bottom: none;">Interactive chat interface to test your fine-tuned models</span></li>
<li style="text-decoration: none; border: none; border-bottom: none;"><span style="font-weight: bold; text-decoration: none; border: none; border-bottom: none;">✅ Validate Performance:</span> <span style="text-decoration: none; border: none; border-bottom: none;">Run validation tests and view detailed results</span></li>
<li style="text-decoration: none; border: none; border-bottom: none;"><span style="font-weight: bold; text-decoration: none; border: none; border-bottom: none;">📊 Track History:</span> <span style="text-decoration: none; border: none; border-bottom: none;">View all your trained models and training logs</span></li>
</ul>
        """)
        features_text.setObjectName("homeFeaturesText")
        features_text.setWordWrap(True)
        features_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        features_text.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
        font = features_text.font()
        font.setUnderline(False)
        features_text.setFont(font)
        self.themed_widgets["labels"].append(features_text)
        left_layout.addWidget(features_text)
        
        # Quick Start Guide section
        guide_header = QLabel("📋 Quick Start Guide")
        guide_header.setObjectName("homeGuideHeader")
        guide_header.setStyleSheet(f"background: transparent; color: {text_color}; font-size: 18pt; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
        font = guide_header.font()
        font.setUnderline(False)
        font = guide_header.font()
        font.setUnderline(False)
        guide_header.setFont(font)
        self.themed_widgets["labels"].append(guide_header)
        left_layout.addWidget(guide_header)
        guide_text = QLabel("""
<ol style="line-height: 2; text-decoration: none; border: none; border-bottom: none;">
<li style="text-decoration: none; border: none; border-bottom: none;"><span style="font-weight: bold; text-decoration: none; border: none; border-bottom: none;">Prepare Your Dataset:</span> <span style="text-decoration: none; border: none; border-bottom: none;">Create a JSONL file with format:</span>
<pre style="background: #2a2a2a; padding: 10px; border-radius: 5px; margin: 10px 0; text-decoration: none; border: none; border-bottom: none;">
{"instruction": "Your instruction here", "output": "Expected output here"}
</pre>
</li>
<li style="text-decoration: none; border: none; border-bottom: none;"><span style="font-weight: bold; text-decoration: none; border: none; border-bottom: none;">Go to Train Model:</span> <span style="text-decoration: none; border: none; border-bottom: none;">Select a base model and upload your dataset</span></li>
<li style="text-decoration: none; border: none; border-bottom: none;"><span style="font-weight: bold; text-decoration: none; border: none; border-bottom: none;">Configure Training:</span> <span style="text-decoration: none; border: none; border-bottom: none;">Adjust epochs, batch size, and LoRA parameters</span></li>
<li style="text-decoration: none; border: none; border-bottom: none;"><span style="font-weight: bold; text-decoration: none; border: none; border-bottom: none;">Start Training:</span> <span style="text-decoration: none; border: none; border-bottom: none;">Click the train button and monitor progress</span></li>
<li style="text-decoration: none; border: none; border-bottom: none;"><span style="font-weight: bold; text-decoration: none; border: none; border-bottom: none;">Test Your Model:</span> <span style="text-decoration: none; border: none; border-bottom: none;">Use the Test Model tab to try your fine-tuned model</span></li>
</ol>
        """)
        guide_text.setObjectName("homeGuideText")
        guide_text.setWordWrap(True)
        guide_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        guide_text.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
        font = guide_text.font()
        font.setUnderline(False)
        guide_text.setFont(font)
        self.themed_widgets["labels"].append(guide_text)
        left_layout.addWidget(guide_text)
        
        left_layout.addStretch(1)
        
        # Add left container with 40% stretch (2 parts out of 5 total = 40%)
        columns_layout.addWidget(left_container, 2)
        
        # RIGHT COLUMN: System Status + Software Requirements (60% width)
        right_container = QFrame()
        right_container.setFrameShape(QFrame.StyledPanel)
        right_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_container.setObjectName("rightColumnContainer")  # Give it a unique name
        colors = self._get_theme_colors()
        right_container.setStyleSheet(f"""
            QFrame#rightColumnContainer {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(60, 60, 80, 0.4), stop:1 rgba(40, 40, 60, 0.4));
                {self._get_frame_border_style(colors["primary"])}
                border-radius: 12px;
                padding: 15px;
            }}
        """)
        self.themed_widgets["frames"].append(right_container)
        right_layout = QVBoxLayout(right_container)
        right_layout.setSpacing(15)
        right_layout.setContentsMargins(15, 15, 15, 15)

        # System Status section
        text_color = self._get_text_color()
        sys_status_header_row = QHBoxLayout()
        sys_status_header = QLabel("📊 System Status")
        sys_status_header.setObjectName("homeSysStatusHeader")
        sys_status_header.setStyleSheet(f"background: transparent; color: {text_color}; font-size: 18pt; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
        font = sys_status_header.font()
        font.setUnderline(False)
        sys_status_header.setFont(font)
        self.themed_widgets["labels"].append(sys_status_header)
        sys_status_header_row.addWidget(sys_status_header)
        sys_status_header_row.addStretch(1)
        
        refresh_btn = QPushButton("🔄 Refresh Hardware Detection")
        refresh_btn.setObjectName("refreshGpuBtn")
        colors = self._get_theme_colors()
        refresh_btn.setStyleSheet(f"""
            QPushButton#refreshGpuBtn {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(60, 60, 80, 0.4), stop:1 rgba(40, 40, 60, 0.4));
                {self._get_frame_border_style(colors["primary"])}
                border-radius: 12px;
                padding: 8px 15px;
                color: white;
                font-size: 11pt;
                font-weight: bold;
            }}
            QPushButton#refreshGpuBtn:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(80, 80, 100, 0.5), stop:1 rgba(60, 60, 80, 0.5));
                border: 2px solid {colors["accent"]};
            }}
            QPushButton#refreshGpuBtn:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(50, 50, 70, 0.5), stop:1 rgba(30, 30, 50, 0.5));
            }}
        """)
        refresh_btn.clicked.connect(self._refresh_gpu_detection)
        self.themed_widgets["buttons"].append(refresh_btn)
        sys_status_header_row.addWidget(refresh_btn)
        right_layout.addLayout(sys_status_header_row)

        # System info cards
        sys_frame = QWidget()
        sys_frame.setStyleSheet("background: transparent; border: none;")  # Explicitly no styling
        sys_layout = QVBoxLayout(sys_frame)
        sys_layout.setSpacing(6)  # Tighter spacing
        sys_layout.setContentsMargins(10, 8, 10, 8)  # Tighter margins

        # Get real GPU info
        cuda_info = self.system_info.get("cuda", {})
        gpus = self._sort_gpus_by_memory(cuda_info.get("gpus", []))
        
        if gpus:
            status_color = self._get_status_color(True)
            gpu_status = QLabel(f"✅ <span style='font-weight: bold; text-decoration: none; border: none; border-bottom: none;'>{len(gpus)} GPU{'s' if len(gpus) > 1 else ''} detected</span>")
            gpu_status.setObjectName("homeGpuStatus")
            gpu_status.setStyleSheet(f"background: transparent; color: {status_color}; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
            font = gpu_status.font()
            font.setUnderline(False)
            gpu_status.setFont(font)
            self.themed_widgets["labels"].append(gpu_status)
            sys_layout.addWidget(gpu_status)
            
            # Display each GPU
            for idx, gpu in enumerate(gpus):
                gpu_row = QHBoxLayout()
                gpu_name = gpu.get("name", "Unknown GPU")
                gpu_mem = gpu.get("memory", "Unknown")
                gpu_display_idx = gpu.get("_orig_index", idx)
                
                gpu_label = QLabel(f"<span style='font-weight: bold; text-decoration: none; border: none; border-bottom: none;'>GPU {gpu_display_idx}:</span> <span style='text-decoration: none; border: none; border-bottom: none;'>{gpu_name}</span>")
                gpu_label.setObjectName(f"homeGpuLabel{idx}")
                gpu_label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
                font = gpu_label.font()
                font.setUnderline(False)
                gpu_label.setFont(font)
                self.themed_widgets["labels"].append(gpu_label)
                gpu_row.addWidget(gpu_label)
                gpu_row.addStretch(1)
                gpu_mem_label = QLabel(f"💾 <span style='font-weight: bold; text-decoration: none; border: none; border-bottom: none;'>{gpu_mem}</span>")
                gpu_mem_label.setObjectName(f"homeGpuMemLabel{idx}")
                gpu_mem_label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
                font = gpu_mem_label.font()
                font.setUnderline(False)
                gpu_mem_label.setFont(font)
                self.themed_widgets["labels"].append(gpu_mem_label)
                gpu_row.addWidget(gpu_mem_label)
                sys_layout.addLayout(gpu_row)
        else:
            gpu_status = QLabel("⚠️ <span style='font-weight: bold; text-decoration: none; border: none; border-bottom: none;'>No GPUs detected</span>")
            gpu_status.setObjectName("homeGpuStatus")
            gpu_status.setStyleSheet("background: transparent; color: #FF9800; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
            font = gpu_status.font()
            font.setUnderline(False)
            gpu_status.setFont(font)
            sys_layout.addWidget(gpu_status)
            cpu_label = QLabel("Training will use CPU (slower)")
            cpu_label.setObjectName("homeCpuLabel")
            cpu_label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none;")
            self.themed_widgets["labels"].append(cpu_label)
            sys_layout.addWidget(cpu_label)

        # CPU Information
        hardware_info = self.system_info.get("hardware", {})
        cpu_name = hardware_info.get("cpu_name", "Unknown CPU")
        cpu_cores = hardware_info.get("cpu", {}).get("cores", "?")
        ram_gb = hardware_info.get("ram_gb", 0)
        
        cpu_row = QHBoxLayout()
        cpu_name_label = QLabel(f"<span style='font-weight: bold; text-decoration: none; border: none; border-bottom: none;'>CPU:</span> <span style='text-decoration: none; border: none; border-bottom: none;'>{cpu_name}</span>")
        cpu_name_label.setObjectName("homeCpuNameLabel")
        cpu_name_label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
        font = cpu_name_label.font()
        font.setUnderline(False)
        cpu_name_label.setFont(font)
        self.themed_widgets["labels"].append(cpu_name_label)
        cpu_row.addWidget(cpu_name_label)
        cpu_row.addStretch(1)
        
        cpu_specs_text = f"<span style='font-weight: bold; text-decoration: none; border: none; border-bottom: none;'>{cpu_cores} cores</span>"
        if ram_gb:
            cpu_specs_text = f"<span style='font-weight: bold; text-decoration: none; border: none; border-bottom: none;'>{cpu_cores} cores</span> | <span style='font-weight: bold; text-decoration: none; border: none; border-bottom: none;'>💾 {ram_gb} GB RAM</span>"
        cpu_specs_label = QLabel(cpu_specs_text)
        cpu_specs_label.setObjectName("homeCpuSpecsLabel")
        cpu_specs_label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
        font = cpu_specs_label.font()
        font.setUnderline(False)
        cpu_specs_label.setFont(font)
        self.themed_widgets["labels"].append(cpu_specs_label)
        cpu_row.addWidget(cpu_specs_label)
        sys_layout.addLayout(cpu_row)

        # Status - compact single line
        status_row = QHBoxLayout()
        status_label = QLabel("<span style='font-weight: bold; text-decoration: none; border: none; border-bottom: none;'>Status:</span>")
        status_label.setObjectName("homeStatusLabel")
        status_label.setStyleSheet(f"background: transparent; color: {text_color}; text-decoration: none; border: none; border-bottom: none;")
        font = status_label.font()
        font.setUnderline(False)
        status_label.setFont(font)
        self.themed_widgets["labels"].append(status_label)
        status_row.addWidget(status_label)
        status_color = self._get_status_color(True)
        status_val = QLabel(f"<span style='font-size: 16pt; font-weight: bold; color: {status_color}; text-decoration: none; border: none; border-bottom: none;'>Ready</span>")
        status_val.setObjectName("homeStatusVal")
        status_val.setStyleSheet("background: transparent; text-decoration: none; border: none; border-bottom: none;")
        font = status_val.font()
        font.setUnderline(False)
        status_val.setFont(font)
        self.themed_widgets["labels"].append(status_val)
        status_row.addWidget(status_val)
        status_row.addStretch(1)
        sys_layout.addLayout(status_row)

        right_layout.addWidget(sys_frame)
        
        # Software Requirements section
        requirements_header = QLabel("⚙️ Software Requirements & Setup")
        requirements_header.setObjectName("homeRequirementsHeader")
        requirements_header.setStyleSheet(f"background: transparent; color: {text_color}; font-size: 18pt; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
        font = requirements_header.font()
        font.setUnderline(False)
        requirements_header.setFont(font)
        self.themed_widgets["labels"].append(requirements_header)
        right_layout.addWidget(requirements_header)

        setup_frame = QWidget()
        setup_frame.setStyleSheet("background: transparent; border: none;")  # Explicitly no styling
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
        
        # Dependencies status (FAST, non-blocking)
        # IMPORTANT: Do not call SmartInstaller.get_installation_checklist() here.
        # It runs subprocess import checks and can freeze the UI at the 50% splash.
        deps_row = QHBoxLayout()
        deps_ok = True
        missing_packages: list[str] = []

        try:
            import importlib.util

            def _has(pkg: str) -> bool:
                return importlib.util.find_spec(pkg) is not None

            critical = ["torch", "transformers", "tokenizers", "datasets", "accelerate", "peft", "numpy"]
            for pkg in critical:
                if not _has(pkg):
                    deps_ok = False
                    missing_packages.append(pkg)
        except Exception as e:
            print(f"Error checking dependencies (fast): {e}")
            deps_ok = False
            missing_packages.append("dependency-check-error")

        if deps_ok:
            deps_msg = "✅ Core packages found (full validation runs via Fix Issues)"
        else:
            deps_msg = f"Missing: {', '.join(missing_packages[:6])}"
            if len(missing_packages) > 6:
                deps_msg += f" (+{len(missing_packages) - 6} more)"
        
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
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 rgba(240, 147, 251, 0.6), stop:1 rgba(245, 87, 108, 0.6));
                    border: 2px solid #f093fb;
                    border-radius: 12px;
                    color: white;
                    font-size: 13pt;
                    font-weight: bold;
                    padding: 10px 18px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 rgba(245, 163, 255, 0.7), stop:1 rgba(255, 106, 126, 0.7));
                    border: 2px solid #f5a3ff;
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 rgba(208, 131, 219, 0.7), stop:1 rgba(213, 71, 92, 0.7));
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
        main_layout = QVBoxLayout(w)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 1. TOP SEARCH BAR (Beautiful and prominent)
        search_container = QFrame()
        search_container.setObjectName("searchContainer")
        search_container.setMinimumHeight(80)
        search_container.setStyleSheet("""
            QFrame#searchContainer {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                            stop:0 rgba(102, 126, 234, 0.1), stop:1 rgba(118, 75, 162, 0.1));
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 12px;
            }
        """)
        search_h_layout = QHBoxLayout(search_container)
        search_h_layout.setContentsMargins(20, 10, 20, 10)
        search_h_layout.setSpacing(15)
        
        search_icon = QLabel("🔍")
        search_icon.setStyleSheet("font-size: 20pt; background: transparent;")
        search_h_layout.addWidget(search_icon)
        
        self.hf_query = QLineEdit()
        self.hf_query.setPlaceholderText("Search thousands of models on Hugging Face (e.g., Qwen2.5, Llama-3, DeepSeek)...")
        self.hf_query.setMinimumHeight(45)
        self.hf_query.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.05);
                border: 2px solid rgba(102, 126, 234, 0.5);
                border-radius: 8px;
                padding: 5px 15px;
                font-size: 14pt;
                color: white;
            }
            QLineEdit:focus {
                border: 2px solid #764ba2;
                background-color: rgba(255, 255, 255, 0.1);
            }
        """)
        self.hf_query.returnPressed.connect(self._hf_search)
        search_h_layout.addWidget(self.hf_query, 1)
        
        self.hf_search_btn = QPushButton("Search Models")
        self.hf_search_btn.setMinimumHeight(45)
        self.hf_search_btn.setMinimumWidth(150)
        self.hf_search_btn.clicked.connect(self._hf_search)
        search_h_layout.addWidget(self.hf_search_btn)
        
        main_layout.addWidget(search_container)
        
        # 2. MAIN CONTENT TABS
        self.models_content_tabs = QTabWidget()
        self.models_content_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(102, 126, 234, 0.2);
                background: transparent;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: rgba(102, 126, 234, 0.1);
                color: #888;
                padding: 10px 30px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                font-size: 12pt;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
                color: white;
            }
        """)
        
        # 2a. BROWSE TAB (Curated + Search Results)
        browse_tab = QWidget()
        browse_layout = QVBoxLayout(browse_tab)
        browse_layout.setContentsMargins(10, 10, 10, 10)
        
        self.browse_stack = QStackedWidget()
        
        # Curated View
        curated_view = QWidget()
        curated_v_layout = QVBoxLayout(curated_view)
        
        curated_header = QHBoxLayout()
        curated_title = QLabel("📚 Recommended Models")
        curated_title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #667eea;")
        curated_header.addWidget(curated_title)
        curated_header.addStretch(1)
        curated_v_layout.addLayout(curated_header)
        
        curated_scroll = QScrollArea()
        curated_scroll.setWidgetResizable(True)
        curated_scroll.setFrameShape(QFrame.NoFrame)
        curated_scroll.setStyleSheet("background: transparent;")
        
        self.curated_container = QWidget()
        self.curated_layout = QGridLayout(self.curated_container)
        self.curated_layout.setSpacing(20)
        self.curated_layout.setContentsMargins(5, 5, 5, 5)
        
        curated_scroll.setWidget(self.curated_container)
        curated_v_layout.addWidget(curated_scroll)
        self.browse_stack.addWidget(curated_view)
        
        # Search Results View
        search_view = QWidget()
        search_v_layout = QVBoxLayout(search_view)
        
        search_results_header = QHBoxLayout()
        self.search_results_title = QLabel("🔍 Search Results")
        self.search_results_title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #667eea;")
        search_results_header.addWidget(self.search_results_title)
        
        self.back_to_curated_btn = QPushButton("← Back to Recommendations")
        self.back_to_curated_btn.setStyleSheet("background: transparent; color: #888; font-size: 11pt;")
        self.back_to_curated_btn.clicked.connect(lambda: self.browse_stack.setCurrentIndex(0))
        search_results_header.addWidget(self.back_to_curated_btn)
        search_results_header.addStretch(1)
        search_v_layout.addLayout(search_results_header)
        
        search_scroll = QScrollArea()
        search_scroll.setWidgetResizable(True)
        search_scroll.setFrameShape(QFrame.NoFrame)
        
        self.search_results_container = QWidget()
        self.search_results_layout = QGridLayout(self.search_results_container)
        self.search_results_layout.setSpacing(20)
        self.search_results_layout.setContentsMargins(5, 5, 5, 5)
        
        search_scroll.setWidget(self.search_results_container)
        search_v_layout.addWidget(search_scroll)
        self.browse_stack.addWidget(search_view)
        
        browse_layout.addWidget(self.browse_stack)
        self.models_content_tabs.addTab(browse_tab, "🚀 Browse Models")
        
        # 2b. DOWNLOADED TAB
        downloaded_tab = QWidget()
        downloaded_layout = QVBoxLayout(downloaded_tab)
        
        downloaded_header = QHBoxLayout()
        downloaded_title = QLabel("📥 My Local Models")
        downloaded_title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #667eea;")
        downloaded_header.addWidget(downloaded_title)
        downloaded_header.addStretch(1)
        
        refresh_btn = QPushButton("🔄 Refresh List")
        refresh_btn.clicked.connect(self._refresh_models)
        downloaded_header.addWidget(refresh_btn)
        downloaded_layout.addLayout(downloaded_header)
        
        downloaded_scroll = QScrollArea()
        downloaded_scroll.setWidgetResizable(True)
        downloaded_scroll.setFrameShape(QFrame.NoFrame)
        
        self.downloaded_container = QWidget()
        self.downloaded_layout = QGridLayout(self.downloaded_container) # Use grid for consistency
        self.downloaded_layout.setSpacing(20)
        self.downloaded_layout.setContentsMargins(5, 5, 5, 5)
        
        downloaded_scroll.setWidget(self.downloaded_container)
        downloaded_layout.addWidget(downloaded_scroll)
        
        self.models_content_tabs.addTab(downloaded_tab, "💾 Downloaded")
        
        main_layout.addWidget(self.models_content_tabs)
        
        # 3. BOTTOM STATUS & CONFIG
        bottom_row = QHBoxLayout()
        
        # Download path config
        path_frame = QFrame()
        path_frame.setStyleSheet("background: rgba(0,0,0,0.2); border-radius: 6px; padding: 2px;")
        path_layout = QHBoxLayout(path_frame)
        path_layout.setContentsMargins(10, 5, 10, 5)
        path_layout.addWidget(QLabel("📂 Download Path:"))
        self.hf_target_dir = QLineEdit(str(self.root / "models"))
        self.hf_target_dir.setStyleSheet("background: transparent; border: none;")
        path_layout.addWidget(self.hf_target_dir, 1)
        self.hf_browse_btn = QPushButton("Browse")
        self.hf_browse_btn.clicked.connect(self._browse_hf_target)
        path_layout.addWidget(self.hf_browse_btn)
        bottom_row.addWidget(path_frame, 1)
        
        # Status message (collapsed by default)
        self.models_status = QPlainTextEdit()
        self.models_status.setReadOnly(True)
        self.models_status.setPlaceholderText("Download status and logs will appear here...")
        self.models_status.setMaximumHeight(80)
        bottom_row.addWidget(self.models_status, 1)
        
        main_layout.addLayout(bottom_row)
        
        # Store model cards for theme updates
        self.model_cards = []
        self.downloaded_model_cards = []
        self.search_model_cards = []
        
        # Track active downloads
        self.active_downloads = {}  # model_id -> (thread, card)
        
        return w
        
        # Store model cards for theme updates
        self.model_cards = []
        self.downloaded_model_cards = []
        
        # Track active downloads
        self.active_downloads = {}  # model_id -> (thread, card)
        
        return w
    
    def _refresh_models(self) -> None:
        """Refresh all models - curated and downloaded"""
        # Clear existing cards
        while self.downloaded_layout.count():
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
        
        # Downloaded models (Grid layout)
        models_dir = self.root / "models"
        models_dirs = [models_dir]
        
        row, col = 0, 0
        max_cols = 3 # More columns for downloaded models since they are smaller
        
        for base_dir in models_dirs:
            if base_dir.exists():
                for model_dir in sorted(base_dir.iterdir()):
                    if model_dir.is_dir():
                        model_name = model_dir.name
                        
                        # Check if model is complete
                        status = self.model_checker.check_model(model_dir)
                        if not status.is_complete:
                            size = "⚠️ INCOMPLETE"
                            icons = "❌"
                        else:
                            size = get_model_size(str(model_dir))
                            capabilities = detect_model_capabilities(model_name=model_name, model_path=str(model_dir))
                            icons = get_capability_icons(capabilities)
                        
                        card = DownloadedModelCard(model_name, str(model_dir), size, icons)
                        card.set_theme(self.dark_mode)
                        card.selected.connect(self._on_model_selected)
                        card.delete_clicked.connect(self._on_delete_model)
                        self.downloaded_layout.addWidget(card, row, col)
                        self.downloaded_model_cards.append(card)
                        
                        col += 1
                        if col >= max_cols:
                            col = 0
                            row += 1
        
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
            is_downloaded = False
            for base_dir in models_dirs:
                if base_dir.exists() and (base_dir / model_slug).exists():
                    is_downloaded = True
                    break
            
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
    
    def _on_delete_model(self, model_path: str):
        """Delete a downloaded model directory forcefully"""
        path = Path(model_path)
        model_name = path.name
        
        # Check if any inference process is running
        running_procs = []
        if hasattr(self, 'test_proc_a') and self.test_proc_a is not None: running_procs.append("Model A")
        if hasattr(self, 'test_proc_b') and self.test_proc_b is not None: running_procs.append("Model B")
        if hasattr(self, 'test_proc_c') and self.test_proc_c is not None: running_procs.append("Model C")
        
        if running_procs:
            QMessageBox.warning(
                self, "Model Locked",
                f"Cannot delete model while inference is running for: {', '.join(running_procs)}.\n\n"
                "Please stop inference before deleting."
            )
            return

        # Confirmation dialog
        reply = QMessageBox.question(
            self, "Delete Model",
            f"Are you sure you want to delete the model '{model_name}'?\n\n"
            f"This will permanently remove all files from:\n{model_path}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                self._log_models(f"🗑️ Deleting model: {model_name}...")
                
                # Robust deletion with retries for Windows
                import time
                import subprocess
                
                success = False
                for attempt in range(1, 4):
                    try:
                        if sys.platform == 'win32':
                            # Force delete on Windows using cmd
                            subprocess.run(['cmd', '/c', 'rmdir', '/S', '/Q', str(path)], 
                                         capture_output=True, timeout=30)
                        else:
                            shutil.rmtree(path, ignore_errors=True)
                        
                        if not path.exists():
                            success = True
                            break
                    except Exception:
                        pass
                    
                    if attempt < 3:
                        time.sleep(1)
                
                if success:
                    self._log_models(f"✓ Model '{model_name}' deleted successfully.")
                else:
                    # Final attempt with direct shutil if path still exists
                    if path.exists():
                        shutil.rmtree(path)
                    
                    if not path.exists():
                        self._log_models(f"✓ Model '{model_name}' deleted successfully.")
                    else:
                        raise RuntimeError("Directory still exists after multiple deletion attempts.")
                
                # Refresh all local state (Download tab cards + dropdowns)
                self._refresh_locals()
                    
            except Exception as e:
                self._log_models(f"❌ Error deleting model: {str(e)}")
                QMessageBox.critical(self, "Error", f"Could not delete model directory:\n{str(e)}\n\nFiles may be locked by another process.")
    
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
        if not q:
            return
            
        # Switch to Search Results view
        self.browse_stack.setCurrentIndex(1)
        self.search_results_title.setText(f"🔍 Search Results for: '{q}'")
        
        # Clear existing results
        while self.search_results_layout.count():
            item = self.search_results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.search_model_cards.clear()
        
        self._log_models(f"🔍 Searching Hugging Face for: {q}...")
        
        try:
            hits = search_hf_models(q, limit=24) # Show up to 24 results
            
            if not hits:
                no_results = QLabel(f"No models found matching '{q}'")
                no_results.setStyleSheet("font-size: 14pt; color: #888; padding: 20px;")
                self.search_results_layout.addWidget(no_results, 0, 0)
                return
                
            row, col = 0, 0
            for h in hits:
                model_id = h.model_id
                model_name = model_id.split("/")[-1]
                
                # Check if downloaded
                model_slug = model_id.replace("/", "__")
                models_dir = self.root / "models"
                is_downloaded = (models_dir / model_slug).exists()
                
                capabilities = detect_model_capabilities(model_id=model_id)
                icons = get_capability_icons(capabilities)
                
                # Format stats
                dl_text = f"{h.downloads:,}" if h.downloads else "0"
                likes_text = f"{h.likes:,}" if h.likes else "0"
                
                # Create card
                card = ModelCard(
                    model_name, 
                    model_id, 
                    "", # No description for search results yet
                    "Unknown size", 
                    icons, 
                    is_downloaded, 
                    downloads=dl_text, 
                    likes=likes_text
                )
                card.set_theme(self.dark_mode)
                card.download_clicked.connect(self._download_curated_model)
                self.search_results_layout.addWidget(card, row, col)
                self.search_model_cards.append(card)
                
                col += 1
                if col >= 2: # 2 columns for search results
                    col = 0
                    row += 1
                    
            self._log_models(f"✓ Found {len(hits)} models for: {q}")
            
        except Exception as e:
            self._log_models(f"❌ HF search failed: {e}")
            error_label = QLabel(f"Error searching models: {str(e)}")
            error_label.setStyleSheet("color: #f44336; font-size: 12pt;")
            self.search_results_layout.addWidget(error_label, 0, 0)

    def _hf_download_selected(self) -> None:
        """Deprecated: use individual card download buttons"""
        pass

    def _log_models(self, msg: str) -> None:
        self.models_status.appendPlainText(msg)

    # ---------------- Train tab ----------------
    def _build_train_tab(self) -> QWidget:
        w = QWidget()
        main_layout = QHBoxLayout(w)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Use QHBoxLayout with fixed stretch factors for 75/25 ratio (no splitter)
        # This guarantees the ratio and prevents manual resizing
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(20)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        
        # LEFT COLUMN: Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        
        # TOP ROW: Model and Dataset in 2 columns
        top_row_header = QLabel("🎯 Model & Dataset Configuration")
        top_row_header.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none;")
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
        model_header.setObjectName("trainModelHeader")
        colors = self._get_theme_colors()
        model_header.setStyleSheet(f"font-size: 14pt; color: {colors['primary']}; border: none; padding: 0;")
        self.themed_widgets["labels"].append(model_header)
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
        params_label = QLabel("⚙️ Training Parameters")
        params_label.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none;")
        left_layout.addWidget(params_label)
        
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
        gpu_select_label = QLabel("💻 Select GPU(s) for Training")
        gpu_select_label.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none;")
        left_layout.addWidget(gpu_select_label)
        
        gpu_frame = QFrame()
        gpu_frame.setFrameShape(QFrame.StyledPanel)
        gpu_layout = QVBoxLayout(gpu_frame)
        
        # GPU status using REAL system detection
        cuda_info = self.system_info.get("cuda", {})
        gpus = self._sort_gpus_by_memory(cuda_info.get("gpus", []))
        
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
        
        self.gpu_index_map = []
        if gpus:
            for idx, gpu in enumerate(gpus):
                gpu_name = gpu.get("name", f"GPU {idx}")
                orig_idx = gpu.get("_orig_index", idx)
                self.gpu_select.addItem(f"GPU {orig_idx}: {gpu_name}")
                self.gpu_index_map.append(orig_idx)
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
        self.train_start.setObjectName("train_start")
        self.train_start.setMinimumHeight(50)
        self.train_start.clicked.connect(lambda: (self._start_training(), self._switch_to_dashboard()))
        # Training button styling will be handled by theme system
        start_btn_layout.addWidget(self.train_start)
        
        self.train_stop = QPushButton("⏹ Stop")
        self.train_stop.setObjectName("train_stop")
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

        # Add to layout with fixed 75/25 ratio using stretch factors
        # Stretch factor 3:1 = 75%:25% ratio (guaranteed, no manual resizing possible)
        columns_layout.addWidget(left_scroll, 3)  # Left = 75% (3 parts)
        columns_layout.addWidget(self.train_right_stack, 1)  # Right = 25% (1 part)
        
        # Create a container widget for the columns layout
        columns_widget = QWidget()
        columns_widget.setLayout(columns_layout)
        
        main_layout.addWidget(columns_widget)
        
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
        
        self.loss_chart_label = QLabel("")
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
        header = QLabel("🔍 Dataset Viewer")
        header.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none;")
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
        # Batch size spinbox is always hidden (auto mode is default and recommended)
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

        # Initialize training output buffer
        self._train_output_buffer = ""
        self._train_line_count = 0
        
        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        
        # Set UTF-8 encoding for Windows to handle emojis in transformers/unsloth output
        # CRITICAL: Force unbuffered output for real-time GUI updates
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONUNBUFFERED", "1")  # Force unbuffered output (even stronger than -u)
        env.insert("PYTHONLEGACYWINDOWSSTDIO", "0")  # Use new Windows stdio for better real-time output
        
        # Set GPU selection from Train tab dropdown
        if hasattr(self, 'gpu_select') and self.gpu_select.isEnabled():
            selected_display_idx = self.gpu_select.currentIndex()
            # Map displayed index to original CUDA index
            real_cuda_idx = None
            if hasattr(self, 'gpu_index_map') and self.gpu_index_map and selected_display_idx < len(self.gpu_index_map):
                real_cuda_idx = self.gpu_index_map[selected_display_idx]
            else:
                real_cuda_idx = selected_display_idx
            env.insert("CUDA_VISIBLE_DEVICES", str(real_cuda_idx))
            env.insert("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            self.train_log.appendPlainText(f"[INFO] Using GPU {real_cuda_idx}: {self.gpu_select.currentText()}")
        
        proc.setProcessEnvironment(env)
        
        proc.readyReadStandardOutput.connect(lambda: self._append_training_output(proc))
        proc.errorOccurred.connect(lambda err: self._on_training_error(proc, err))
        proc.finished.connect(lambda code, status: self._train_finished(code, status))

        self.train_log.clear()
        self.train_log.appendPlainText("=== Starting Training ===")
        self.train_log.appendPlainText(f"Command: {' '.join(cmd)}")
        self.train_log.appendPlainText("=" * 50)
        
        # Enable/disable buttons
        self.train_start.setEnabled(False)
        self.train_stop.setEnabled(True)
        
        # Start process
        proc.start()
        
        if not proc.waitForStarted(5000):
            self.train_log.appendPlainText("\n[ERROR] Failed to start training process!")
            self.train_start.setEnabled(True)
            self.train_stop.setEnabled(False)
            return
            
        self.train_proc = proc
        self.train_log.appendPlainText("[INFO] Training process started successfully")
        
        # Start background timer for GPU stats during training
        if not hasattr(self, '_train_stats_timer'):
            self._train_stats_timer = QTimer(self)
            self._train_stats_timer.timeout.connect(self._update_training_stats_periodic)
        
        self._train_stats_timer.start(2000) # Update every 2 seconds

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
        
        # Stop stats timer
        if hasattr(self, '_train_stats_timer'):
            self._train_stats_timer.stop()
            
        self.train_log.appendPlainText("\n[INFO] Terminating training process...")
        self.train_proc.terminate()
        if not self.train_proc.waitForFinished(5000):
            self.train_proc.kill()

    def _train_finished(self, exit_code=0, exit_status=QProcess.NormalExit) -> None:
        # Stop stats timer
        if hasattr(self, '_train_stats_timer'):
            self._train_stats_timer.stop()
            
        # Flush any remaining output in the buffer
        if hasattr(self, '_train_output_buffer') and self._train_output_buffer:
            remaining = self._train_output_buffer.strip()
            if remaining:
                # Process the last segment as metrics/logs
                self._parse_single_line_metrics(remaining)
                self._filter_and_append_to_train_log(remaining)
            self._train_output_buffer = ""
            
        status_str = "NormalExit" if exit_status == QProcess.NormalExit else "CrashExit"
        self.train_log.appendPlainText(f"\n[INFO] Training process finished. exit_code={exit_code}, status={status_str}")
        self.train_proc = None
        self.train_start.setEnabled(True)
        self.train_stop.setEnabled(False)
        self._refresh_locals()

    # ---------------- Test tab ----------------
    def _build_test_tab(self) -> QWidget:
        w = QWidget()
        main_layout = QHBoxLayout(w)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # LEFT COLUMN (3/4 width) - Main test interface
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        # Store reference to layout for chat display replacement
        self.test_left_layout = left_layout

        # Title and model count selector
        title_row = QHBoxLayout()
        test_title = QLabel("🧪 Test Models - Side-by-Side Chat")
        test_title.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none;")
        title_row.addWidget(test_title)
        title_row.addStretch(1)
        
        # Model count selector (two round checkboxes)
        title_row.addWidget(QLabel("Number of models:"))
        
        # Checkbox for 2 models
        self.test_model_count_2 = QCheckBox("2")
        self.test_model_count_2.setChecked(False)  # Default to 3 models
        self.test_model_count_2.setTristate(False)
        self.test_model_count_2.toggled.connect(self._on_model_count_2_toggled)
        title_row.addWidget(self.test_model_count_2)
        
        # Checkbox for 3 models
        self.test_model_count_3 = QCheckBox("3")
        self.test_model_count_3.setChecked(True)  # Default to 3 models
        self.test_model_count_3.setTristate(False)
        self.test_model_count_3.toggled.connect(self._on_model_count_3_toggled)
        title_row.addWidget(self.test_model_count_3)
        left_layout.addLayout(title_row)
        
        # GPU selection for inference
        gpu_frame = QGroupBox("⚙️ Hardware Settings")
        gpu_layout = QVBoxLayout(gpu_frame)
        
        cuda_info = self.system_info.get("cuda", {})
        gpus = self._sort_gpus_by_memory(cuda_info.get("gpus", []))
        
        # GPU selection dropdown for test tab
        gpu_row = QHBoxLayout()
        gpu_row.addWidget(QLabel("GPU for Inference:"))
        self.test_gpu_select = QComboBox()
        self.test_gpu_index_map = []
        
        if gpus:
            for idx, gpu in enumerate(gpus):
                gpu_name = gpu.get("name", f"GPU {idx}")
                vram = gpu.get("memory", "N/A")
                orig_idx = gpu.get("_orig_index", idx)
                self.test_gpu_select.addItem(f"GPU {orig_idx}: {gpu_name} ({vram})")
                self.test_gpu_index_map.append(orig_idx)
            info_text = f"✅ {len(gpus)} GPU(s) detected - select one for inference"
        else:
            self.test_gpu_select.addItem("No GPUs available - CPU mode")
            self.test_gpu_select.setEnabled(False)
            info_text = "⚠️ No GPUs detected (CPU mode)"
        
        gpu_row.addWidget(self.test_gpu_select, 1)
        gpu_layout.addLayout(gpu_row)
        
        self.test_gpu_info = QLabel(info_text)
        self.test_gpu_info.setStyleSheet("color: #888; padding: 5px; font-size: 10pt;")
        self.test_gpu_info.setWordWrap(True)
        gpu_layout.addWidget(self.test_gpu_info)
        
        left_layout.addWidget(gpu_frame)

        # Side-by-side model comparison (TOP - Chat)
        # Headers and model selectors (outside scroll)
        headers_layout = QHBoxLayout()
        headers_layout.setContentsMargins(0, 0, 0, 0)  # No margins for alignment
        headers_layout.setSpacing(6)  # 6 pixels between model headers
        
        # MODEL A Header and selector
        model_a_header_widget = QWidget()
        model_a_header_widget.setContentsMargins(0, 0, 0, 0)  # No margins for alignment
        model_a_header_layout = QVBoxLayout(model_a_header_widget)
        model_a_header_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        model_a_header_layout.setSpacing(6)  # 6 pixels between header and selector
        header_a = QLabel("🔵 <b>Model A</b>")
        header_a.setObjectName("modelAHeader")
        colors = self._get_theme_colors()
        header_a.setStyleSheet(f"font-size: 16pt; padding: 10px; background: {self._get_gradient_style(colors['primary'], colors['secondary'])}; color: white; border-radius: 6px;")
        self.themed_widgets["labels"].append(header_a)
        model_a_header_layout.addWidget(header_a)
        self.test_model_a = QComboBox()
        self.test_model_a.setEditable(True)
        model_a_header_layout.addWidget(self.test_model_a)
        headers_layout.addWidget(model_a_header_widget, 1)
        
        # MODEL B Header and selector
        model_b_header_widget = QWidget()
        model_b_header_widget.setContentsMargins(0, 0, 0, 0)  # No margins for alignment
        model_b_header_layout = QVBoxLayout(model_b_header_widget)
        model_b_header_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        model_b_header_layout.setSpacing(6)  # 6 pixels between header and selector
        header_b = QLabel("🟢 <b>Model B</b>")
        header_b.setObjectName("modelBHeader")
        header_b.setStyleSheet(f"font-size: 16pt; padding: 10px; background: {self._get_gradient_style(colors['primary'], colors['secondary'])}; color: white; border-radius: 6px;")
        self.themed_widgets["labels"].append(header_b)
        model_b_header_layout.addWidget(header_b)
        self.test_model_b = QComboBox()
        self.test_model_b.setEditable(True)
        model_b_header_layout.addWidget(self.test_model_b)
        headers_layout.addWidget(model_b_header_widget, 1)
        
        # MODEL C Header and selector (initially hidden)
        model_c_header_widget = QWidget()
        model_c_header_widget.setContentsMargins(0, 0, 0, 0)  # No margins for alignment
        model_c_header_layout = QVBoxLayout(model_c_header_widget)
        model_c_header_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        model_c_header_layout.setSpacing(6)  # 6 pixels between header and selector
        header_c = QLabel("🟣 <b>Model C</b>")
        header_c.setObjectName("modelCHeader")
        header_c.setStyleSheet(f"font-size: 16pt; padding: 10px; background: {self._get_gradient_style(colors['primary'], colors['secondary'])}; color: white; border-radius: 6px;")
        self.themed_widgets["labels"].append(header_c)
        model_c_header_layout.addWidget(header_c)
        self.test_model_c = QComboBox()
        self.test_model_c.setEditable(True)
        model_c_header_layout.addWidget(self.test_model_c)
        headers_layout.addWidget(model_c_header_widget, 1)
        model_c_header_widget.setVisible(True)  # Visible by default (3 models)
        self.test_model_c_header_widget = model_c_header_widget
        
        left_layout.addLayout(headers_layout)
        
        # SYNCHRONIZED CHAT DISPLAY
        from desktop_app.synchronized_chat_display import SynchronizedChatDisplay
        self.chat_display = SynchronizedChatDisplay(num_models=3)  # Start with 3 models
        left_layout.addWidget(self.chat_display, 1)

        # Shared prompt input area (BOTTOM)
        prompt_layout = QVBoxLayout()
        prompt_layout.addWidget(QLabel("<b>💬 Type your message:</b>"))
        
        self.test_prompt = QTextEdit()
        self.test_prompt.setPlaceholderText("Type your message here...")
        self.test_prompt.setMinimumHeight(120)
        self.test_prompt.setMaximumHeight(120)
        self.test_prompt.textChanged.connect(self._update_token_count)
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
        left_layout.addLayout(prompt_layout)

        # RIGHT COLUMN (1/4 width) - Instruction adjustment tools
        right_widget = QWidget()
        # Set minimum width to ensure content is visible, but allow expansion
        right_widget.setMinimumWidth(360)  # Minimum to show content
        right_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # Model A/B selector buttons at top
        model_selector_layout = QHBoxLayout()
        model_selector_layout.setSpacing(5)
        self.test_model_a_btn = QPushButton("🔵 A")
        self.test_model_a_btn.setCheckable(True)
        self.test_model_a_btn.setChecked(True)
        self.test_model_a_btn.clicked.connect(lambda: self._switch_model_settings(0))
        self.test_model_b_btn = QPushButton("🟢 B")
        self.test_model_b_btn.setCheckable(True)
        self.test_model_b_btn.clicked.connect(lambda: self._switch_model_settings(1))
        self.test_model_c_btn = QPushButton("🟣 C")
        self.test_model_c_btn.setCheckable(True)
        self.test_model_c_btn.clicked.connect(lambda: self._switch_model_settings(2))
        self.test_model_c_btn.setVisible(True)  # Visible by default (3 models)
        model_selector_layout.addWidget(self.test_model_a_btn)
        model_selector_layout.addWidget(self.test_model_b_btn)
        model_selector_layout.addWidget(self.test_model_c_btn)
        right_layout.addLayout(model_selector_layout)

        # Stacked widget for Model A, Model B, and Model C settings
        self.test_model_settings_stack = QStackedWidget()
        
        # Helper function to create model settings page
        def create_model_settings_page(model_name: str) -> QWidget:
            page = QWidget()
            # Set object name first, then apply background color based on model (60% transparency)
            page.setObjectName("modelSettingsPage")
            if model_name == "A":
                # Blue for Model A: rgba(0, 100, 200, 0.6)
                page.setStyleSheet("QWidget#modelSettingsPage { background-color: rgba(0, 100, 200, 0.6); }")
            elif model_name == "B":
                # Green for Model B: rgba(0, 200, 100, 0.6)
                page.setStyleSheet("QWidget#modelSettingsPage { background-color: rgba(0, 200, 100, 0.6); }")
            else:
                # Purple for Model C: rgba(155, 89, 182, 0.6)
                page.setStyleSheet("QWidget#modelSettingsPage { background-color: rgba(155, 89, 182, 0.6); }")
            
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            # Ensure scroll area can expand properly
            scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)
            scroll_layout.setContentsMargins(10, 10, 10, 10)
            scroll_layout.setSpacing(12)

            # Instruction Templates - MOVED TO TOP
            template_group = QGroupBox("📋 Instruction Templates")
            template_layout = QVBoxLayout(template_group)
            
            # Template selection row
            template_row = QHBoxLayout()
            template_select = QComboBox()
            template_select.addItems([
                "None",
                "Alpaca",
                "Vicuna",
                "ChatML",
                "Llama-2",
                "Custom"
            ])
            # Load saved custom instructions into this dropdown
            self._load_saved_instructions_into_combo(template_select)
            template_select.currentTextChanged.connect(lambda t: self._apply_instruction_template(t, system_prompt))
            template_row.addWidget(template_select, 1)
            
            # Save button
            save_btn = QPushButton("💾 Save")
            save_btn.setToolTip("Save current system prompt as a custom instruction")
            save_btn.clicked.connect(lambda: self._save_custom_instruction(system_prompt, template_select))
            template_row.addWidget(save_btn)
            template_layout.addLayout(template_row)
            
            scroll_layout.addWidget(template_group)

            # System Prompt
            system_group = QGroupBox("📝 System Prompt")
            system_layout = QVBoxLayout(system_group)
            system_prompt = QTextEdit()
            system_prompt.setPlaceholderText("Enter system instructions...")
            system_prompt.setMinimumHeight(200)
            system_prompt.setMaximumHeight(300)
            system_layout.addWidget(system_prompt)
            scroll_layout.addWidget(system_group)

            # Generation Parameters - VERTICAL LAYOUT (1 per row)
            params_group = QGroupBox("⚙️ Generation Parameters")
            params_layout = QVBoxLayout(params_group)
            params_layout.setSpacing(5)

            # Temperature
            temp_layout = QHBoxLayout()
            temp_layout.addStretch(1)
            temp_layout.addWidget(QLabel("Temperature:"))
            temp_label = QLabel("0.7")
            temp_label.setMinimumWidth(40)
            temp_layout.addWidget(temp_label)
            temperature = QDoubleSpinBox()
            temperature.setRange(0.0, 2.0)
            temperature.setSingleStep(0.1)
            temperature.setValue(0.7)
            temperature.setDecimals(1)
            temperature.setMinimumWidth(80)
            temperature.valueChanged.connect(lambda v: temp_label.setText(f"{v:.1f}"))
            temp_layout.addWidget(temperature)
            params_layout.addLayout(temp_layout)

            # Max Tokens
            max_tokens_layout = QHBoxLayout()
            max_tokens_layout.addStretch(1)
            max_tokens_layout.addWidget(QLabel("Max Tokens:"))
            def format_max_tokens(value):
                """Format max tokens value in short form if >= 1000"""
                if value < 1000:
                    return str(value)
                elif value < 1000000:
                    # Format as K (thousands)
                    if value % 1000 == 0:
                        return f"{value // 1000}K"
                    else:
                        return f"{value / 1000:.1f}K".rstrip('0').rstrip('.')
                else:
                    # Format as M (millions)
                    if value % 1000000 == 0:
                        return f"{value // 1000000}M"
                    else:
                        return f"{value / 1000000:.1f}M".rstrip('0').rstrip('.')
            
            max_tokens_label = QLabel(format_max_tokens(512))
            max_tokens_label.setMinimumWidth(40)
            max_tokens_layout.addWidget(max_tokens_label)
            max_tokens = QSpinBox()
            max_tokens.setRange(1, 999999999)  # No practical limit
            max_tokens.setSingleStep(32)
            max_tokens.setValue(512)
            max_tokens.setMinimumWidth(80)
            max_tokens.valueChanged.connect(lambda v: max_tokens_label.setText(format_max_tokens(v)))
            max_tokens_layout.addWidget(max_tokens)
            params_layout.addLayout(max_tokens_layout)

            # Top-p
            top_p_layout = QHBoxLayout()
            top_p_layout.addStretch(1)
            top_p_layout.addWidget(QLabel("Top-p:"))
            top_p_label = QLabel("0.9")
            top_p_label.setMinimumWidth(40)
            top_p_layout.addWidget(top_p_label)
            top_p = QDoubleSpinBox()
            top_p.setRange(0.0, 1.0)
            top_p.setSingleStep(0.05)
            top_p.setValue(0.9)
            top_p.setDecimals(2)
            top_p.setMinimumWidth(80)
            top_p.valueChanged.connect(lambda v: top_p_label.setText(f"{v:.2f}"))
            top_p_layout.addWidget(top_p)
            params_layout.addLayout(top_p_layout)

            # Repetition Penalty
            rep_pen_layout = QHBoxLayout()
            rep_pen_layout.addStretch(1)
            rep_pen_layout.addWidget(QLabel("Repetition Penalty:"))
            rep_pen_label = QLabel("1.0")
            rep_pen_label.setMinimumWidth(40)
            rep_pen_layout.addWidget(rep_pen_label)
            repetition_penalty = QDoubleSpinBox()
            repetition_penalty.setRange(0.0, 2.0)
            repetition_penalty.setSingleStep(0.1)
            repetition_penalty.setValue(1.0)
            repetition_penalty.setDecimals(1)
            repetition_penalty.setMinimumWidth(80)
            repetition_penalty.valueChanged.connect(lambda v: rep_pen_label.setText(f"{v:.1f}"))
            rep_pen_layout.addWidget(repetition_penalty)
            params_layout.addLayout(rep_pen_layout)

            scroll_layout.addWidget(params_group)

            # Token Count
            token_count = QLabel("Tokens: 0")
            token_count.setStyleSheet("color: #888; font-size: 10pt;")
            scroll_layout.addWidget(token_count)

            scroll_layout.addStretch(1)
            scroll.setWidget(scroll_content)
            
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.addWidget(scroll)
            
            # Store references for access
            setattr(page, 'system_prompt', system_prompt)
            setattr(page, 'temperature', temperature)
            setattr(page, 'max_tokens', max_tokens)
            setattr(page, 'top_p', top_p)
            setattr(page, 'repetition_penalty', repetition_penalty)
            setattr(page, 'template_select', template_select)
            setattr(page, 'token_count', token_count)
            
            return page

        # Create Model A, Model B, and Model C settings pages
        model_a_page = create_model_settings_page("A")
        model_b_page = create_model_settings_page("B")
        model_c_page = create_model_settings_page("C")
        
        # Connect system prompt signals after pages are created
        model_a_page.system_prompt.textChanged.connect(lambda: self._update_token_count_for_page(model_a_page))
        model_b_page.system_prompt.textChanged.connect(lambda: self._update_token_count_for_page(model_b_page))
        model_c_page.system_prompt.textChanged.connect(lambda: self._update_token_count_for_page(model_c_page))
        
        self.test_model_settings_stack.addWidget(model_a_page)
        self.test_model_settings_stack.addWidget(model_b_page)
        self.test_model_settings_stack.addWidget(model_c_page)
        
        # Store references for easy access
        self.test_model_a_settings = model_a_page
        self.test_model_b_settings = model_b_page
        self.test_model_c_settings = model_c_page
        
        right_layout.addWidget(self.test_model_settings_stack)

        # Add columns to main layout
        # Use stretch factors to maintain ratio: left takes most space, right gets what it needs
        # Stretch 0 for right means it only takes its preferred size, not extra space
        main_layout.addWidget(left_widget, 1)  # Takes remaining space after right widget
        main_layout.addWidget(right_widget, 0)  # Takes only its preferred size, no stretch

        # Store for theme updates
        self.chat_display_widget = self.chat_display
        
        # Initialize process and buffer variables
        self.test_proc_a = None
        self.test_proc_b = None
        self.inference_buffer_a = ""
        self.inference_buffer_b = ""
        self.inference_buffer_c = ""

        # Initialize default values
        self._update_token_count()

        return w
    
    def _run_side_by_side_test(self) -> None:
        """Run inference on both models simultaneously"""
        user_prompt = self.test_prompt.toPlainText().strip()
        if not user_prompt:
            QMessageBox.warning(self, "Test", "Please enter a prompt.")
            return
        
        model_a_text = self.test_model_a.currentText().strip()
        model_b_text = self.test_model_b.currentText().strip()
        model_c_text = self.test_model_c.currentText().strip() if hasattr(self, 'test_model_c') else ""
        
        # Check if at least one model is selected
        has_model = (
            (not model_a_text.startswith("(No models") and model_a_text) or
            (not model_b_text.startswith("(No models") and model_b_text) or
            (hasattr(self, 'test_model_c') and not model_c_text.startswith("(No models") and model_c_text)
        )
        
        if not has_model:
            QMessageBox.warning(self, "Test", "Please download and select at least one model from the Download tab.")
            return
        
        # Get full paths (only for valid model selections)
        model_a_path = None
        model_b_path = None
        model_c_path = None
        
        if not model_a_text.startswith("(No models") and model_a_text:
            idx = self.test_model_a.currentIndex()
            path_data = self.test_model_a.itemData(idx)
            if path_data:
                model_a_path = str(path_data)  # Ensure it's a string
        
        if not model_b_text.startswith("(No models") and model_b_text:
            idx = self.test_model_b.currentIndex()
            path_data = self.test_model_b.itemData(idx)
            if path_data:
                model_b_path = str(path_data)  # Ensure it's a string
        
        if hasattr(self, 'test_model_c') and not model_c_text.startswith("(No models") and model_c_text:
            idx = self.test_model_c.currentIndex()
            path_data = self.test_model_c.itemData(idx)
            if path_data:
                model_c_path = str(path_data)  # Ensure it's a string
        
        # Get system prompts for each model
        system_prompt_a = ""
        if model_a_path and hasattr(self.test_model_a_settings, 'system_prompt'):
            system_prompt_a = self.test_model_a_settings.system_prompt.toPlainText().strip()
        
        system_prompt_b = ""
        if model_b_path and hasattr(self.test_model_b_settings, 'system_prompt'):
            system_prompt_b = self.test_model_b_settings.system_prompt.toPlainText().strip()
        
        system_prompt_c = ""
        if model_c_path and hasattr(self, 'test_model_c_settings') and hasattr(self.test_model_c_settings, 'system_prompt'):
            system_prompt_c = self.test_model_c_settings.system_prompt.toPlainText().strip()
        
        # Keep prompts separate - system prompt will be passed separately for instruct models
        prompt_a = user_prompt
        prompt_b = user_prompt
        prompt_c = user_prompt
        
        # Add user message to all columns on the same line
        self.chat_display.add_user_message(user_prompt)
        
        # Run Model A
        if model_a_path:
            self.chat_display.start_model_a_response()
            self._run_inference_a(model_a_path, prompt_a, system_prompt_a)
        
        # Run Model B
        if model_b_path:
            self.chat_display.start_model_b_response()
            self._run_inference_b(model_b_path, prompt_b, system_prompt_b)
        
        # Run Model C
        if model_c_path:
            self.chat_display.start_model_c_response()
            self._run_inference_c(model_c_path, prompt_c, system_prompt_c)
        
        # Clear prompt
        self.test_prompt.clear()
    
    def _filter_inference_output(self, buffer_text: str) -> str:
        """Shared filtering function for both models - removes unwanted log messages and technical output"""
        # First check for OUTPUT marker - this is the cleanest output
        if "--- OUTPUT ---" in buffer_text:
            parts = buffer_text.split("--- OUTPUT ---", 1)
            if len(parts) > 1:
                output = parts[1].strip()
                # If output starts with [ERROR], it's an error message - show it
                if output.startswith("[ERROR]"):
                    return output
                
                # EVEN with OUTPUT marker, we should still filter out any technical lines 
                # that might have leaked into the output stream (e.g. from stderr)
                lines = output.split('\n')
                filtered_lines = []
                technical_patterns = [
                    '[INFO]', '[WARN]', '[OK]', '[DEBUG]', 'Loading', 'Generating...', 'it/s', '█',
                    'Asking to truncate', 'no maximum length is provided',
                    '--base-model', '--prompt', '--adapter-dir', '--system-prompt', # Filter arguments
                    'Python ', 'Usage:', 'python run_adapter.py' # Filter help text
                ]
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    if any(pattern in line_stripped for pattern in technical_patterns):
                        continue
                    # Extra check for common log markers that might not have brackets
                    if line_stripped.startswith(('INFO:', 'WARNING:', 'ERROR:', 'DEBUG:')) and not line_stripped.startswith('ERROR: [ERROR]'):
                        if not line_stripped.startswith('ERROR: '): # Keep actual errors but not just the prefix
                            continue
                    filtered_lines.append(line_stripped)
                return '\n'.join(filtered_lines).strip()
        
        # Check if this looks like a traceback or error output
        # If we see Python traceback patterns, try to extract just the error message
        if "Traceback (most recent call last)" in buffer_text or "File \"" in buffer_text:
            # Look for error messages in the traceback
            lines = buffer_text.split('\n')
            error_lines = []
            in_traceback = False
            for line in lines:
                line_stripped = line.strip()
                if "Traceback" in line_stripped:
                    in_traceback = True
                    continue
                if in_traceback and (line_stripped.startswith(('Error:', 'Exception:', 'RuntimeError:', 'ValueError:', 'TypeError:', 'AttributeError:')) or 
                                     line_stripped.startswith('[ERROR]')):
                    error_lines.append(line_stripped)
                    break
            if error_lines:
                return f"[ERROR] {error_lines[0]}"
            # If we can't find a clear error, return a generic message
            return "[ERROR] An error occurred during model execution. Check logs for details."
        
        # Filter out log lines and technical output
        lines = buffer_text.split('\n')
        filtered_lines = []
        
        # Patterns to filter out (case-insensitive matching)
        filter_patterns = [
            '[INFO]', '[WARN]', '[OK]', '[DEBUG]',
            'FutureWarning', 'UserWarning', 'TRANSFORMERS_CACHE',
            'warnings.warn', 'DeprecationWarning', 'RuntimeWarning', 'ImportWarning',
            'Loading tokenizer', 'Loading base model', 'Loading model', 'Loading checkpoint',
            'Windows detected', 'Generating...', 'Generating text', 'Starting generation',
            'Loading checkpoint shards:', 'Using device', 'Model loaded',
            'CUDA', 'torch', 'transformers', 'device', 'dtype', 'device_map',
            'tokenizer', 'config', 'weights', 'safetensors', '.bin', '.json',
            'it/s', '█', '▌', '▐', '│', 'progress', 'eta', 'steps',
            'checkpoint', 'shard', 'parameter', 'layer', 'module',
            'return cls._from_pretrained', 'self.sp_model', 'get_spm_processor',
            'LoadFromFile', 'SentencePieceProcessor', 'main()', '^' * 10,  # Filter sentencepiece traceback patterns
            'Asking to truncate', 'no maximum length is provided' # Filter truncation warnings
        ]
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            line_lower = line_stripped.lower()
            
            # Skip lines that match filter patterns
            if any(pattern.lower() in line_lower for pattern in filter_patterns):
                continue
            
            # Skip lines that are just carets (^) - common in tracebacks
            if all(c in '^ ' for c in line_stripped):
                continue
            
            # Skip lines that look like file paths or Python source
            if line_lower.startswith(('c:\\', '/', 'file:', 'path:', 'using ', 'loading ', 'return ', 'def ', 'class ', 'import ')):
                continue
            
            # Skip lines that are just numbers, percentages, or progress indicators
            line_clean = line_lower.replace(' ', '').replace('|', '').replace('%', '').replace(':', '').replace('-', '').replace('^', '')
            if line_clean.isdigit() or (line_clean.replace('.', '').isdigit() and len(line_clean) < 10):
                continue
            
            # Skip lines that are just separators or formatting
            if all(c in '|-=_ ' for c in line_stripped):
                continue
            
            # Keep everything else
            filtered_lines.append(line_stripped)
        
        result = '\n'.join(filtered_lines).strip()
        
        # If we filtered everything out, just return empty string for now
        # This prevents "[ERROR] Unable to parse..." from showing while the model is still loading
        if not result:
            return ""
        
        return result
    
    def _run_inference_a(self, model_path: str, prompt: str, system_prompt: str = ""):
        """Run inference for Model A using QProcess"""
        # Reset buffer
        self.inference_buffer_a = ""
        
        # Ensure model_path is a string
        if not model_path or not isinstance(model_path, str):
            self.chat_display.update_model_a_response("[ERROR] Invalid model path provided.")
            return
        
        # Check if this is an adapter or base model
        from pathlib import Path
        model_path_obj = Path(model_path)
        is_adapter = (model_path_obj / "adapter_config.json").exists() or \
                    (model_path_obj / "adapter_model.safetensors").exists() or \
                    (model_path_obj / "adapter_model.bin").exists()
        
        # Detect if this is an instruct model (check if "instruct", "chat", or "-it" is in the path)
        model_path_lower = str(model_path).lower()
        is_instruct = "instruct" in model_path_lower or "chat" in model_path_lower or "-it" in model_path_lower
        model_type = "instruct" if is_instruct else "base"
        
        # Get parameters from Model A settings
        max_tokens = 512
        temperature = 0.7
        if hasattr(self.test_model_a_settings, 'max_tokens'):
            max_tokens = self.test_model_a_settings.max_tokens.value()
        if hasattr(self.test_model_a_settings, 'temperature'):
            temperature = self.test_model_a_settings.temperature.value()
        
        # Build command - use run_adapter.py for both base models and adapters
        cmd = [
            sys.executable, "-u", "run_adapter.py",
            "--prompt", prompt,
            "--max-new-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--model-type", model_type
        ]
        
        # Add system prompt if provided
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]
        
        # Ensure model_path is a string before adding to command
        model_path_str = str(model_path) if model_path else ""
        if not model_path_str:
            self.chat_display.update_model_a_response("[ERROR] Invalid model path: path is empty.")
            return
        
        if is_adapter:
            # Load as adapter (requires base model + adapter)
            cmd += ["--adapter-dir", model_path_str]
            # TODO: Need to specify base model - for now use default
            cmd += ["--base-model", "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"]
        else:
            # Load as base model only
            cmd += ["--base-model", model_path_str, "--no-adapter"]
        
        # Create QProcess
        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        
        # Set GPU selection via environment variable
        env = QProcessEnvironment.systemEnvironment()
        if hasattr(self, 'test_gpu_select') and self.test_gpu_select.isEnabled():
            selected_gpu_idx = self.test_gpu_select.currentIndex()
            env.insert("CUDA_VISIBLE_DEVICES", str(selected_gpu_idx))
        
        proc.setProcessEnvironment(env)
        
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
        
        # Use shared filtering function
        filtered_output = self._filter_inference_output(self.inference_buffer_a)
        if filtered_output:
            self.chat_display.update_model_a_response(filtered_output)
    
    def _on_inference_finished_a(self):
        """Called when Model A inference finishes"""
        # Final update with complete output
        if self.inference_buffer_a.strip():
            self._update_inference_output_a(self.test_proc_a)
            
            # Check if after final update we still have no filtered output
            final_filtered = self._filter_inference_output(self.inference_buffer_a)
            if not final_filtered:
                self.chat_display.update_model_a_response("[ERROR] Model failed to produce any valid output. Check logs for details.")
        
        self.test_proc_a = None
    
    def _run_inference_b(self, model_path: str, prompt: str, system_prompt: str = ""):
        """Run inference for Model B using QProcess"""
        # Reset buffer
        self.inference_buffer_b = ""
        
        # Ensure model_path is a string
        if not model_path or not isinstance(model_path, str):
            self.chat_display.update_model_b_response("[ERROR] Invalid model path provided.")
            return
        
        # Check if this is an adapter or base model
        from pathlib import Path
        model_path_obj = Path(model_path)
        is_adapter = (model_path_obj / "adapter_config.json").exists() or \
                    (model_path_obj / "adapter_model.safetensors").exists() or \
                    (model_path_obj / "adapter_model.bin").exists()
        
        # Detect if this is an instruct model (check if "instruct", "chat", or "-it" is in the path)
        model_path_lower = str(model_path).lower()
        is_instruct = "instruct" in model_path_lower or "chat" in model_path_lower or "-it" in model_path_lower
        model_type = "instruct" if is_instruct else "base"
        
        # Get parameters from Model B settings
        max_tokens = 512
        temperature = 0.7
        if hasattr(self.test_model_b_settings, 'max_tokens'):
            max_tokens = self.test_model_b_settings.max_tokens.value()
        if hasattr(self.test_model_b_settings, 'temperature'):
            temperature = self.test_model_b_settings.temperature.value()
        
        # Build command - use run_adapter.py for both base models and adapters
        cmd = [
            sys.executable, "-u", "run_adapter.py",
            "--prompt", prompt,
            "--max-new-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--model-type", model_type
        ]
        
        # Add system prompt if provided
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]
        
        # Ensure model_path is a string before adding to command
        model_path_str = str(model_path) if model_path else ""
        if not model_path_str:
            self.chat_display.update_model_b_response("[ERROR] Invalid model path: path is empty.")
            return
        
        if is_adapter:
            # Load as adapter (requires base model + adapter)
            cmd += ["--adapter-dir", model_path_str]
            # TODO: Need to specify base model - for now use default
            cmd += ["--base-model", "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"]
        else:
            # Load as base model only
            cmd += ["--base-model", model_path_str, "--no-adapter"]
        
        # Create QProcess
        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        
        # Set GPU selection via environment variable
        env = QProcessEnvironment.systemEnvironment()
        if hasattr(self, 'test_gpu_select') and self.test_gpu_select.isEnabled():
            selected_gpu_idx = self.test_gpu_select.currentIndex()
            env.insert("CUDA_VISIBLE_DEVICES", str(selected_gpu_idx))
        
        proc.setProcessEnvironment(env)
        
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
        
        # Use shared filtering function
        filtered_output = self._filter_inference_output(self.inference_buffer_b)
        if filtered_output:
            self.chat_display.update_model_b_response(filtered_output)
    
    def _on_inference_finished_b(self):
        """Called when Model B inference finishes"""
        # Final update with complete output
        if self.inference_buffer_b.strip():
            self._update_inference_output_b(self.test_proc_b)
            
            # Check if after final update we still have no filtered output
            final_filtered = self._filter_inference_output(self.inference_buffer_b)
            if not final_filtered:
                self.chat_display.update_model_b_response("[ERROR] Model failed to produce any valid output. Check logs for details.")
                
        self.test_proc_b = None
    
    def _run_inference_c(self, model_path: str, prompt: str, system_prompt: str = ""):
        """Run inference for Model C using QProcess"""
        # Reset buffer
        if not hasattr(self, 'inference_buffer_c'):
            self.inference_buffer_c = ""
        else:
            self.inference_buffer_c = ""
        
        # Ensure model_path is a string
        if not model_path or not isinstance(model_path, str):
            self.chat_display.update_model_c_response("[ERROR] Invalid model path provided.")
            return
        
        # Check if this is an adapter or base model
        from pathlib import Path
        model_path_obj = Path(model_path)
        is_adapter = (model_path_obj / "adapter_config.json").exists() or \
                    (model_path_obj / "adapter_model.safetensors").exists() or \
                    (model_path_obj / "adapter_model.bin").exists()
        
        # Detect if this is an instruct model (check if "instruct", "chat", or "-it" is in the path)
        model_path_lower = str(model_path).lower()
        is_instruct = "instruct" in model_path_lower or "chat" in model_path_lower or "-it" in model_path_lower
        model_type = "instruct" if is_instruct else "base"
        
        # Get parameters from Model C settings
        max_tokens = 512
        temperature = 0.7
        if hasattr(self, 'test_model_c_settings'):
            if hasattr(self.test_model_c_settings, 'max_tokens'):
                max_tokens = self.test_model_c_settings.max_tokens.value()
            if hasattr(self.test_model_c_settings, 'temperature'):
                temperature = self.test_model_c_settings.temperature.value()
        
        # Build command - use run_adapter.py for both base models and adapters
        cmd = [
            sys.executable, "-u", "run_adapter.py",
            "--prompt", prompt,
            "--max-new-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--model-type", model_type
        ]
        
        # Add system prompt if provided
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]
        
        # Ensure model_path is a string before adding to command
        model_path_str = str(model_path) if model_path else ""
        if not model_path_str:
            self.chat_display.update_model_c_response("[ERROR] Invalid model path: path is empty.")
            return
        
        if is_adapter:
            # Load as adapter (requires base model + adapter)
            cmd += ["--adapter-dir", model_path_str]
            # TODO: Need to specify base model - for now use default
            cmd += ["--base-model", "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"]
        else:
            # Load as base model only
            cmd += ["--base-model", model_path_str, "--no-adapter"]
        
        # Create QProcess
        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        
        # Set GPU selection via environment variable
        env = QProcessEnvironment.systemEnvironment()
        if hasattr(self, 'test_gpu_select') and self.test_gpu_select.isEnabled():
            selected_gpu_idx = self.test_gpu_select.currentIndex()
            env.insert("CUDA_VISIBLE_DEVICES", str(selected_gpu_idx))
        
        proc.setProcessEnvironment(env)
        
        # Connect to read output and update last bubble
        proc.readyReadStandardOutput.connect(
            lambda: self._update_inference_output_c(proc)
        )
        proc.finished.connect(lambda: self._on_inference_finished_c())
        
        proc.start()
        self.test_proc_c = proc
    
    def _update_inference_output_c(self, proc: QProcess):
        """Update Model C chat bubble with streaming output"""
        # Read output from process
        data = proc.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='replace')
        
        # Accumulate text in buffer
        if not hasattr(self, 'inference_buffer_c'):
            self.inference_buffer_c = ""
        self.inference_buffer_c += text
        
        # Use shared filtering function
        filtered_output = self._filter_inference_output(self.inference_buffer_c)
        if filtered_output:
            self.chat_display.update_model_c_response(filtered_output)
    
    def _on_inference_finished_c(self):
        """Called when Model C inference finishes"""
        # Final update with complete output
        if hasattr(self, 'inference_buffer_c') and self.inference_buffer_c.strip():
            self._update_inference_output_c(self.test_proc_c)
            
            # Check if after final update we still have no filtered output
            final_filtered = self._filter_inference_output(self.inference_buffer_c)
            if not final_filtered:
                self.chat_display.update_model_c_response("[ERROR] Model failed to produce any valid output. Check logs for details.")
                
        self.test_proc_c = None
    
    def _switch_model_settings(self, model_index: int) -> None:
        """Switch between Model A, Model B, and Model C settings"""
        if model_index == 0:
            self.test_model_a_btn.setChecked(True)
            self.test_model_b_btn.setChecked(False)
            if hasattr(self, 'test_model_c_btn'):
                self.test_model_c_btn.setChecked(False)
            self.test_model_settings_stack.setCurrentIndex(0)
        elif model_index == 1:
            self.test_model_a_btn.setChecked(False)
            self.test_model_b_btn.setChecked(True)
            if hasattr(self, 'test_model_c_btn'):
                self.test_model_c_btn.setChecked(False)
            self.test_model_settings_stack.setCurrentIndex(1)
        else:
            self.test_model_a_btn.setChecked(False)
            self.test_model_b_btn.setChecked(False)
            self.test_model_c_btn.setChecked(True)
            self.test_model_settings_stack.setCurrentIndex(2)
    
    def _on_model_count_2_toggled(self, checked: bool) -> None:
        """Handle 2 models checkbox toggle"""
        print(f"DEBUG: _on_model_count_2_toggled called with checked={checked}")
        if checked:
            # Block signals to prevent feedback loop
            self.test_model_count_3.blockSignals(True)
            self.test_model_count_3.setChecked(False)
            self.test_model_count_3.blockSignals(False)
            self._on_model_count_changed("2")
        else:
            # If unchecking 2, ensure 3 is checked (at least one must be checked)
            if not self.test_model_count_3.isChecked():
                self.test_model_count_3.blockSignals(True)
                self.test_model_count_3.setChecked(True)
                self.test_model_count_3.blockSignals(False)
                self._on_model_count_changed("3")
    
    def _on_model_count_3_toggled(self, checked: bool) -> None:
        """Handle 3 models checkbox toggle"""
        print(f"DEBUG: _on_model_count_3_toggled called with checked={checked}")
        if checked:
            # Block signals to prevent feedback loop
            self.test_model_count_2.blockSignals(True)
            self.test_model_count_2.setChecked(False)
            self.test_model_count_2.blockSignals(False)
            self._on_model_count_changed("3")
        else:
            # If unchecking 3, ensure 2 is checked (at least one must be checked)
            if not self.test_model_count_2.isChecked():
                self.test_model_count_2.blockSignals(True)
                self.test_model_count_2.setChecked(True)
                self.test_model_count_2.blockSignals(False)
                self._on_model_count_changed("2")
    
    def _on_model_count_changed(self, count_str: str) -> None:
        """Handle model count change (2 or 3 models)"""
        print(f"DEBUG: _on_model_count_changed called with count_str={count_str}")
        count = int(count_str)
        if count == 3:
            print("DEBUG: Setting up for 3 models")
            # Show Model C
            print("DEBUG: Showing Model C widgets")
            print(f"DEBUG: test_model_c_header_widget exists: {hasattr(self, 'test_model_c_header_widget')}")
            print(f"DEBUG: test_model_c_btn exists: {hasattr(self, 'test_model_c_btn')}")
            self.test_model_c_header_widget.setVisible(True)
            self.test_model_c_btn.setVisible(True)
            # Update button text to short form (A, B, C)
            self.test_model_a_btn.setText("🔵 A")
            self.test_model_b_btn.setText("🟢 B")
            self.test_model_c_btn.setText("🟣 C")
            # Update chat display to 3 columns
            print("DEBUG: Replacing chat display with 3-column version")
            from desktop_app.synchronized_chat_display import SynchronizedChatDisplay
            old_display = self.chat_display
            # Use stored layout reference
            if hasattr(self, 'test_left_layout'):
                # Find index of old display in layout
                index = self.test_left_layout.indexOf(old_display)
                print(f"DEBUG: Old display index in layout: {index}")
                if index == -1:
                    print("DEBUG: WARNING: indexOf returned -1, using addWidget instead")
                    # Remove old display
                    self.test_left_layout.removeWidget(old_display)
                    old_display.setParent(None)
                    old_display.deleteLater()
                    # Create and add new display
                    self.chat_display = SynchronizedChatDisplay(num_models=3)
                    self.chat_display.set_theme(self.dark_mode)
                    self.test_left_layout.addWidget(self.chat_display, 1)
                else:
                    # Remove old display
                    self.test_left_layout.removeWidget(old_display)
                    old_display.setParent(None)
                    old_display.deleteLater()
                    # Create and add new display at same position
                    self.chat_display = SynchronizedChatDisplay(num_models=3)
                    self.chat_display.set_theme(self.dark_mode)
                    self.test_left_layout.insertWidget(index, self.chat_display, 1)
                self.chat_display_widget = self.chat_display
                print("DEBUG: Chat display replaced successfully")
            else:
                print("DEBUG: ERROR: test_left_layout not found!")
        else:
            # Hide Model C
            print("DEBUG: Hiding Model C widgets")
            self.test_model_c_header_widget.setVisible(False)
            # Switch to Model A if Model C was selected
            if self.test_model_c_btn.isChecked():
                self._switch_model_settings(0)
            self.test_model_c_btn.setVisible(False)
            # Update button text to full form (Model A, Model B)
            self.test_model_a_btn.setText("🔵 Model A")
            self.test_model_b_btn.setText("🟢 Model B")
            # Update chat display to 2 columns
            print("DEBUG: Replacing chat display with 2-column version")
            from desktop_app.synchronized_chat_display import SynchronizedChatDisplay
            old_display = self.chat_display
            # Use stored layout reference
            if hasattr(self, 'test_left_layout'):
                # Find index of old display in layout
                index = self.test_left_layout.indexOf(old_display)
                print(f"DEBUG: Old display index in layout: {index}")
                if index == -1:
                    print("DEBUG: WARNING: indexOf returned -1, using addWidget instead")
                    # Remove old display
                    self.test_left_layout.removeWidget(old_display)
                    old_display.setParent(None)
                    old_display.deleteLater()
                    # Create and add new display
                    self.chat_display = SynchronizedChatDisplay(num_models=2)
                    self.chat_display.set_theme(self.dark_mode)
                    self.test_left_layout.addWidget(self.chat_display, 1)
                else:
                    # Remove old display
                    self.test_left_layout.removeWidget(old_display)
                    old_display.setParent(None)
                    old_display.deleteLater()
                    # Create and add new display at same position
                    self.chat_display = SynchronizedChatDisplay(num_models=2)
                    self.chat_display.set_theme(self.dark_mode)
                    self.test_left_layout.insertWidget(index, self.chat_display, 1)
                self.chat_display_widget = self.chat_display
                print("DEBUG: Chat display replaced successfully")
            else:
                print("DEBUG: ERROR: test_left_layout not found!")
        self._update_token_count()
    
    def _update_token_count(self) -> None:
        """Update token count for current model settings"""
        if not hasattr(self, 'test_model_settings_stack'):
            return
        current_page = self.test_model_settings_stack.currentWidget()
        self._update_token_count_for_page(current_page)
    
    def _update_token_count_for_page(self, current_page=None) -> None:
        """Update token count for a specific page"""
        if current_page is None:
            if not hasattr(self, 'test_model_settings_stack'):
                return
            current_page = self.test_model_settings_stack.currentWidget()
        
        if not current_page or not hasattr(current_page, 'token_count'):
            return
        
        system_prompt = ""
        if hasattr(current_page, 'system_prompt'):
            system_prompt = current_page.system_prompt.toPlainText().strip()
        
        user_prompt = ""
        if hasattr(self, 'test_prompt'):
            user_prompt = self.test_prompt.toPlainText().strip()
        
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        total_text = system_prompt + "\n" + user_prompt if system_prompt and user_prompt else (system_prompt or user_prompt)
        estimated_tokens = len(total_text) // 4
        current_page.token_count.setText(f"Tokens: ~{estimated_tokens}")
    
    def _apply_instruction_template(self, template_name: str, system_prompt_widget: QTextEdit) -> None:
        """Apply a predefined instruction template"""
        if template_name == "None":
            return
        
        templates = {
            "Alpaca": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
            "Vicuna": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
            "ChatML": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
            "Llama-2": "[INST] <<SYS>>\nYou are a helpful, harmless, and honest assistant.\n<</SYS>>\n\n",
            "Custom": ""
        }
        
        if template_name in templates:
            system_prompt_widget.setPlainText(templates[template_name])
            self._update_token_count()
        else:
            # Check if it's a saved custom instruction (remove 💾 prefix if present)
            instruction_name = template_name.replace("💾 ", "").strip()
            saved_instructions = self._load_saved_instructions_dict()
            if instruction_name in saved_instructions:
                system_prompt_widget.setPlainText(saved_instructions[instruction_name])
                self._update_token_count()
    
    def _get_saved_instructions_file(self) -> Path:
        """Get the path to the saved instructions JSON file"""
        return self.root / "saved_instructions.json"
    
    def _load_saved_instructions_dict(self) -> dict:
        """Load saved custom instructions from JSON file"""
        instructions_file = self._get_saved_instructions_file()
        if instructions_file.exists():
            try:
                with open(instructions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _load_saved_instructions_into_combo(self, combo: QComboBox) -> None:
        """Load saved custom instructions and add them to the given combo box"""
        saved_instructions = self._load_saved_instructions_dict()
        
        # Add saved custom instructions
        for name in sorted(saved_instructions.keys()):
            combo.addItem(f"💾 {name}")
    
    def _save_custom_instruction(self, system_prompt_widget: QTextEdit, template_combo: QComboBox) -> None:
        """Save the current system prompt as a custom instruction"""
        current_text = system_prompt_widget.toPlainText().strip()
        
        if not current_text:
            QMessageBox.warning(self, "Save Instruction", "Please enter some text in the System Prompt field before saving.")
            return
        
        # Get name from user
        name, ok = QInputDialog.getText(
            self,
            "Save Custom Instruction",
            "Enter a name for this instruction:",
            text=""
        )
        
        if not ok or not name.strip():
            return
        
        name = name.strip()
        
        # Check if name already exists
        saved_instructions = self._load_saved_instructions_dict()
        if name in saved_instructions:
            reply = QMessageBox.question(
                self,
                "Overwrite Instruction",
                f"An instruction named '{name}' already exists. Do you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # Save the instruction
        saved_instructions[name] = current_text
        instructions_file = self._get_saved_instructions_file()
        
        try:
            with open(instructions_file, 'w', encoding='utf-8') as f:
                json.dump(saved_instructions, f, indent=2, ensure_ascii=False)
            
            # Reload this dropdown
            self._load_saved_instructions_into_combo(template_combo)
            
            # Select the newly saved instruction
            index = template_combo.findText(f"💾 {name}")
            if index >= 0:
                template_combo.setCurrentIndex(index)
            
            # Also reload other template dropdowns (Model A and Model B)
            if hasattr(self, 'test_model_settings_stack'):
                for i in range(self.test_model_settings_stack.count()):
                    page = self.test_model_settings_stack.widget(i)
                    if page and hasattr(page, 'template_select'):
                        # Clear and reload
                        combo = getattr(page, 'template_select')
                        # Remove custom items (keep built-in)
                        items_to_remove = []
                        for j in range(combo.count()):
                            text = combo.itemText(j)
                            if text.startswith("💾 "):
                                items_to_remove.append(j)
                        for j in reversed(items_to_remove):
                            combo.removeItem(j)
                        # Add all saved instructions
                        self._load_saved_instructions_into_combo(combo)
            
            QMessageBox.information(self, "Saved", f"Instruction '{name}' has been saved successfully!")
        except IOError as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save instruction: {str(e)}")
    
    def _clear_test_chat(self) -> None:
        """Clear both chat histories"""
        self.chat_display.clear()
        self.test_prompt.clear()

    # ---------------- Info/About tab ----------------
    def _is_package_functional(self, pkg_name: str) -> bool:
        """Thorough check for package functionality beyond just version presence.
        
        IMPORTANT: Uses LLM/.venv Python (not bootstrap/.venv) to check packages.
        """
        if pkg_name not in ["torch", "triton-windows", "bitsandbytes"]:
            return True # Assume other packages are fine if version matches
        
        # Get the target venv Python (LLM/.venv, not bootstrap/.venv)
        try:
            target_python = self._get_target_venv_python()
        except Exception:
            # If we can't find target venv, fall back to current Python (bootstrap)
            # This is not ideal but better than crashing
            target_python = sys.executable
        
        import subprocess
        
        try:
            if pkg_name == "torch":
                # Check torch in target venv
                code = """
import torch
import sys
has_cuda = hasattr(torch, 'cuda')
has_tensor = hasattr(torch, 'Tensor')
if has_cuda and has_tensor:
    print('OK')
else:
    print('BROKEN')
"""
                result = subprocess.run(
                    [target_python, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                return result.returncode == 0 and 'OK' in result.stdout
                
            elif pkg_name == "triton-windows":
                # Check triton in target venv (package name is triton-windows but module is triton)
                # Note: triton-windows may not have triton.ops (Windows port limitation),
                # so we only check if triton itself can be imported
                code = """
try:
    import triton
    # Check if triton has basic functionality
    has_version = hasattr(triton, '__version__')
    has_compile = hasattr(triton, 'compile')
    if has_version and has_compile:
        print('OK')
    else:
        print('BROKEN')
except ImportError:
    print('NOT_INSTALLED')
except Exception as e:
    print('BROKEN')
"""
                result = subprocess.run(
                    [target_python, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                return result.returncode == 0 and 'OK' in result.stdout
                
            elif pkg_name == "bitsandbytes":
                # Check bitsandbytes in target venv
                code = """
try:
    import bitsandbytes
    from bitsandbytes.nn import Linear8bitLt
    print('OK')
except Exception:
    print('BROKEN')
"""
                result = subprocess.run(
                    [target_python, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                return result.returncode == 0 and 'OK' in result.stdout
        except Exception:
            return False
        return True
    
    def _check_version_mismatch(self, pkg_name: str, installed_version: str, requirement_spec: str) -> bool:
        """
        Check if installed version mismatches the profile requirement.
        
        Args:
            pkg_name: Package name
            installed_version: Currently installed version
            requirement_spec: Requirement specifier (e.g., "==4.51.3" or ">=4.51.3,!=4.52.*")
        
        Returns:
            True if version mismatch detected, False if OK
        """
        try:
            from packaging.specifiers import SpecifierSet
            from packaging import version as pkg_version
            
            # Handle exact version specs
            if requirement_spec.startswith("=="):
                required = requirement_spec[2:].strip()
                # Strip build tags for comparison
                inst_base = installed_version.split("+")[0]
                req_base = required.split("+")[0]
                return inst_base != req_base
            
            # Handle complex specs
            spec = SpecifierSet(requirement_spec)
            base_version = installed_version.split("+")[0]
            return not spec.contains(pkg_version.parse(base_version))
        except Exception:
            # If we can't validate, assume OK
            return False
    
    def _get_profile_requirements(self) -> dict:
        """
        Get package requirements from the current hardware profile.
        
        Returns:
            Dict of {package_name: version_spec}
        """
        try:
            import sys
            from pathlib import Path
            
            # Add LLM directory to path if not already there
            llm_dir = Path(__file__).parent.parent
            if str(llm_dir) not in sys.path:
                sys.path.insert(0, str(llm_dir))
            
            # Import modules - system_detector is in root, profile_selector is in core/
            # Match the import style used at the top of the file
            try:
                # system_detector is in LLM root (same as top-level import)
                from system_detector import SystemDetector
                # profile_selector is in core/ (same as top-level import)
                from core.profile_selector import ProfileSelector
            except ImportError as e:
                print(f"[GUI] ERROR: Failed to import SystemDetector/ProfileSelector: {e}")
                print(f"[GUI] LLM dir: {llm_dir}")
                print(f"[GUI] sys.path: {sys.path[:3]}...")
                import traceback
                traceback.print_exc()
                return {}
            
            # Detect hardware
            detector = SystemDetector()
            hw_profile = detector.get_hardware_profile()
            
            # Select profile
            compat_matrix_path = llm_dir / "metadata" / "compatibility_matrix.json"
            if not compat_matrix_path.exists():
                print(f"[GUI] Warning: compatibility_matrix.json not found at {compat_matrix_path}")
                return {}
            
            selector = ProfileSelector(compat_matrix_path)
            profile_name, package_versions, warnings = selector.select_profile(hw_profile)
            
            print(f"[GUI] Loaded profile: {profile_name}")
            print(f"[GUI] Profile versions: {list(package_versions.keys())}")
            if 'bitsandbytes' in package_versions:
                print(f"[GUI] bitsandbytes required version: {package_versions['bitsandbytes']}")
            
            # Convert to requirement specs
            # Profile versions may be ranges (e.g., ">=0.13.0,<0.16.0") or exact (e.g., "0.14.0")
            requirements = {}
            for pkg_name, version in package_versions.items():
                # Check if version is already a range specifier
                if any(op in str(version) for op in [">=", "<=", ">", "<", "!=", ","]):
                    # It's already a range - use as-is
                    requirements[pkg_name] = str(version)
                else:
                    # Exact version - add == prefix
                    requirements[pkg_name] = f"=={version}"
            
            return requirements
        except Exception as e:
            # Log the error but don't crash
            print(f"[GUI] Failed to load profile requirements: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: Try to load from profile JSON files directly
            try:
                print("[GUI] Attempting fallback: loading profile directly from JSON files")
                profiles_dir = llm_dir / "profiles"
                if profiles_dir.exists():
                    # Try to find a profile file with bitsandbytes
                    profile_files = list(profiles_dir.glob("*.json"))
                    for profile_file in profile_files:
                        try:
                            with open(profile_file, 'r', encoding='utf-8') as f:
                                profile_data = json.load(f)
                                if "packages" in profile_data:
                                    # Load all packages from this profile
                                    # Profile versions may be ranges or exact versions
                                    fallback_requirements = {}
                                    for pkg_name, pkg_version in profile_data["packages"].items():
                                        # Check if version is already a range specifier
                                        if any(op in str(pkg_version) for op in [">=", "<=", ">", "<", "!=", ","]):
                                            # It's already a range - use as-is
                                            fallback_requirements[pkg_name] = str(pkg_version)
                                        else:
                                            # Exact version - add == prefix
                                            fallback_requirements[pkg_name] = f"=={pkg_version}"
                                    print(f"[GUI] Fallback loaded {len(fallback_requirements)} packages from {profile_file.name}")
                                    if 'bitsandbytes' in fallback_requirements:
                                        print(f"[GUI] Fallback bitsandbytes: {fallback_requirements['bitsandbytes']}")
                                    return fallback_requirements
                        except Exception as profile_error:
                            print(f"[GUI] Failed to load {profile_file.name}: {profile_error}")
                            continue
            except Exception as fallback_error:
                print(f"[GUI] Fallback also failed: {fallback_error}")
                import traceback
                traceback.print_exc()
            # Final fallback to empty dict
            print("[GUI] ERROR: All profile loading methods failed. Profiles are the ONLY source of truth.")
            print("[GUI] Cannot fall back to requirements.txt - profiles must work!")
            return {}

    def _build_requirements_tab(self) -> QWidget:
        """Build Requirements tab in 3 equal columns (1/3 each)."""
        w = QWidget()
        main_layout = QHBoxLayout(w)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(0)  # Spacing handled by columns
        
        # Use a splitter for fixed 1/3 divisions
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: rgba(102, 126, 234, 0.15); width: 2px; }")
        # Disable manual resizing - maintain fixed 1/3 ratio
        splitter.setChildrenCollapsible(False)
        
        # Column 1 (Scroll Area)
        col1_scroll = QScrollArea()
        col1_scroll.setWidgetResizable(True)
        col1_scroll.setFrameShape(QFrame.NoFrame)
        col1_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        col1_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        col1_container = QWidget()
        col1_layout = QVBoxLayout(col1_container)
        col1_layout.setContentsMargins(15, 10, 15, 10)
        
        col1_title = QLabel("ML CORE")
        col1_title.setStyleSheet("""
            color: #667eea; 
            font-weight: 700; 
            font-size: 11pt; 
            letter-spacing: 3px;
            padding-bottom: 12px;
            border-bottom: 2px solid rgba(102, 126, 234, 0.2);
            margin-bottom: 8px;
        """)
        col1_layout.addWidget(col1_title)
        
        self.requirements_col1 = QVBoxLayout()
        self.requirements_col1.setSpacing(12)
        col1_layout.addLayout(self.requirements_col1)
        col1_layout.addStretch()
        
        col1_scroll.setWidget(col1_container)
        
        # Column 2 (Scroll Area)
        col2_scroll = QScrollArea()
        col2_scroll.setWidgetResizable(True)
        col2_scroll.setFrameShape(QFrame.NoFrame)
        col2_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        col2_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        col2_container = QWidget()
        col2_layout = QVBoxLayout(col2_container)
        col2_layout.setContentsMargins(15, 10, 15, 10)
        
        col2_title = QLabel("OPTIMIZATION & TOOLS")
        col2_title.setStyleSheet("""
            color: #667eea; 
            font-weight: 700; 
            font-size: 11pt; 
            letter-spacing: 3px;
            padding-bottom: 12px;
            border-bottom: 2px solid rgba(102, 126, 234, 0.2);
            margin-bottom: 8px;
        """)
        col2_layout.addWidget(col2_title)
        
        self.requirements_col2 = QVBoxLayout()
        self.requirements_col2.setSpacing(12)
        col2_layout.addLayout(self.requirements_col2)
        col2_layout.addStretch()
        
        col2_scroll.setWidget(col2_container)
        
        # Column 3 (Details Panel)
        self.col3_details = QWidget()
        col3_layout = QVBoxLayout(self.col3_details)
        col3_layout.setContentsMargins(20, 0, 10, 0)
        col3_layout.setSpacing(15)
        
        # Add Refresh Button at the top
        refresh_btn = QPushButton("🔄 Refresh Status")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: rgba(102, 126, 234, 0.2);
                color: #667eea;
                border: 1px solid rgba(102, 126, 234, 0.5);
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 10pt;
                text-align: center;
            }
            QPushButton:hover {
                background: rgba(102, 126, 234, 0.3);
                border: 1px solid #667eea;
            }
            QPushButton:pressed {
                background: rgba(102, 126, 234, 0.4);
            }
        """)
        refresh_btn.clicked.connect(self._refresh_requirements_grid)
        col3_layout.addWidget(refresh_btn)
        
        # Title section with divider
        title_section = QWidget()
        title_layout = QVBoxLayout(title_section)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)
        
        self.selected_pkg_title = QLabel("Select a Requirement")
        self.selected_pkg_title.setStyleSheet("""
            font-size: 20pt; 
            font-weight: 700; 
            color: #ffffff;
            letter-spacing: 0.5px;
            padding-bottom: 12px;
            border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        """)
        title_layout.addWidget(self.selected_pkg_title)
        
        self.selected_pkg_desc = QLabel("Click a card on the left to see details and management tools.")
        self.selected_pkg_desc.setWordWrap(True)
        self.selected_pkg_desc.setStyleSheet("""
            color: #b0b0b0; 
            font-size: 11pt;
            line-height: 1.5;
            padding-top: 8px;
        """)
        title_layout.addWidget(self.selected_pkg_desc)
        
        col3_layout.addWidget(title_section)
        
        # Action Buttons Container (Hidden by default)
        self.pkg_actions_widget = QWidget()
        actions_layout = QVBoxLayout(self.pkg_actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(10)
        
        self.install_btn = QPushButton("📥 Install / Update")
        self.install_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(76, 175, 80, 0.6), stop:1 rgba(102, 126, 234, 0.6));
                color: #ffffff;
                border: 1px solid rgba(76, 175, 80, 0.8);
                padding: 12px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(76, 175, 80, 0.9), stop:1 rgba(102, 126, 234, 0.9));
                border: 1px solid rgba(76, 175, 80, 1.0);
            }
            QPushButton:disabled {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(76, 175, 80, 0.3), stop:1 rgba(102, 126, 234, 0.3));
                color: rgba(255, 255, 255, 0.4);
                border: 1px solid rgba(76, 175, 80, 0.45);
            }
        """)

        self.repair_btn = QPushButton("🔧 Repair Component")
        self.repair_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 152, 0, 0.6), stop:1 rgba(255, 193, 7, 0.6));
                color: #ffffff;
                border: 1px solid rgba(255, 152, 0, 0.8);
                padding: 12px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 152, 0, 0.9), stop:1 rgba(255, 193, 7, 0.9));
                border: 1px solid rgba(255, 152, 0, 1.0);
            }
            QPushButton:disabled {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 152, 0, 0.3), stop:1 rgba(255, 193, 7, 0.3));
                color: rgba(255, 255, 255, 0.4);
                border: 1px solid rgba(255, 152, 0, 0.45);
            }
        """)

        self.uninstall_btn = QPushButton("🗑️ Uninstall Component")
        self.uninstall_btn.setStyleSheet("""
            QPushButton {
                background: rgba(244, 67, 54, 0.6);
                color: #ffffff;
                border: 1px solid rgba(244, 67, 54, 0.8);
                padding: 12px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: rgba(244, 67, 54, 0.9);
                border: 1px solid rgba(244, 67, 54, 1.0);
            }
            QPushButton:disabled {
                background: rgba(244, 67, 54, 0.3);
                color: rgba(255, 255, 255, 0.4);
                border: 1px solid rgba(244, 67, 54, 0.45);
            }
        """)

        # Selected package for the per-package action buttons.
        # The buttons are connected ONCE; clicking a card just changes this value.
        self._selected_requirements_pkg: str | None = None
        self.install_btn.clicked.connect(self._on_selected_pkg_install)
        self.repair_btn.clicked.connect(self._on_selected_pkg_repair)
        self.uninstall_btn.clicked.connect(self._on_selected_pkg_uninstall)
        
        actions_layout.addWidget(self.install_btn)
        actions_layout.addWidget(self.repair_btn)
        actions_layout.addWidget(self.uninstall_btn)
        col3_layout.addWidget(self.pkg_actions_widget)
        self.pkg_actions_widget.setVisible(False)
        
        # Log Section (Always visible)
        log_section = QWidget()
        log_section_layout = QVBoxLayout(log_section)
        log_section_layout.setContentsMargins(0, 0, 0, 0)
        log_section_layout.setSpacing(8)
        
        log_label = QLabel("ACTIVITY LOG")
        log_label.setStyleSheet("""
            color: #667eea; 
            font-weight: 700; 
            font-size: 11pt; 
            letter-spacing: 2px;
            padding-bottom: 8px;
            border-bottom: 2px solid rgba(102, 126, 234, 0.2);
        """)
        log_section_layout.addWidget(log_label)
        
        self.requirements_log = QTextEdit()
        self.requirements_log.setReadOnly(True)
        self.requirements_log.setStyleSheet("""
            QTextEdit {
                background: rgba(0, 0, 0, 0.4);
                color: #4ade80;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid rgba(102, 126, 234, 0.2);
                border-radius: 8px;
                padding: 12px;
            }
        """)
        log_section_layout.addWidget(self.requirements_log, 1)
        
        col3_layout.addWidget(log_section, 1)
        
        # Add to splitter and set equal widths
        splitter.addWidget(col1_scroll)
        splitter.addWidget(col2_scroll)
        splitter.addWidget(self.col3_details)
        
        # Set stretch factors to 1 for all to make them equal
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        
        # Store splitter reference
        self.requirements_splitter = splitter
        
        # Force fixed 1/3 ratio - maintain equal sizes on any resize
        def maintain_fixed_ratio():
            width = splitter.width()
            if width > 0:
                third = width // 3
                current = splitter.sizes()
                # Only update if not already equal (avoid infinite loops)
                if len(current) == 3 and (current[0] != third or current[1] != third or current[2] != third):
                    splitter.setSizes([third, third, third])
        
        # Use event filter to catch all resize events
        class FixedRatioFilter(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Resize and obj is splitter:
                    QTimer.singleShot(1, maintain_fixed_ratio)
                return False
        
        filter_obj = FixedRatioFilter()
        splitter.installEventFilter(filter_obj)
        if not hasattr(self, '_requirements_splitter_filter'):
            self._requirements_splitter_filter = filter_obj  # Keep reference
        
        # Set initial sizes
        QTimer.singleShot(100, maintain_fixed_ratio)
        
        # Also check periodically (backup in case events are missed)
        if not hasattr(self, '_requirements_ratio_timer'):
            self._requirements_ratio_timer = QTimer()
            self._requirements_ratio_timer.timeout.connect(maintain_fixed_ratio)
            self._requirements_ratio_timer.start(100)  # Check every 100ms
        
        main_layout.addWidget(splitter)
        
        self.package_cards = {}
        self._refresh_requirements_grid()
        
        return w

    def _refresh_requirements_grid(self):
        """Populate the requirements columns with package cards."""
        # Clear existing
        for layout in [self.requirements_col1, self.requirements_col2]:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget(): item.widget().deleteLater()
        
        self.package_cards = {}
        
        # Get target venv Python for checking installed versions
        # IMPORTANT: Check packages in LLM/.venv, not bootstrap/.venv
        try:
            target_python = self._get_target_venv_python()
        except Exception as e:
            print(f"[GUI] WARNING: Could not get target venv Python: {e}")
            target_python = None
        
        # Helper function to get installed version from target venv
        def get_installed_version(pkg_name: str) -> Optional[str]:
            """Get installed version from target venv (LLM/.venv), not bootstrap"""
            if target_python is None:
                return None
            try:
                import subprocess
                code = f"""
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    print(version('{pkg_name}'))
except PackageNotFoundError:
    print('NOT_FOUND')
"""
                result = subprocess.run(
                    [target_python, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output and output != "NOT_FOUND":
                        return output
            except Exception:
                pass
            return None
        
        # PROFILE IS THE ONLY SOURCE OF TRUTH - NO requirements.txt FALLBACK
        profile_requirements = self._get_profile_requirements()
        
        if not profile_requirements:
            print("[GUI] ERROR: Failed to load profile requirements! This is a critical error.")
            print("[GUI] Profiles are the ONLY source of truth. Cannot proceed without profile.")
            # Show error to user
            QMessageBox.critical(
                self,
                "Profile Loading Failed",
                "Failed to load hardware profile requirements.\n\n"
                "Profiles are the only source of truth for package versions.\n"
                "Please check:\n"
                "- metadata/compatibility_matrix.json exists\n"
                "- profiles/*.json files exist\n"
                "- System detection is working correctly"
            )
            # Return empty - GUI will show packages as missing
            required_packages = {}
        else:
            print(f"[GUI] Loaded {len(profile_requirements)} packages from profile (ONLY SOURCE OF TRUTH)")
            if 'bitsandbytes' in profile_requirements:
                print(f"[GUI] Profile requires bitsandbytes: {profile_requirements['bitsandbytes']}")
            required_packages = profile_requirements
        
        # Add any remaining defaults for packages not in profile
        defaults = {
            "triton-windows": ">=3.0.0",
            "PySide6": "==6.8.1",
            "unsloth": "Latest"
        }
        for pkg, ver in defaults.items():
            if pkg not in required_packages:
                required_packages[pkg] = ver
        
        col1_pkgs = ["torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes"]
        col2_pkgs = ["triton-windows", "unsloth", "PySide6", "numpy", "pandas", "huggingface_hub"]
        
        descriptions = {
            "torch": "Foundational deep learning framework. Required for all model operations.",
            "transformers": "HuggingFace library providing model architectures and loading utilities.",
            "peft": "Parameter-Efficient Fine-Tuning. Enables LoRA and QLoRA training.",
            "datasets": "Utilities for loading and processing training data efficiently.",
            "accelerate": "Handles multi-GPU and mixed-precision training orchestration.",
            "bitsandbytes": "Enables 4-bit and 8-bit quantization for low-VRAM training.",
            "triton-windows": "GPU programming language. Required for Unsloth optimizations on Windows.",
            "unsloth": "Advanced training optimizer that provides 2x speedup and 70% less VRAM usage.",
            "PySide6": "The Qt-based GUI framework used to build this desktop application.",
            "numpy": "Numerical computing library. Essential for tensor and array operations.",
            "pandas": "Data manipulation library used for dataset analysis and visualization.",
            "huggingface_hub": "Client library for interacting with the Hugging Face Model Hub."
        }
        
        for pkg_list, layout in [(col1_pkgs, self.requirements_col1), (col2_pkgs, self.requirements_col2)]:
            for pkg_name in pkg_list:
                req_ver = required_packages.get(pkg_name, "")
                inst_ver = None
                
                # Check installed version in target venv (LLM/.venv), not bootstrap
                if pkg_name == "triton-windows":
                    # triton-windows often isn't available for some Python versions on Windows.
                    # We treat it as optional/unavailable rather than a hard "missing".
                    # Package name is "triton-windows" but module is "triton", so check both.
                    inst_ver = get_installed_version("triton-windows")
                    if inst_ver is None:
                        inst_ver = get_installed_version("triton")
                else:
                    inst_ver = get_installed_version(pkg_name)
                
                is_installed = bool(inst_ver)
                is_functional = self._is_package_functional(pkg_name) if is_installed else True
                
                # Check for version mismatch (wrong version installed)
                # Always check version if installed and requirement is specified, regardless of functional status
                # This ensures we detect wrong versions even if the package appears functional
                version_mismatch = False
                if is_installed and req_ver:
                    version_mismatch = self._check_version_mismatch(pkg_name, inst_ver, req_ver)
                
                if is_installed:
                    if not is_functional:
                        status_text = "BROKEN"
                        status_color = "#f44336"
                    elif version_mismatch:
                        status_text = "WRONG VERSION"
                        status_color = "#ff9800"  # Orange for version mismatch
                    else:
                        status_text = "OK"
                        status_color = "#4CAF50"
                else:
                    if pkg_name == "triton-windows" and sys.platform == "win32":
                        status_text = "OPTIONAL"
                        status_color = "#9e9e9e"
                    else:
                        status_text = "MISSING"
                        status_color = "#f44336"
                
                card = PackageCard(
                    pkg_name, req_ver, inst_ver, is_installed, 
                    "ok" if (is_installed and is_functional and not version_mismatch) else ("mismatch" if version_mismatch else "missing"),
                    status_color, status_text, descriptions.get(pkg_name, "")
                )
                card.clicked.connect(self._on_package_card_clicked)
                layout.addWidget(card)
                self.package_cards[pkg_name] = card
        
        # Add stretches to keep cards at the top
        self.requirements_col1.addStretch()
        self.requirements_col2.addStretch()

    def _on_package_card_clicked(self, pkg_name):
        # Update selection state
        for name, card in self.package_cards.items():
            card.set_selected(name == pkg_name)
            
        pkg_card = self.package_cards[pkg_name]
        self.selected_pkg_title.setText(pkg_name.upper())
        
        status_info = f"<span style='color: {pkg_card.status_color}; font-weight: bold;'>{pkg_card.status_text}</span>"
        details = f"Status: {status_info}<br>"
        details += f"Required: {pkg_card.required_version if pkg_card.required_version else 'Any'}<br>"
        details += f"Installed: {pkg_card.installed_version if pkg_card.installed_version else 'Not installed'}<br><br>"
        details += pkg_card.description
        
        self.selected_pkg_desc.setText(details)
        self.pkg_actions_widget.setVisible(True)

        # Select package for action buttons
        self._selected_requirements_pkg = pkg_name

        # Get package status
        status = pkg_card.status_text

        # Configure buttons based on package status
        # Note: Repair now uses smart repair mode - only fixes broken packages, doesn't reinstall everything
        if status == "OK":
            self.install_btn.setEnabled(False)
            self.repair_btn.setEnabled(False)
            self.uninstall_btn.setEnabled(True)
            self.install_btn.setToolTip("Package already installed")
            self.repair_btn.setToolTip("Package is working correctly")
        elif status == "MISSING":
            self.install_btn.setEnabled(True)
            self.repair_btn.setEnabled(False)
            self.uninstall_btn.setEnabled(False)
            self.install_btn.setToolTip("Install this package")
            self.repair_btn.setToolTip("Package not installed")
        elif status == "WRONG VERSION":
            self.install_btn.setEnabled(True)
            self.repair_btn.setEnabled(True)
            self.uninstall_btn.setEnabled(True)
            self.install_btn.setToolTip("Update to correct version")
            self.repair_btn.setToolTip("Repair will uninstall wrong version and install correct version")
        else:  # BROKEN or other issues
            self.install_btn.setEnabled(True)
            self.repair_btn.setEnabled(True)
            self.uninstall_btn.setEnabled(True)
            self.install_btn.setToolTip("Update this package")
            self.repair_btn.setToolTip("Repair environment (fixes broken/wrong packages)")

    def _run_pkg_task(self, pkg_name, action, checked=False):
        """Execute a package-specific task (install/repair/uninstall)."""
        # Visible feedback in the UI immediately.
        self.requirements_log.clear()
        self.requirements_log.append(f"<b>Executing {action} for {pkg_name}...</b>\n")

        # IMPORTANT:
        # - Per-package install/uninstall must NOT trigger a full requirements install.
        # - Repair is environment-level (uses InstallerV2 targeted repair).
        if action == "repair":
            self._run_installer_task("repair", f"Repairing environment (from {pkg_name})...")
            return

        # For install/uninstall, run a single pip operation for just this package.
        pkg_card = self.package_cards.get(pkg_name)
        version_spec = getattr(pkg_card, "required_version", "") if pkg_card else ""

        # "Latest" means no pin.
        if isinstance(version_spec, str) and version_spec and version_spec.lower() != "latest":
            package_spec = f"{pkg_name}{version_spec}"
        else:
            package_spec = pkg_name

        # Disable all cards during task
        for card in self.package_cards.values():
            card.setEnabled(False)
        self.install_btn.setEnabled(False)
        self.repair_btn.setEnabled(False)
        self.uninstall_btn.setEnabled(False)

        pip_action = "uninstall" if action == "uninstall" else "install"
        self.pip_thread = PipPackageThread(pip_action, package_spec)
        self.pip_thread.log_output.connect(lambda m: self.requirements_log.append(m))
        self.pip_thread.finished_signal.connect(self._on_pip_task_finished)
        self.pip_thread.start()

    def _on_pip_task_finished(self, success: bool):
        for card in self.package_cards.values():
            card.setEnabled(True)
        msg = "✅ Task completed successfully!" if success else "❌ Task failed. Check log for details."
        self.requirements_log.append(f"\n<b>{msg}</b>")
        self._refresh_requirements_grid()

    def _on_selected_pkg_install(self, checked=False):
        if not self._selected_requirements_pkg:
            return
        self._run_pkg_task(self._selected_requirements_pkg, "install", checked)

    def _on_selected_pkg_repair(self, checked=False):
        if not self._selected_requirements_pkg:
            return
        self._run_pkg_task(self._selected_requirements_pkg, "repair", checked)

    def _on_selected_pkg_uninstall(self, checked=False):
        if not self._selected_requirements_pkg:
            return
        self._run_pkg_task(self._selected_requirements_pkg, "uninstall", checked)

    def _on_install_requirements(self):
        self._run_installer_task("dependencies", "Installing requirements...")

    def _on_repair_environment(self):
        self._run_installer_task("repair", "Repairing environment...")

    def _on_uninstall_requirements(self):
        confirm = QMessageBox.question(self, "Confirm Uninstall", "Are you sure you want to uninstall all managed packages?", QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self._run_installer_task("uninstall", "Uninstalling packages...")

    def _run_installer_task(self, task_type, start_msg):
        self.requirements_log.append(f"<b>{start_msg}</b>\n")
        
        # Disable all cards during task
        for card in self.package_cards.values(): card.setEnabled(False)
        
        self.installer_thread = InstallerThread(task_type)
        self.installer_thread.log_output.connect(lambda m: self.requirements_log.append(m))
        self.installer_thread.finished_signal.connect(self._on_installer_task_finished)
        self.installer_thread.start()

    def _on_installer_task_finished(self, success):
        for card in self.package_cards.values(): card.setEnabled(True)
        msg = "✅ Task completed successfully!" if success else "❌ Task failed. Check log for details."
        self.requirements_log.append(f"\n<b>{msg}</b>")
        self._refresh_requirements_grid()

    def _build_info_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # Title
        title_container = QWidget()
        title_container.setStyleSheet("background: transparent;")
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)
        
        # Title text (centered)
        title = QLabel("ℹ️ About LLM Fine-tuning Studio")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24pt; font-weight: bold; text-decoration: none;")
        title_layout.addWidget(title)
        
        layout.addWidget(title_container)
        
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
        
        # Credits title
        credits_title = QLabel("💝 Credits")
        credits_title.setAlignment(Qt.AlignLeft)
        credits_title.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none; margin: 0; padding: 0;")
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
        
        license_title = QLabel("📜 License Information")
        license_title.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none;")
        license_inner.addWidget(license_title)
        
        license_text = QLabel("""
<p style="line-height: 1.3; font-size: 10pt;">
This application uses the following open-source libraries and tools:
</p>

<h3 style='text-decoration: none;' style="color: #667eea; margin-top: 12px; font-size: 12pt;">Core Libraries</h3>
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

<h3 style='text-decoration: none;' style="color: #667eea; margin-top: 12px; font-size: 12pt;">Training & Data</h3>
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

<h3 style='text-decoration: none;' style="color: #667eea; margin-top: 12px; font-size: 12pt;">Acceleration</h3>
<ul style="line-height: 1.4; font-size: 10pt;">
<li><b>xFormers</b> - BSD (Meta)<br>
    <span style="color: #888;">Memory-efficient attention</span></li>
<li><b>CUDA Toolkit 12.4</b> - NVIDIA EULA<br>
    <span style="color: #888;">GPU acceleration</span></li>
<li><b>Triton</b> - MIT (OpenAI)<br>
    <span style="color: #888;">GPU programming</span></li>
</ul>

<h3 style='text-decoration: none;' style="color: #667eea; margin-top: 12px; font-size: 12pt;">Utilities</h3>
<ul style="line-height: 1.4; font-size: 10pt;">
<li><b>huggingface_hub</b> - Apache 2.0</li>
<li><b>psutil</b> - BSD-3-Clause</li>
<li><b>pandas</b> - BSD-3-Clause</li>
<li><b>numpy</b> - BSD-3-Clause</li>
</ul>

<h3 style='text-decoration: none;' style="color: #667eea; margin-top: 12px;">Models</h3>
<ul style="line-height: 1.4;">
<li><b>Llama Models</b> - Llama Community License (Meta)<br>
    <span style="color: #888;">Commercial use with restrictions</span></li>
<li><b>Other Models</b> - Various licenses<br>
    <span style="color: #888;">Check model card on Hugging Face</span></li>
</ul>

<h3 style='text-decoration: none;' style="color: #667eea; margin-top: 12px;">Important Notes</h3>
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
        
        # Note: Corner bottom-right image is handled by hybrid_frame system via _update_frame_corner_br()
        # which uses corner_br_owl_info for Info tab (index 8) and sizes it to 150px width (height adaptable)
        
        # Get asset path for owl_credits.webp and add as overlay in top-left corner
        owl_credits_path = None
        if hasattr(self, '_get_frame_asset_path'):
            owl_credits_path = self._get_frame_asset_path("owl_credits")
        else:
            # Fallback: construct path directly
            llm_dir = Path(__file__).parent.parent
            root_dir = llm_dir.parent
            assets_dir = root_dir / "hybrid_frame_module" / "assets"
            webp_path = assets_dir / "owl_credits.webp"
            if webp_path.exists():
                owl_credits_path = str(webp_path)
        
        # Add owl_credits image as overlay - align BOTTOM edge with BOTTOM edge of Credits title line
        if owl_credits_path:
            owl_label = QLabel(credits_frame)  # Parent is credits_frame for simpler coordinate system
            owl_label.setAttribute(Qt.WA_TransparentForMouseEvents)  # Don't block mouse events
            pixmap = QPixmap(owl_credits_path)
            if not pixmap.isNull():
                # Scale to width 150, height adaptable
                scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                owl_label.setPixmap(scaled_pixmap)
                owl_label.setFixedSize(scaled_pixmap.size())
                owl_label.setStyleSheet("background: transparent; border: none;")
                owl_label.raise_()  # Make sure it's on top
                owl_label.show()  # Make sure it's visible
                
                # Position overlay: align BOTTOM edge of image with BOTTOM edge of Credits title text line
                def update_owl_position():
                    if not credits_frame.isVisible() or not credits_title.isVisible():
                        return
                    
                    # Ensure widgets have valid geometry
                    if credits_title.width() <= 0 or credits_title.height() <= 0:
                        return
                    
                    # Get credits_title position relative to credits_frame (parent of owl_label)
                    title_rect = credits_title.geometry()  # This is relative to credits_frame
                    if title_rect.width() <= 0 or title_rect.height() <= 0:
                        return
                    
                    # Get the BOTTOM edge of the Credits title text line
                    title_bottom_y = title_rect.y() + title_rect.height()
                    
                    label_width = owl_label.width()
                    label_height = owl_label.height()
                    
                    if label_width > 0 and label_height > 0:
                        # Position: left side of Credits title, BOTTOM of image aligned with BOTTOM of text line
                        x = title_rect.x() - label_width - 12  # 12px spacing before text
                        y = title_bottom_y - label_height  # Image BOTTOM edge = text line BOTTOM edge
                        
                        # Ensure we don't go outside the widget bounds
                        if y < 0:
                            y = 0
                        if x < 0:
                            x = 0
                        
                        owl_label.setGeometry(int(x), int(y), label_width, label_height)
                        owl_label.show()
                        owl_label.raise_()
                
                # Use event filter to catch resize events
                class OwlResizeFilter(QObject):
                    def eventFilter(self, obj, event):
                        if (obj == credits_frame or obj == credits_title) and (event.type() == QEvent.Type.Resize or event.type() == QEvent.Type.Move):
                            QTimer.singleShot(0, update_owl_position)
                        elif (obj == credits_frame or obj == credits_title) and event.type() == QEvent.Type.Show:
                            QTimer.singleShot(0, update_owl_position)
                            QTimer.singleShot(50, update_owl_position)
                            QTimer.singleShot(200, update_owl_position)
                        return False
                
                owl_resize_filter = OwlResizeFilter()
                credits_frame.installEventFilter(owl_resize_filter)
                credits_title.installEventFilter(owl_resize_filter)
                
                # Update position initially and after delays to ensure layout is complete
                QTimer.singleShot(0, update_owl_position)
                QTimer.singleShot(50, update_owl_position)
                QTimer.singleShot(100, update_owl_position)
                QTimer.singleShot(300, update_owl_position)
                QTimer.singleShot(500, update_owl_position)
                QTimer.singleShot(1000, update_owl_position)
        
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
    def _append_training_output(self, proc: QProcess) -> None:
        """Parse training output and update dashboard metrics + filtered logs"""
        # Read available data
        data = proc.readAllStandardOutput().data()
        if not data:
            return
            
        # Decode text
        new_text = data.decode("utf-8", errors="replace")
        
        # Add to buffer
        if not hasattr(self, '_train_output_buffer'):
            self._train_output_buffer = ""
        self._train_output_buffer += new_text
        
        # Process complete lines/segments (tqdm uses \r)
        import re
        if '\n' in self._train_output_buffer or '\r' in self._train_output_buffer:
            # Split by either \n or \r to handle both standard logs and progress bars
            lines = re.split(r'[\n\r]+', self._train_output_buffer)
            
            # The last element is potentially a partial line, keep it in buffer
            self._train_output_buffer = lines.pop()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Parse metrics from this line
                self._parse_single_line_metrics(line)
                
                # Filter and append to log
                self._filter_and_append_to_train_log(line)

    def _filter_and_append_to_train_log(self, line: str) -> None:
        """Filter out noise and append important training logs"""
        # Skip lines that are just noise
        line_clean = line.strip()
        if not line_clean:
            return
            
        skip_patterns = [
            '🦥 Unsloth Zoo will now patch',  # Repeated unsloth messages
            'Unsloth: Tokenizing',  # Tokenization progress bars
            '%|' in line,  # Any tqdm progress bar
            'examples/s' in line or 'it/s' in line,  # Progress bar speed indicators
            line_clean and line_clean[0].isdigit() and '%|' in line,  # Progress bar lines starting with numbers
            '[' in line and ']' in line and ('examples/s' in line or 'it/s' in line),  # tqdm format
        ]
        
        if any(skip_patterns):
            return
        
        # Only show: important messages, errors, warnings, tracebacks, and training summaries
        line_lower = line.lower()
        should_show = (
            # Important markers
            '[INFO]' in line or '[WARNING]' in line or '[ERROR]' in line or
            'traceback' in line_lower or 'error:' in line_lower or 'exception:' in line_lower or
            'starting' in line_lower or 'saving' in line_lower or 'complete' in line_lower or
            'loading' in line_lower or 'preparing' in line_lower or
            # Training step updates (JSON format with loss/epoch)
            ('{' in line and 'loss' in line_lower and ('epoch' in line_lower or 'step' in line_lower)) or
            # Epoch/step boundaries (but not progress bars)
            ('epoch' in line_lower or 'step' in line_lower) and ('/' in line or 'of' in line) and '%|' not in line or
            # Errors and warnings (more inclusive)
            'failed' in line_lower or 'not found' in line_lower or 'no such' in line_lower or
            'invalid' in line_lower or 'missing' in line_lower or
            # File paths and OS errors
            'path' in line_lower or 'directory' in line_lower or 'file' in line_lower or
            # Important recommendations
            'recommended' in line_lower or 'attention' in line_lower
        )
        
        # Always show the first few lines of training to confirm it started correctly
        # or if the line seems to be a real message (not just progress noise)
        if not hasattr(self, '_train_line_count'):
            self._train_line_count = 0
        self._train_line_count += 1
        
        if self._train_line_count < 20 or should_show:
            self.train_log.appendPlainText(line.rstrip('\r'))
    
    def _parse_single_line_metrics(self, line: str) -> None:
        """Parse a single line of training output for metrics"""
        import re
        import json
        import ast
        
        line = line.strip()
        if not line:
            return
            
        # Try to find a JSON-like dictionary in the line
        # Use a more flexible regex that finds things like {'loss': 1.23} or {"loss": 1.23}
        dict_match = re.search(r'\{[^{}]+\}', line)
        if dict_match:
            dict_str = dict_match.group(0)
            metrics = None
            
            # Try standard JSON first
            try:
                metrics = json.loads(dict_str)
            except:
                # Try ast.literal_eval for Python-style dicts (single quotes)
                try:
                    metrics = ast.literal_eval(dict_str)
                except:
                    # Last resort: replace ' with " and try JSON
                    try:
                        metrics = json.loads(dict_str.replace("'", '"'))
                    except:
                        pass
            
            if metrics and isinstance(metrics, dict):
                # Update epoch (float value, e.g., 0.01)
                if 'epoch' in metrics and hasattr(self, 'epoch_card'):
                    epoch_val = float(metrics['epoch'])
                    total_epochs = self.train_epochs.value() if hasattr(self, 'train_epochs') else 1
                    # Show 2 decimal places if it's a fraction, otherwise int/total
                    if epoch_val < total_epochs and (epoch_val * 10) % 10 != 0:
                        self._update_metric_card(self.epoch_card, f"{epoch_val:.2f}/{total_epochs}")
                    else:
                        self._update_metric_card(self.epoch_card, f"{int(epoch_val)}/{total_epochs}")
                
                # Update step
                if 'step' in metrics and hasattr(self, 'steps_card'):
                    step_val = int(metrics['step'])
                    # Try to get total steps or estimate
                    total_steps = metrics.get('total_steps', '?')
                    self._update_metric_card(self.steps_card, f"{step_val}/{total_steps}")
                
                # Update loss
                if 'loss' in metrics and hasattr(self, 'loss_card'):
                    loss_val = float(metrics['loss'])
                    self._update_metric_card(self.loss_card, f"{loss_val:.4f}")
                
                # Update learning rate
                if 'learning_rate' in metrics and hasattr(self, 'learning_rate_card'):
                    lr_val = float(metrics['learning_rate'])
                    self._update_metric_card(self.learning_rate_card, f"{lr_val:.2e}")
                elif 'lr' in metrics and hasattr(self, 'learning_rate_card'):
                    lr_val = float(metrics['lr'])
                    self._update_metric_card(self.learning_rate_card, f"{lr_val:.2e}")
                
                # Update speed
                if 'samples_per_second' in metrics and hasattr(self, 'speed_card'):
                    self._update_metric_card(self.speed_card, f"{float(metrics['samples_per_second']):.1f} s/s")
                elif 'train_samples_per_second' in metrics and hasattr(self, 'speed_card'):
                    self._update_metric_card(self.speed_card, f"{float(metrics['train_samples_per_second']):.1f} s/s")
                
                return # Successfully parsed from dict

        # Fallback: regex patterns for non-JSON/dict formats
        # Handle formats like "loss: 1.23" or "epoch [1/5]" or "step 100"
        
        # Parse epoch: "epoch 1/5" or "Epoch 1/5" or "epoch: 1.23"
        epoch_match = re.search(r"['\"]?epoch['\"]?[:\s=]+([\d./]+)", line, re.IGNORECASE)
        if epoch_match:
            val = epoch_match.group(1)
            if '/' in val:
                self._update_metric_card(self.epoch_card, val)
            else:
                try:
                    epoch_val = float(val)
                    total_epochs = self.train_epochs.value() if hasattr(self, 'train_epochs') else 1
                    self._update_metric_card(self.epoch_card, f"{epoch_val:.2f}/{total_epochs}")
                except: pass
        
        # Parse step: "step 100/500" or "Step: 100"
        step_match = re.search(r"['\"]?step['\"]?[:\s=]+([\d/]+)", line, re.IGNORECASE)
        if step_match:
            val = step_match.group(1)
            if hasattr(self, '_total_train_steps') and self._total_train_steps and '/' not in val:
                val = f"{val}/{self._total_train_steps}"
            self._update_metric_card(self.steps_card, val)
        
        # Parse loss: "loss: 0.1234" or "'loss': 0.1234"
        loss_match = re.search(r"['\"]?loss['\"]?[:\s=]+([\d.]+)", line, re.IGNORECASE)
        if loss_match:
            try:
                loss_val = float(loss_match.group(1))
                self._update_metric_card(self.loss_card, f"{loss_val:.4f}")
            except: pass
        
        # Parse learning rate: "lr: 0.0002"
        lr_match = re.search(r"['\"]?(?:learning_rate|lr)['\"]?[:\s=]+([\d.e-]+)", line, re.IGNORECASE)
        if lr_match:
            try:
                lr_val = float(lr_match.group(1))
                self._update_metric_card(self.learning_rate_card, f"{lr_val:.2e}")
            except: pass
        
        # Parse speed: "X.XX it/s" or "X.XX samples/s"
        speed_match = re.search(r"([\d.]+)\s*(?:it|sample|s)s?/s", line, re.IGNORECASE)
        if speed_match:
            try:
                speed_val = float(speed_match.group(1))
                self._update_metric_card(self.speed_card, f"{speed_val:.1f} s/s")
            except: pass

        # Parse samples_per_sec (JSON metric)
        sps_match = re.search(r"['\"]?(?:samples_per_sec|samples_per_second)['\"]?[:\s=]+([\d.]+)", line, re.IGNORECASE)
        if sps_match:
            try:
                sps_val = float(sps_match.group(1))
                self._update_metric_card(self.speed_card, f"{sps_val:.1f} s/s")
            except: pass

        # Parse ETA seconds
        eta_match = re.search(r"['\"]?eta_sec['\"]?[:\s=]+([\d.]+)", line, re.IGNORECASE)
        if eta_match:
            try:
                eta_val = float(eta_match.group(1))
                minutes = int(eta_val // 60)
                seconds = int(eta_val % 60)
                self._update_metric_card(self.eta_card, f"{minutes}m {seconds}s")
            except: pass

        # Parse total_steps to improve denominator
        total_steps_match = re.search(r"['\"]?total_steps['\"]?[:\s=]+([\d]+)", line, re.IGNORECASE)
        if total_steps_match:
            try:
                self._total_train_steps = int(total_steps_match.group(1))
                # If we already have a current step, refresh display
                current_step_text = self.steps_card.value_label.text()
                if '/' in current_step_text:
                    current_val = current_step_text.split('/')[0]
                else:
                    current_val = current_step_text
                self._update_metric_card(self.steps_card, f"{current_val}/{self._total_train_steps}")
            except:
                pass

    def _parse_training_metrics(self, text: str) -> None:
        """Parse training output text for metrics (legacy, calls _parse_single_line_metrics)"""
        # This remains for backward compatibility but usually we call _parse_single_line_metrics now
        lines = text.replace('\r', '\n').split('\n')
        for line in lines:
            self._parse_single_line_metrics(line)

    def _update_training_stats_periodic(self) -> None:
        """Periodic background update for training stats (GPU memory, etc.)"""
        try:
            from system_detector import SystemDetector
            detector = SystemDetector()
            
            # Update GPU memory
            gpu_stats = detector.get_gpu_memory_usage()
            if gpu_stats and hasattr(self, 'gpu_mem_card'):
                # For simplicity, if multiple GPUs, show the one with highest usage (usually the one training)
                # Or if we have CUDA_VISIBLE_DEVICES, we could try to match it.
                max_gpu = max(gpu_stats, key=lambda x: x['used_mb'])
                used_gb = max_gpu['used_mb'] / 1024.0
                total_gb = max_gpu['total_mb'] / 1024.0
                self._update_metric_card(self.gpu_mem_card, f"{used_gb:.1f}/{total_gb:.1f} GB")
        except:
            pass
    
    def _update_metric_card(self, card: QWidget, value: str) -> None:
        """Update the value displayed in a metric card"""
        # Use the set_value method stored on the card
        if hasattr(card, 'set_value'):
            card.set_value(value)
        else:
            # Fallback: find value label manually
            labels = card.findChildren(QLabel)
            for label in labels:
                if "18pt" in label.styleSheet():
                    label.setText(value)
                    return
    
    def _append_proc_output(self, proc: QProcess, widget: QPlainTextEdit) -> None:
        """Generic output appender for non-training processes"""
        data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            widget.appendPlainText(data.rstrip("\n"))

    def _background_system_check(self):
        """Backward-compatible entrypoint (kept for older call sites)."""
        self._start_background_detection()

    def _log_to_app_log(self, message: str) -> None:
        """Best-effort logger for pythonw launches (no console)."""
        try:
            logs_dir = self.root / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Use timestamped session log name if available
            if hasattr(self, 'session_log_name'):
                log_path = logs_dir / self.session_log_name
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime("%y%m%d")
                time_str = datetime.now().strftime("%H%M")
                log_path = logs_dir / f"[{timestamp}][app][{time_str}].log"
                self.session_log_name = log_path.name
                
            with log_path.open("a", encoding="utf-8", errors="replace") as f:
                f.write(message.rstrip() + "\n")
        except Exception:
            pass

    def _start_background_detection(self) -> None:
        """Run system detection in a worker thread and apply results to the UI."""
        # Ensure UI is fully constructed before running detection (timers can fire during __init__
        # because the splash screen calls processEvents()).
        if not hasattr(self, "tabs") or self.tabs is None:
            QTimer.singleShot(250, self._start_background_detection)
            return

        if self._bg_detect_started:
            return
        self._bg_detect_started = True
        self._log_to_app_log("[BACKGROUND] Starting system detection thread...")

        t = SystemDetectThread()
        self._bg_detect_thread = t

        def _on_detected(system_info: dict) -> None:
            try:
                self.system_info.update(system_info)
                self._update_header_system_info()

                # Refresh Home tab to show detected values (deferred to avoid blocking)
                try:
                    if hasattr(self, "tabs") and self.tabs is not None and self.tabs.count() > 0:
                        # Only rebuild if Home tab is currently visible to avoid unnecessary work
                        current_index = self.tabs.currentIndex()
                        if current_index == 0:
                            # Defer rebuild to next event loop cycle
                            QTimer.singleShot(100, self._rebuild_home_tab)
                        else:
                            # Home tab not visible, just mark it as needing update
                            self._home_tab_needs_update = True
                except Exception as e:
                    self._log_to_app_log(f"[BACKGROUND] Failed to schedule Home tab update: {e}")

                # Refresh GPU dropdowns (Train/Test)
                try:
                    self._refresh_gpu_selectors()
                except Exception as e:
                    self._log_to_app_log(f"[BACKGROUND] Failed to refresh GPU selectors: {e}")

                gpu_count = len(self.system_info.get("cuda", {}).get("gpus", []) or [])
                self._log_to_app_log(f"[BACKGROUND] Detection complete. GPU(s)={gpu_count}")
            finally:
                self._bg_detect_thread = None

        def _on_error(trace: str) -> None:
            self._log_to_app_log("[BACKGROUND] Detection crashed:\n" + trace)
            self._bg_detect_thread = None

        t.detected.connect(_on_detected)
        t.error.connect(_on_error)
        t.finished.connect(lambda: self._log_to_app_log("[BACKGROUND] Detection thread finished."))
        t.start()
    
    def _update_header_system_info(self):
        """Update the header system info labels with detected values"""
        python_info = self.system_info.get("python", {})
        pytorch_info = self.system_info.get("pytorch", {})
        hardware_info = self.system_info.get("hardware", {})

        # Python label
        if hasattr(self, "header_python_label"):
            pyver = python_info.get("version", "Unknown")
            self.header_python_label.setText(f"🐍 Python {pyver}")

        # PyTorch label
        if hasattr(self, "header_pytorch_label"):
            if pytorch_info.get("found"):
                ptver = pytorch_info.get("version", "Unknown")
                if pytorch_info.get("cuda_available"):
                    cuver = pytorch_info.get("cuda_version", "Unknown")
                    self.header_pytorch_label.setText(f"🔥 PyTorch {ptver} (CUDA {cuver})")
                else:
                    self.header_pytorch_label.setText(f"🔥 PyTorch {ptver} (CPU)")
            else:
                self.header_pytorch_label.setText("🔥 PyTorch: Not found")

        # RAM label
        if hasattr(self, "header_ram_label"):
            ram_gb = hardware_info.get("ram_gb", 0) or 0
            # Round to nearest common RAM size for display
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
            self.header_ram_label.setText(f"💾 RAM: {ram_display} GB")

    def _refresh_gpu_selectors(self) -> None:
        """Refresh Train/Test GPU dropdowns based on latest detection results."""
        cuda_info = self.system_info.get("cuda", {})
        gpus = cuda_info.get("gpus", []) or []

        # Train tab GPU selector
        if hasattr(self, "gpu_select") and hasattr(self, "gpu_status_label") and hasattr(self, "training_info_label"):
            self.gpu_select.blockSignals(True)
            self.gpu_select.clear()

            if gpus:
                self.gpu_status_label.setText(f"✅ {len(gpus)} GPU{'s' if len(gpus) > 1 else ''} detected")
                self.gpu_status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
                self.gpu_select.setEnabled(True)
                for idx, gpu in enumerate(gpus):
                    gpu_name = gpu.get("name", f"GPU {idx}")
                    self.gpu_select.addItem(f"GPU {idx}: {gpu_name}")
                self.training_info_label.setText(f"⚡ Training will use: {self.gpu_select.currentText()}")
            else:
                self.gpu_status_label.setText("⚠️ No GPUs detected")
                self.gpu_status_label.setStyleSheet("font-weight: bold; color: #FF9800;")
                self.gpu_select.addItem("No GPUs available - CPU mode")
                self.gpu_select.setEnabled(False)
                self.training_info_label.setText("⚠️ Training will use CPU (slower)")

            self.gpu_select.blockSignals(False)

        # Test tab GPU selector
        if hasattr(self, "test_gpu_select") and hasattr(self, "test_gpu_info"):
            self.test_gpu_select.blockSignals(True)
            self.test_gpu_select.clear()

            if gpus:
                for idx, gpu in enumerate(gpus):
                    gpu_name = gpu.get("name", f"GPU {idx}")
                    vram = gpu.get("memory", "VRAM unknown")
                    self.test_gpu_select.addItem(f"GPU {idx}: {gpu_name} ({vram})")
                self.test_gpu_select.setEnabled(True)
                self.test_gpu_info.setText(f"✅ {len(gpus)} GPU(s) detected - select one for inference")
            else:
                self.test_gpu_select.addItem("No GPUs available - CPU mode")
                self.test_gpu_select.setEnabled(False)
                self.test_gpu_info.setText("⚠️ No GPUs detected (CPU mode)")

            self.test_gpu_select.blockSignals(False)
    
    def _refresh_locals(self) -> None:
        # Refresh downloaded models display
        self._refresh_models()
        
        # Get downloaded models from the models directory (where downloads actually go)
        models_dir = self.root / "models"
        downloaded_models = []
        
        if models_dir.exists():
            for model_dir in sorted(models_dir.iterdir()):
                if model_dir.is_dir():
                    # Check if it has config.json AND actual model weights
                    has_config = (model_dir / "config.json").exists()
                    has_weights = any(
                        (model_dir / f).exists() 
                        for f in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "adapter_model.safetensors", "adapter_model.bin"]
                    )
                    if has_config and (has_weights or len(list(model_dir.glob("*.safetensors"))) > 0 or len(list(model_dir.glob("*.bin"))) > 0):
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
        
        # Add base models from models folder
        if downloaded_models:
            for model_name in downloaded_models:
                # Convert directory name to HuggingFace format (org__model -> org/model)
                display_name = model_name.replace("__", "/")
                self.test_model_a.addItem(f"📦 {display_name}", str(models_dir / model_name))
        
        # Add trained adapters (only if they have actual model weights)
        adapter_dir = self.root / "fine_tuned"
        if adapter_dir.exists():
            # Check for COMPLETE adapters (with model weights, not just config)
            trained_adapters = []
            for d in adapter_dir.iterdir():
                if not d.is_dir():
                    continue
                # Check for actual model weight files
                has_weights = any([
                    (d / "adapter_model.safetensors").exists(),
                    (d / "adapter_model.bin").exists(),
                    (d / "pytorch_model.bin").exists(),
                    (d / "model.safetensors").exists()
                ])
                if has_weights:
                    trained_adapters.append(d)
            
            if trained_adapters:
                for adapter_path in sorted(trained_adapters):
                    adapter_name = adapter_path.name
                    self.test_model_a.addItem(f"🎯 {adapter_name} (adapter)", str(adapter_path))
        
        # Show message if no models available at all
        total_models_a = self.test_model_a.count()
        if total_models_a == 0:
            self.test_model_a.addItem("(No models available - download from Models tab)")
        
        if current_a and current_a != "None":
            idx = self.test_model_a.findText(current_a)
            if idx >= 0:
                self.test_model_a.setCurrentIndex(idx)
        
        # Update Model B dropdown
        current_b = self.test_model_b.currentText()
        self.test_model_b.clear()
        
        # Add base models from models folder
        if downloaded_models:
            for model_name in downloaded_models:
                # Convert directory name to HuggingFace format (org__model -> org/model)
                display_name = model_name.replace("__", "/")
                self.test_model_b.addItem(f"📦 {display_name}", str(models_dir / model_name))
        
        # Add trained adapters (only if they have actual model weights)
        if adapter_dir.exists():
            # Check for COMPLETE adapters (with model weights, not just config)
            trained_adapters = []
            for d in adapter_dir.iterdir():
                if not d.is_dir():
                    continue
                # Check for actual model weight files
                has_weights = any([
                    (d / "adapter_model.safetensors").exists(),
                    (d / "adapter_model.bin").exists(),
                    (d / "pytorch_model.bin").exists(),
                    (d / "model.safetensors").exists()
                ])
                if has_weights:
                    trained_adapters.append(d)
            
            if trained_adapters:
                for adapter_path in sorted(trained_adapters):
                    adapter_name = adapter_path.name
                    self.test_model_b.addItem(f"🎯 {adapter_name} (adapter)", str(adapter_path))
        
        # Show message if no models available at all
        total_models_b = self.test_model_b.count()
        if total_models_b == 0:
            self.test_model_b.addItem("(No models available - download from Models tab)")
        
        if current_b and current_b != "None":
            idx = self.test_model_b.findText(current_b)
            if idx >= 0:
                self.test_model_b.setCurrentIndex(idx)
        
        # Update Model C dropdown (if it exists)
        if hasattr(self, 'test_model_c'):
            current_c = self.test_model_c.currentText()
            self.test_model_c.clear()
            
            # Add base models from models folder
            if downloaded_models:
                for model_name in downloaded_models:
                    # Convert directory name to HuggingFace format (org__model -> org/model)
                    display_name = model_name.replace("__", "/")
                    self.test_model_c.addItem(f"📦 {display_name}", str(models_dir / model_name))
            
            # Add trained adapters (only if they have actual model weights)
            if adapter_dir.exists():
                # Check for COMPLETE adapters (with model weights, not just config)
                trained_adapters = []
                for d in adapter_dir.iterdir():
                    if not d.is_dir():
                        continue
                    # Check for actual model weight files
                    has_weights = any([
                        (d / "adapter_model.safetensors").exists(),
                        (d / "adapter_model.bin").exists(),
                        (d / "pytorch_model.bin").exists(),
                        (d / "model.safetensors").exists()
                    ])
                    if has_weights:
                        trained_adapters.append(d)
                
                if trained_adapters:
                    for adapter_path in sorted(trained_adapters):
                        adapter_name = adapter_path.name
                        self.test_model_c.addItem(f"🎯 {adapter_name} (adapter)", str(adapter_path))
            
            # Show message if no models available at all
            total_models_c = self.test_model_c.count()
            if total_models_c == 0:
                self.test_model_c.addItem("(No models available - download from Models tab)")
            
            if current_c and current_c != "None":
                idx = self.test_model_c.findText(current_c)
                if idx >= 0:
                    self.test_model_c.setCurrentIndex(idx)

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
    # Setup error logging FIRST before anything else
    logs_dir = get_app_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Use timestamped name format: [yymmdd][name][hhmm]
    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m%d")
    time_str = datetime.now().strftime("%H%M")
    session_log_name = f"[{timestamp}][app][{time_str}].log"
    app_log_path = logs_dir / session_log_name
    startup_error_log = logs_dir / f"[{timestamp}][startup_error][{time_str}].log"
    
    def write_startup_error(error_msg: str, traceback_str: str = ""):
        """Write startup error to log file"""
        try:
            with open(startup_error_log, "a", encoding="utf-8") as f:
                from datetime import datetime
                f.write(f"\n{'='*60}\n")
                f.write(f"[{datetime.now()}] STARTUP ERROR\n")
                f.write(f"{'='*60}\n")
                f.write(f"{error_msg}\n")
                if traceback_str:
                    f.write(f"\nTraceback:\n{traceback_str}\n")
                f.write(f"{'='*60}\n\n")
                f.flush()
        except:
            pass
    
    # Optional startup watchdog (only enabled when explicitly requested)
    # Set environment variable LLM_STARTUP_WATCHDOG=1 to enable.
    try:
        import os
        if os.environ.get("LLM_STARTUP_WATCHDOG") == "1":
            import faulthandler
            hang_log_path = logs_dir / "startup_hang.log"
            hang_log = open(hang_log_path, "a", encoding="utf-8", errors="replace")
            hang_log.write("\n==== startup_hang watchdog enabled ====\n")
            hang_log.flush()
            faulthandler.enable(file=hang_log, all_threads=True)
            faulthandler.dump_traceback_later(30, repeat=True, file=hang_log)
    except Exception as e:
        write_startup_error(f"Failed to setup watchdog: {e}")

    try:
        app = QApplication(sys.argv)
        # Set base font size for the entire application - INCREASED to 16pt
        app_font = QFont()
        # Keep this modest; very large fonts require scroll areas (added above).
        app_font.setPointSize(14)
        app.setFont(app_font)
        
        # Show splash screen with minimal info
        splash = SplashScreen()
        splash.show()
        splash.update_progress(10, "Starting up", "Initializing LLM Fine-tuning Studio...")
        app.processEvents()  # Force display
        
        # Create main window quickly (NO detection during init - pass None for splash)
        splash.update_progress(50, "Creating interface", "")
        app.processEvents()
        
        win = MainWindow(splash=None)  # Don't pass splash to skip slow detection
        
        # Show main window IMMEDIATELY
        splash.update_progress(100, "Ready!", "")
        app.processEvents()
        
        # Apply decorative frame overlay if enabled
        if USE_HYBRID_FRAME:
            try:
                # Import frame module
                llm_dir = Path(__file__).parent.parent
                if str(llm_dir) not in sys.path:
                    sys.path.insert(0, str(llm_dir))
                from ui_frame.hybrid_frame import HybridFrameWindow, FrameAssets
                
                # Create frame with assets
                # Assets are in hybrid_frame_module/assets/ (at project root, not in LLM/)
                root_dir = llm_dir.parent  # Go up from LLM/ to project root
                assets_dir = root_dir / "hybrid_frame_module" / "assets"
                
                # Helper function to prefer WebP, fallback to PNG
                def get_asset_path(name: str) -> str | None:
                    webp_path = assets_dir / f"{name}.webp"
                    png_path = assets_dir / f"{name}.png"
                    if webp_path.exists():
                        return str(webp_path)
                    elif png_path.exists():
                        return str(png_path)
                    return None
                
                assets = FrameAssets(
                    corner_tl=get_asset_path("corner_tl"),
                    corner_tr=get_asset_path("corner_tr"),
                    corner_bl=get_asset_path("corner_bl"),
                    corner_br=get_asset_path("corner_br"),
                    top_center=get_asset_path("top_center_owl"),
                )
                
                # Create frame as overlay
                frame = HybridFrameWindow(
                    assets, 
                    corner_size=18, 
                    border_thickness=18, 
                    safe_padding=2,
                    parent_window=win  # Pass MainWindow as parent
                )
                
                # Store reference to frame for cleanup and theme updates
                win._hybrid_frame = frame
                # Store assets_dir and get_asset_path for dynamic corner_br updates
                win._frame_assets_dir = assets_dir
                win._get_frame_asset_path = get_asset_path
                
                # Set initial corner_br image for home page (tab index 0)
                initial_corner_br = get_asset_path("corner_br_owl_coding")
                if initial_corner_br:
                    frame.set_corner_br(initial_corner_br)
                
                # Apply initial theme colors to frame
                colors = win._get_theme_colors()
                from PySide6.QtGui import QColor
                frame_color = QColor(colors["primary"])
                frame_color.setAlpha(220)
                frame_accent = QColor(colors["accent"])
                frame_accent.setAlpha(200)
                # Background color - darker version of primary for solid fill
                bg_color = QColor(colors["primary"])
                bg_color = bg_color.darker(300)  # Much darker for background
                frame.set_frame_colors(frame_color, frame_accent, bg_color)
                
                # Position frame to match MainWindow (with extra space above for center image and right for corner_tr)
                # Also shift outside by half border thickness
                badge_h = int(90 * 0.65)  # Height of center image
                extra_top = badge_h // 2
                extra_right = 75  # Extend right for corner_tr (150px wide, centered at edge = 75px extension)
                shift_out = 18 // 2  # Shift outside by half border thickness (9 pixels)
                frame_geom = win.geometry()
                frame_geom.setX(frame_geom.x() - shift_out)
                frame_geom.setY(frame_geom.y() - extra_top - shift_out)
                frame_geom.setHeight(frame_geom.height() + extra_top + 2 * shift_out)
                frame_geom.setWidth(frame_geom.width() + extra_right + 2 * shift_out)
                frame.setGeometry(frame_geom)
                
                # Show MainWindow first (underneath)
                win.show()
                splash.finish(win)
                
                # Then show frame overlay on top
                frame.show()
                
            except Exception as e:
                import traceback
                error_msg = f"Frame init failed: {e}"
                traceback_str = traceback.format_exc()
                print(error_msg)
                print(traceback_str)
                write_startup_error(error_msg, traceback_str)
                # Fall through to show window without frame
                win.show()
                splash.finish(win)
        else:
            # No frame - show window normally
            win.show()
            splash.finish(win)
        
        # Do system detection in background after GUI is shown (threaded; safe if launched via pythonw)
        QTimer.singleShot(500, lambda: win._start_background_detection())
        
        return app.exec()
    except Exception as e:
        import traceback
        error_msg = f"Fatal startup error: {e}"
        traceback_str = traceback.format_exc()
        print(error_msg)
        print(traceback_str)
        write_startup_error(error_msg, traceback_str)
        # Also write to app.log for launcher to find
        try:
            with open(app_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"FATAL STARTUP ERROR\n")
                f.write(f"{error_msg}\n")
                f.write(f"\nTraceback:\n{traceback_str}\n")
                f.write(f"{'='*60}\n\n")
        except:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
