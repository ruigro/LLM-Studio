from __future__ import annotations

import sys
import os
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer, QThread, Signal, QProcessEnvironment, QRect, QSize
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QTextEdit, QPlainTextEdit,
    QSpinBox, QDoubleSpinBox, QMessageBox, QListWidget, QListWidgetItem, QSplitter, QToolBar, QScrollArea, QGridLayout, QFrame, QProgressBar, QSizePolicy, QTabBar, QStyleOptionTab, QStyle, QStackedWidget, QGroupBox
)
from PySide6.QtGui import QAction, QIcon, QFont, QMouseEvent, QCursor, QPixmap

# Feature flag for hybrid frame wrapper (enabled by default)
# To disable: set USE_HYBRID_FRAME=0 before running
USE_HYBRID_FRAME = os.getenv("USE_HYBRID_FRAME", "1") == "1"

from desktop_app.model_card_widget import ModelCard, DownloadedModelCard
from desktop_app.training_widgets import MetricCard
from desktop_app.chat_widget import ChatWidget
from desktop_app.splash_screen import SplashScreen

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
                # Use the current Python executable (should be venv Python if running from venv)
                python_exe = sys.executable
                self.log_output.emit(f"Using Python: {python_exe}")
                success = installer.repair_all(python_executable=python_exe)
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


class SystemDetectThread(QThread):
    """Thread for running system detection without freezing UI."""
    detected = Signal(dict)        # system_info dict
    error = Signal(str)            # error string

    def run(self):
        try:
            detector = SystemDetector()
            system_info = {
                "python": detector.detect_python(),
                "cuda": detector.detect_cuda(),
                "pytorch": detector.detect_pytorch(),
                "hardware": detector.detect_hardware(),
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
        
        # Model integrity checker
        if splash:
            splash.update_progress(5, "Initializing model checker", "")
        self.model_checker = ModelIntegrityChecker()

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
        header_layout.setContentsMargins(20, 10, 20, 10)
        header_layout.setSpacing(15)
        
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
        
        # Center: App title with owl icon (transparent background)
        title_container = QWidget()
        title_container.setStyleSheet("background: transparent;")
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(12)
        title_layout.setAlignment(Qt.AlignCenter)
        
        # Owl icon - maximum size to fit header (header min height is 80px, use 70px for icon)
        icon_path = self.root.parent / "icons" / "owl_studio_square.png"
        if icon_path.exists():
            icon_pixmap = QPixmap(str(icon_path))
            # Scale icon to maximum size to fit header (70x70 pixels, leaving some margin)
            icon_pixmap = icon_pixmap.scaled(70, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label = QLabel()
            icon_label.setPixmap(icon_pixmap)
            icon_label.setAlignment(Qt.AlignCenter)
            icon_label.setStyleSheet("background: transparent; border: none; padding: 0px;")
            icon_label.setMinimumSize(70, 70)
            icon_label.setMaximumSize(70, 70)
            title_layout.addWidget(icon_label)
        else:
            # Fallback to emoji if icon not found
            icon_label = QLabel("🤖")
            icon_label.setAlignment(Qt.AlignCenter)
            icon_label.setStyleSheet("background: transparent; font-size: 32pt; border: none; padding: 0px;")
            title_layout.addWidget(icon_label)
        
        # Title text (without emoji)
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
        self.requirements_btn = QPushButton("🔧")  # Tool icon only
        self.info_btn = QPushButton("ℹ️ Info")
        
        # Navigation buttons will be styled by theme system
        
        for btn in [self.home_btn, self.train_btn, self.download_btn, self.test_btn, self.logs_btn, self.requirements_btn, self.info_btn]:
            btn.setCheckable(True)
            # Navigation buttons will be styled by theme system
        
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
        # IMPORTANT: store on self for other callbacks (GPU refresh, background detection, etc.)
        self.tabs = tabs
        
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
        if event.button() == Qt.LeftButton:
            edge = self._get_resize_edge(event.position().toPoint())
            if edge:
                self.resize_edge = edge
                self.resize_start_pos = event.globalPosition().toPoint()
                self.resize_start_geometry = self.geometry()
                event.accept()
                return
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for window resizing and cursor changes"""
        pos = event.position().toPoint()
        global_pos = event.globalPosition().toPoint()
        
        # If resizing, handle resize using global coordinates
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
                
                self.setGeometry(geom)
                # Update start position for next move to prevent accumulation
                self.resize_start_pos = global_pos
                self.resize_start_geometry = geom
                event.accept()
                return
        
        # Always check cursor on mouse move (even when not resizing)
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
        else:
            # Not on edge - restore normal cursor
            if self.cursor_override_active:
                QApplication.restoreOverrideCursor()
                self.cursor_override_active = False
                self.current_cursor_shape = None
            # Also restore widget cursor
            self.unsetCursor()
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release to stop resizing"""
        if event.button() == Qt.LeftButton:
            self.resize_edge = None
            self.resize_start_pos = None
            self.resize_start_geometry = None
            if self.cursor_override_active:
                QApplication.restoreOverrideCursor()
                self.cursor_override_active = False
        super().mouseReleaseEvent(event)
    
    def eventFilter(self, obj, event) -> bool:
        """Event filter to catch mouse events for window resizing from child widgets"""
        if isinstance(event, QMouseEvent):
            # Always use global position for accurate calculations
            global_pos = event.globalPosition().toPoint()
            # Convert to window coordinates for edge detection
            window_pos = self.mapFromGlobal(global_pos)
            
            # Handle mouse enter to ensure cursor updates
            if event.type() == QMouseEvent.Type.Enter:
                # When mouse enters a child widget, check if we're near an edge
                if not self.resize_edge:
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
                return False  # Let event propagate
            
            # Handle mouse leave to restore cursor
            if event.type() == QMouseEvent.Type.Leave:
                if self.cursor_override_active and not self.resize_edge:
                    QApplication.restoreOverrideCursor()
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
                        
                        self.setGeometry(geom)
                        # Update start position for next move to prevent accumulation
                        self.resize_start_pos = global_pos
                        self.resize_start_geometry = geom
                        return True  # Consume event
                
                # Always check cursor on mouse move (even when not resizing)
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
                    # Return False to let event propagate, but cursor is set
                else:
                    # Restore cursor when not on edge
                    if self.cursor_override_active:
                        QApplication.restoreOverrideCursor()
                        self.cursor_override_active = False
                        self.current_cursor_shape = None
                    # Also restore widget cursor
                    self.unsetCursor()
            
            # Handle mouse press for resize start
            elif event.type() == QMouseEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
                edge = self._get_resize_edge(window_pos)
                if edge:
                    self.resize_edge = edge
                    self.resize_start_pos = global_pos
                    self.resize_start_geometry = self.geometry()
                    return True  # Consume event to start resize
            
            # Handle mouse release
            elif event.type() == QMouseEvent.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self.resize_edge:
                    self.resize_edge = None
                    self.resize_start_pos = None
                    self.resize_start_geometry = None
                    # Check current position and set cursor accordingly
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
                            if self.cursor_override_active:
                                QApplication.restoreOverrideCursor()
                            QApplication.setOverrideCursor(QCursor(cursor_shape))
                            self.cursor_override_active = True
                            self.current_cursor_shape = cursor_shape
                    else:
                        if self.cursor_override_active:
                            QApplication.restoreOverrideCursor()
                            self.cursor_override_active = False
                            self.current_cursor_shape = None
                    return True  # Consume event
        
        return super().eventFilter(obj, event)
    
    def _header_mouse_press(self, event: QMouseEvent) -> None:
        """Handle mouse press on header for window dragging"""
        if event.button() == Qt.LeftButton:
            # Check for top edge FIRST before allowing drag (to prioritize resize)
            # Convert header widget position to window coordinates
            header_pos = self.header_widget.mapTo(self, event.position().toPoint())
            edge = self._get_resize_edge(header_pos)
            # If on top edge (or any edge), don't set drag_position - let resize handler take over
            if not edge:
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
            # If on edge, don't accept event - let event filter handle resize
    
    def _header_mouse_move(self, event: QMouseEvent) -> None:
        """Handle mouse move on header for window dragging"""
        if event.buttons() == Qt.LeftButton and self.drag_position is not None:
            # Check if we're currently resizing FIRST - if so, don't handle drag
            if not self.resize_edge:
                # Also check if we're on an edge now (might have moved to edge)
                header_pos = self.header_widget.mapTo(self, event.position().toPoint())
                edge = self._get_resize_edge(header_pos)
                if not edge:
                    self.move(event.globalPosition().toPoint() - self.drag_position)
                    event.accept()
            # If resizing, don't handle drag - let resize take precedence
    
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
        
        # Update HybridFrameWindow if it exists (when using hybrid frame)
        if hasattr(self, '_hybrid_frame') and self._hybrid_frame is not None:
            self._hybrid_frame.setStyleSheet(stylesheet)
            # Also update the content container
            if hasattr(self._hybrid_frame, 'content_container'):
                self._hybrid_frame.content_container.setStyleSheet(stylesheet)
    
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
        gpus = cuda_info.get("gpus", [])
        
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
                
                gpu_label = QLabel(f"<span style='font-weight: bold; text-decoration: none; border: none; border-bottom: none;'>GPU {idx}:</span> <span style='text-decoration: none; border: none; border-bottom: none;'>{gpu_name}</span>")
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
        main_layout = QHBoxLayout(w)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create splitter for 2 columns
        splitter = QSplitter(Qt.Horizontal)
        
        # LEFT COLUMN: Curated Models
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        
        curated_label = QLabel("📚 Curated Models for Fine-tuning")
        curated_label.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
        font = curated_label.font()
        font.setUnderline(False)
        curated_label.setFont(font)
        left_layout.addWidget(curated_label)
        
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
        downloaded_label = QLabel("📥 Downloaded Models")
        downloaded_label.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
        font = downloaded_label.font()
        font.setUnderline(False)
        downloaded_label.setFont(font)
        right_layout.addWidget(downloaded_label)
        
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
        search_label = QLabel("🔍 Search Hugging Face")
        search_label.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none; border: none; border-bottom: none;")
        font = search_label.font()
        font.setUnderline(False)
        search_label.setFont(font)
        right_layout.addWidget(search_label)
        
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

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        
        # Set UTF-8 encoding for Windows to handle emojis in transformers/unsloth output
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        
        # Set GPU selection from Train tab dropdown
        if hasattr(self, 'gpu_select') and self.gpu_select.isEnabled():
            selected_gpu_idx = self.gpu_select.currentIndex()
            env.insert("CUDA_VISIBLE_DEVICES", str(selected_gpu_idx))
            self.train_log.appendPlainText(f"[INFO] Using GPU {selected_gpu_idx}: {self.gpu_select.currentText()}")
        
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
        test_title = QLabel("🧪 Test Models - Side-by-Side Chat")
        test_title.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none;")
        layout.addWidget(test_title)
        
        # GPU selection for inference
        gpu_frame = QGroupBox("⚙️ Hardware Settings")
        gpu_layout = QVBoxLayout(gpu_frame)
        
        cuda_info = self.system_info.get("cuda", {})
        gpus = cuda_info.get("gpus", [])
        
        # GPU selection dropdown for test tab
        gpu_row = QHBoxLayout()
        gpu_row.addWidget(QLabel("GPU for Inference:"))
        self.test_gpu_select = QComboBox()
        
        if gpus:
            for idx, gpu in enumerate(gpus):
                gpu_name = gpu.get("name", f"GPU {idx}")
                vram = gpu.get("memory", "N/A")
                self.test_gpu_select.addItem(f"GPU {idx}: {gpu_name} ({vram})")
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
        
        layout.addWidget(gpu_frame)

        # Side-by-side model comparison (TOP - Chat)
        models_layout = QHBoxLayout()
        models_layout.setSpacing(20)
        
        # MODEL A (Left)
        model_a_widget = QWidget()
        model_a_layout = QVBoxLayout(model_a_widget)
        model_a_layout.setSpacing(10)
        
        # Header
        header_a = QLabel("🔵 <b>Model A</b>")
        header_a.setObjectName("modelAHeader")
        colors = self._get_theme_colors()
        header_a.setStyleSheet(f"font-size: 16pt; padding: 10px; background: {self._get_gradient_style(colors['primary'], colors['secondary'])}; color: white; border-radius: 6px;")
        self.themed_widgets["labels"].append(header_a)
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
        header_b.setObjectName("modelBHeader")
        colors = self._get_theme_colors()
        header_b.setStyleSheet(f"font-size: 16pt; padding: 10px; background: {self._get_gradient_style(colors['primary'], colors['secondary'])}; color: white; border-radius: 6px;")
        self.themed_widgets["labels"].append(header_b)
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
        
        # Check if this is an adapter or base model
        from pathlib import Path
        model_path_obj = Path(model_path)
        is_adapter = (model_path_obj / "adapter_config.json").exists() or \
                    (model_path_obj / "adapter_model.safetensors").exists() or \
                    (model_path_obj / "adapter_model.bin").exists()
        
        # Detect if this is an instruct model (check if "instruct" is in the path)
        model_path_lower = str(model_path).lower()
        is_instruct = "instruct" in model_path_lower or "chat" in model_path_lower
        model_type = "instruct" if is_instruct else "base"
        
        # Build command - use run_adapter.py for both base models and adapters
        cmd = [
            sys.executable, "-u", "run_adapter.py",
            "--prompt", prompt,
            "--max-new-tokens", "512",
            "--temperature", "0.7",
            "--model-type", model_type
        ]
        
        if is_adapter:
            # Load as adapter (requires base model + adapter)
            cmd += ["--adapter-dir", model_path]
            # TODO: Need to specify base model - for now use default
            cmd += ["--base-model", "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"]
        else:
            # Load as base model only
            cmd += ["--base-model", model_path, "--no-adapter"]
        
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
        
        # Extract only the actual output (after "--- OUTPUT ---")
        if "--- OUTPUT ---" in self.inference_buffer_a:
            # Split and take everything after the marker
            parts = self.inference_buffer_a.split("--- OUTPUT ---", 1)
            if len(parts) > 1:
                actual_output = parts[1].strip()
                if actual_output:
                    self.chat_widget_a.update_last_ai_message(actual_output)
        else:
            # Before we see OUTPUT marker, check if there's any useful partial response
            # Filter out all log lines (lines starting with [INFO], [WARN], [OK], etc.)
            lines = self.inference_buffer_a.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Skip log messages and technical output
                if any(x in line for x in [
                    '[INFO]', '[WARN]', '[OK]', '[ERROR]',
                    'FutureWarning', 'UserWarning', 'TRANSFORMERS_CACHE',
                    'warnings.warn', 'DeprecationWarning', 'Loading tokenizer',
                    'Loading base model', 'Windows detected', 'Generating...',
                    'Loading checkpoint shards:', '|', 'it/s', '█', '▌'
                ]):
                    continue
                # Keep everything else
                if line.strip():
                    filtered_lines.append(line)
            
            # Only update if we have filtered content
            if filtered_lines:
                clean_response = '\n'.join(filtered_lines).strip()
                if clean_response:
                    self.chat_widget_a.update_last_ai_message(clean_response)
    
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
        
        # Check if this is an adapter or base model
        from pathlib import Path
        model_path_obj = Path(model_path)
        is_adapter = (model_path_obj / "adapter_config.json").exists() or \
                    (model_path_obj / "adapter_model.safetensors").exists() or \
                    (model_path_obj / "adapter_model.bin").exists()
        
        # Detect if this is an instruct model (check if "instruct" is in the path)
        model_path_lower = str(model_path).lower()
        is_instruct = "instruct" in model_path_lower or "chat" in model_path_lower
        model_type = "instruct" if is_instruct else "base"
        
        # Build command - use run_adapter.py for both base models and adapters
        cmd = [
            sys.executable, "-u", "run_adapter.py",
            "--prompt", prompt,
            "--max-new-tokens", "512",
            "--temperature", "0.7",
            "--model-type", model_type
        ]
        
        if is_adapter:
            # Load as adapter (requires base model + adapter)
            cmd += ["--adapter-dir", model_path]
            # TODO: Need to specify base model - for now use default
            cmd += ["--base-model", "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"]
        else:
            # Load as base model only
            cmd += ["--base-model", model_path, "--no-adapter"]
        
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
        
        # Extract only the actual output (after "--- OUTPUT ---")
        if "--- OUTPUT ---" in self.inference_buffer_b:
            # Split and take everything after the marker
            parts = self.inference_buffer_b.split("--- OUTPUT ---", 1)
            if len(parts) > 1:
                actual_output = parts[1].strip()
                if actual_output:
                    self.chat_widget_b.update_last_ai_message(actual_output)
        else:
            # Before we see OUTPUT marker, check if there's any useful partial response
            # Filter out all log lines (lines starting with [INFO], [WARN], [OK], etc.)
            lines = self.inference_buffer_b.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Skip log messages and technical output
                if any(x in line for x in [
                    '[INFO]', '[WARN]', '[OK]', '[ERROR]',
                    'FutureWarning', 'UserWarning', 'TRANSFORMERS_CACHE',
                    'warnings.warn', 'DeprecationWarning', 'Loading tokenizer',
                    'Loading base model', 'Windows detected', 'Generating...',
                    'Loading checkpoint shards:', '|', 'it/s', '█', '▌'
                ]):
                    continue
                # Keep everything else
                if line.strip():
                    filtered_lines.append(line)
            
            # Only update if we have filtered content
            if filtered_lines:
                clean_response = '\n'.join(filtered_lines).strip()
                if clean_response:
                    self.chat_widget_b.update_last_ai_message(clean_response)
    
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
        """Build Requirements tab showing required vs installed versions (source of truth)"""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("🔧 Required Packages & Versions")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background: transparent; color: white; font-size: 24pt; font-weight: bold; text-decoration: none;")
        layout.addWidget(title)
        
        # Info text
        info = QLabel("This is the source of truth showing required vs installed versions:")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("background: transparent; color: #888; font-size: 11pt; margin-bottom: 20px;")
        layout.addWidget(info)
        
        # IMPORTANT: This tab must be non-blocking.
        # Do NOT call SmartInstaller.get_installation_checklist() here because it runs subprocess checks and can freeze UI.
        
        # Get installed versions using target Python
        try:
            from importlib.metadata import version, PackageNotFoundError
        except ImportError:
            from importlib_metadata import version, PackageNotFoundError
        
        # Parse requirements.txt
        requirements_file = Path(__file__).parent.parent / "requirements.txt"
        required_packages = {}
        if requirements_file.exists():
            with open(requirements_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # Parse package name and version spec
                    # Handle formats like: "package==1.0.0", "package>=1.0.0", "package>=1.0.0,<2.0.0"
                    parts = line.split(';')
                    pkg_line = parts[0].strip()
                    # Extract package name and version
                    if '>=' in pkg_line or '==' in pkg_line or '<=' in pkg_line or '!=' in pkg_line or '<' in pkg_line or '>' in pkg_line:
                        # Has version spec
                        import re
                        match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)(.*)$', pkg_line)
                        if match:
                            pkg_name = match.group(1).split('[')[0]  # Remove extras like package[extra]
                            version_spec = match.group(2).strip()
                            required_packages[pkg_name] = version_spec
                        else:
                            # Fallback: just use the line as-is
                            pkg_name = pkg_line.split()[0]
                            required_packages[pkg_name] = ""
                    else:
                        # No version spec
                        required_packages[pkg_line] = ""
        
        # Add PyTorch packages (from SmartInstaller, not in requirements.txt)
        required_packages["torch"] = "==2.5.1+cu118"
        required_packages["torchvision"] = "==0.20.1+cu118"
        required_packages["torchaudio"] = "==2.5.1+cu118"
        required_packages["triton-windows"] = "==3.0.0"  # Note: Windows uses triton-windows but module is triton
        required_packages["PySide6"] = "==6.8.1"
        required_packages["PySide6-Essentials"] = "==6.8.1"
        required_packages["PySide6-Addons"] = "==6.8.1"
        required_packages["shiboken6"] = "==6.8.1"
        
        # Package list in scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(15)
        
        # Organize packages into categories with descriptions
        categories = {
            "Core ML Libraries": ["numpy", "transformers", "tokenizers", "accelerate", "datasets", "peft", "safetensors"],
            "Tokenization": ["sentencepiece"],
            "Quantization & Optimization": ["bitsandbytes"],
            "Utilities": ["evaluate", "filelock", "tqdm"],
            "GUI & Visualization": ["streamlit", "pandas", "PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6"],
            "Hugging Face Integration": ["huggingface_hub"],
            "Vision Model Support": ["psutil", "timm", "einops", "open-clip-torch", "Pillow"],
            "PyTorch Ecosystem (SmartInstaller)": ["torch", "torchvision", "torchaudio", "triton-windows"],
            "Fast Fine-tuning": ["unsloth"],
        }
        
        descriptions = {
            "numpy": "Pinned for Windows/PyTorch compatibility",
            "transformers": "Required by unsloth 2025.12.9 (min version: >=4.51.3)",
            "tokenizers": "Required by transformers 4.51.3",
            "accelerate": "Distributed training",
            "datasets": "Dataset loading (upper bound per unsloth requirements)",
            "peft": "LoRA and efficient training",
            "safetensors": "Safe model serialization",
            "sentencepiece": "Text tokenization",
            "bitsandbytes": "4-bit/8-bit quantization (Python 3.8+)",
            "evaluate": "Model evaluation metrics",
            "filelock": "File locking",
            "tqdm": "Progress bars",
            "streamlit": "Web interface (optional)",
            "pandas": "Data manipulation",
            "PySide6": "Qt desktop GUI (ALL components must match)",
            "PySide6-Essentials": "PySide6 essentials (must match PySide6 version)",
            "PySide6-Addons": "PySide6 addons (must match PySide6 version)",
            "shiboken6": "PySide6 binding generator (must match PySide6 version)",
            "huggingface_hub": "Model download/upload",
            "psutil": "System monitoring",
            "timm": "Vision models",
            "einops": "Tensor operations",
            "open-clip-torch": "CLIP models",
            "Pillow": "Image processing",
            "torch": "Deep learning framework (CUDA 11.8)",
            "torchvision": "Computer vision",
            "torchaudio": "Audio processing",
            "triton-windows": "GPU programming (Windows)",
            "unsloth": "2x faster LLM fine-tuning",
        }
        
        # Version comparison helper
        try:
            from packaging import version as pkg_version
            from packaging.specifiers import SpecifierSet
            has_packaging = True
        except ImportError:
            has_packaging = False
        
        def check_version_match(installed_ver, required_spec):
            """Check if installed version matches required spec"""
            if not installed_ver or not required_spec:
                return None  # Can't check
            if not has_packaging:
                # Simple check for == cases
                if required_spec.startswith("=="):
                    expected = required_spec[2:].strip()
                    return installed_ver == expected
                return None
            try:
                installed = pkg_version.parse(installed_ver)
                spec = SpecifierSet(required_spec)
                return installed in spec
            except:
                return None
        
        for category, package_names in categories.items():
            # Category header
            cat_label = QLabel(category)
            cat_label.setStyleSheet("background: transparent; color: #667eea; font-size: 16pt; font-weight: bold; text-decoration: none;")
            content_layout.addWidget(cat_label)
            
            # Packages in this category
            for pkg_name in package_names:
                if pkg_name not in required_packages:
                    continue
                
                required_version = required_packages[pkg_name]
                
                # Get installed version quickly from current environment (no subprocess)
                installed_version = None
                dist_candidates = [pkg_name]
                if pkg_name == "triton-windows":
                    dist_candidates = ["triton", "triton-windows"]
                for dist in dist_candidates:
                    try:
                        installed_version = version(dist)
                        break
                    except PackageNotFoundError:
                        continue
                    except Exception:
                        continue

                is_installed = bool(installed_version)
                
                # Determine status and color
                status = "unknown"
                status_color = "#888"
                status_text = ""
                
                if not is_installed:
                    status = "missing"
                    status_color = "#f44336"  # Red
                    status_text = "❌ NOT INSTALLED"
                elif installed_version and required_version:
                    matches = check_version_match(installed_version, required_version)
                    if matches is True:
                        status = "ok"
                        status_color = "#4CAF50"  # Green
                        status_text = "✅ OK"
                    elif matches is False:
                        status = "mismatch"
                        status_color = "#FF9800"  # Yellow/Orange
                        status_text = "⚠️ VERSION MISMATCH"
                    else:
                        status = "unknown"
                        status_color = "#888"
                        status_text = "❓ CANNOT VERIFY"
                elif installed_version:
                    status = "installed"
                    status_color = "#4CAF50"  # Green (installed but no version requirement)
                    status_text = "✅ INSTALLED"
                else:
                    status = "unknown"
                    status_color = "#888"
                    status_text = "❓ UNKNOWN"
                
                pkg_frame = QFrame()
                pkg_frame.setFrameShape(QFrame.StyledPanel)
                pkg_frame.setStyleSheet(f"""
                    QFrame {{
                        background: rgba(60, 60, 80, 0.3);
                        border: 2px solid {status_color};
                        border-radius: 6px;
                        padding: 8px;
                    }}
                """)
                pkg_layout = QVBoxLayout(pkg_frame)
                pkg_layout.setSpacing(4)
                
                # Package name and status
                name_status = QLabel(f"<b>{pkg_name}</b> <span style='color: {status_color};'>{status_text}</span>")
                name_status.setStyleSheet("background: transparent; font-size: 12pt; color: white;")
                pkg_layout.addWidget(name_status)
                
                # Required version
                req_label = QLabel(f"<b>Required:</b> {required_version if required_version else 'any'}")
                req_label.setStyleSheet("background: transparent; font-size: 10pt; color: #aaa;")
                pkg_layout.addWidget(req_label)
                
                # Installed version
                if is_installed and installed_version:
                    inst_label = QLabel(f"<b>Installed:</b> {installed_version}")
                    inst_label.setStyleSheet(f"background: transparent; font-size: 10pt; color: {status_color};")
                    pkg_layout.addWidget(inst_label)
                elif is_installed:
                    inst_label = QLabel("<b>Installed:</b> (version unknown)")
                    inst_label.setStyleSheet("background: transparent; font-size: 10pt; color: #888;")
                    pkg_layout.addWidget(inst_label)
                else:
                    inst_label = QLabel("<b>Installed:</b> NOT INSTALLED")
                    inst_label.setStyleSheet("background: transparent; font-size: 10pt; color: #f44336;")
                    pkg_layout.addWidget(inst_label)
                
                # Description
                if pkg_name in descriptions:
                    desc = QLabel(descriptions[pkg_name])
                    desc.setStyleSheet("background: transparent; font-size: 9pt; color: #888;")
                    desc.setWordWrap(True)
                    pkg_layout.addWidget(desc)
                
                content_layout.addWidget(pkg_frame)
        
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        # Last updated timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_label = QLabel(f"<i>Last updated: {timestamp}</i>")
        timestamp_label.setAlignment(Qt.AlignCenter)
        timestamp_label.setStyleSheet("background: transparent; color: #666; font-size: 9pt; margin-top: 10px;")
        layout.addWidget(timestamp_label)
        
        return w
    
    def _build_info_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("ℹ️ About LLM Fine-tuning Studio")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24pt; font-weight: bold; text-decoration: none;")
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
        
        credits_title = QLabel("💝 Credits")
        credits_title.setStyleSheet("font-size: 18pt; font-weight: bold; text-decoration: none;")
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

    def _background_system_check(self):
        """Backward-compatible entrypoint (kept for older call sites)."""
        self._start_background_detection()

    def _log_to_app_log(self, message: str) -> None:
        """Best-effort logger for pythonw launches (no console)."""
        try:
            logs_dir = self.root / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "app.log"
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

                # Refresh Home tab (System Status + Requirements) to show real detected values
                try:
                    if hasattr(self, "tabs") and self.tabs is not None and self.tabs.count() > 0:
                        current_index = self.tabs.currentIndex()
                        self.tabs.removeTab(0)
                        self.tabs.insertTab(0, self._build_home_tab(), "🏠 Home")
                        self.tabs.setCurrentIndex(current_index)
                except Exception as e:
                    self._log_to_app_log(f"[BACKGROUND] Failed to rebuild Home tab: {e}")

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
        
        # Add base models from models folder
        if downloaded_models:
            for model_name in downloaded_models:
                self.test_model_a.addItem(f"📦 {model_name}", str(models_dir / model_name))
        
        # Also check hf_models folder for downloaded base models
        hf_models_dir = self.root / "hf_models"
        if hf_models_dir.exists():
            hf_downloaded = sorted([d.name for d in hf_models_dir.iterdir() if d.is_dir()])
            for model_name in hf_downloaded:
                # Convert directory name to HuggingFace format (org__model -> org/model)
                display_name = model_name.replace("__", "/")
                self.test_model_a.addItem(f"📦 {display_name}", str(hf_models_dir / model_name))
        
        # Add trained adapters (only if they have actual model weights)
        adapter_dir = self.root / "fine_tuned_adapter"
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
        total_models_a = self.test_model_a.count() - 1  # Exclude "None" item
        if total_models_a == 0:
            self.test_model_a.addItem("(No models available - download from Models tab)")
        
        if current_a and current_a != "None":
            idx = self.test_model_a.findText(current_a)
            if idx >= 0:
                self.test_model_a.setCurrentIndex(idx)
        
        # Update Model B dropdown
        current_b = self.test_model_b.currentText()
        self.test_model_b.clear()
        self.test_model_b.addItem("None")
        
        # Add base models from models folder
        if downloaded_models:
            for model_name in downloaded_models:
                self.test_model_b.addItem(f"📦 {model_name}", str(models_dir / model_name))
        
        # Also check hf_models folder for downloaded base models
        if hf_models_dir.exists():
            hf_downloaded = sorted([d.name for d in hf_models_dir.iterdir() if d.is_dir()])
            for model_name in hf_downloaded:
                # Convert directory name to HuggingFace format (org__model -> org/model)
                display_name = model_name.replace("__", "/")
                self.test_model_b.addItem(f"📦 {display_name}", str(hf_models_dir / model_name))
        
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
        total_models_b = self.test_model_b.count() - 1  # Exclude "None" item
        if total_models_b == 0:
            self.test_model_b.addItem("(No models available - download from Models tab)")
        
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
    # Optional startup watchdog (only enabled when explicitly requested)
    # Set environment variable LLM_STARTUP_WATCHDOG=1 to enable.
    try:
        import os
        if os.environ.get("LLM_STARTUP_WATCHDOG") == "1":
            import faulthandler
            logs_dir = get_app_root() / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            hang_log_path = logs_dir / "startup_hang.log"
            hang_log = open(hang_log_path, "a", encoding="utf-8", errors="replace")
            hang_log.write("\n==== startup_hang watchdog enabled ====\n")
            hang_log.flush()
            faulthandler.enable(file=hang_log, all_threads=True)
            faulthandler.dump_traceback_later(30, repeat=True, file=hang_log)
    except Exception:
        pass

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
    
    # Apply decorative frame wrapper if enabled
    if USE_HYBRID_FRAME:
        try:
            # Import frame module
            llm_dir = Path(__file__).parent.parent
            if str(llm_dir) not in sys.path:
                sys.path.insert(0, str(llm_dir))
            from ui_frame.hybrid_frame import HybridFrameWindow, FrameAssets
            
            # Extract ONLY central widget
            central = win.takeCentralWidget()
            if central is None:
                raise ValueError("No central widget")
            
            # CRITICAL: Apply the MainWindow's stylesheet to preserve styling
            # Get the theme stylesheet from MainWindow (dark mode by default)
            from desktop_app.main import get_theme_stylesheet
            theme_stylesheet = get_theme_stylesheet(win.dark_mode, win.color_theme)
            
            # Create frame with assets
            assets_dir = llm_dir / "LLM" / "ui_frame" / "assets"
            assets = FrameAssets(
                corner_tl=str(assets_dir / "corner_tl.png") if (assets_dir / "corner_tl.png").exists() else None,
                corner_tr=str(assets_dir / "corner_tr.png") if (assets_dir / "corner_tr.png").exists() else None,
                corner_bl=str(assets_dir / "corner_bl.png") if (assets_dir / "corner_bl.png").exists() else None,
                corner_br=str(assets_dir / "corner_br.png") if (assets_dir / "corner_br.png").exists() else None,
                top_center=str(assets_dir / "top_center.png") if (assets_dir / "top_center.png").exists() else None,
            )
            # Smaller frame decorations
            frame = HybridFrameWindow(assets, corner_size=15, border_thickness=4, safe_padding=2)
            
            # Apply the theme stylesheet to the frame so it matches MainWindow
            frame.setStyleSheet(theme_stylesheet)
            
            # Mount widget and setup
            frame.set_content_widget(central)
            
            # Apply stylesheet to central widget to preserve all styling
            central.setStyleSheet(theme_stylesheet)
            
            frame.setWindowTitle(win.windowTitle())
            frame.resize(win.size())
            frame.setMinimumSize(win.minimumSize())
            
            # Hide MainWindow, show only frame
            win.hide()
            frame.show()
            splash.finish(frame)
            
            # Keep MainWindow alive for methods and store frame reference for theme updates
            frame._main_window = win
            win._hybrid_frame = frame  # Store reference so theme updates can update the frame
            
            # Make MainWindow's close button close the frame
            def close_frame():
                frame.close()
            win.close_btn.clicked.disconnect()  # Disconnect existing handler
            win.close_btn.clicked.connect(close_frame)
            
            QTimer.singleShot(0, lambda: win._start_background_detection())
            
            return app.exec()
        except Exception as e:
            print(f"Frame init failed: {e}, using standard window")
            # Fall through to original path
    
    # Original path (unchanged)
    win.show()
    splash.finish(win)
    
    # Do system detection in background after GUI is shown (threaded; safe if launched via pythonw)
    QTimer.singleShot(500, lambda: win._start_background_detection())
    
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
