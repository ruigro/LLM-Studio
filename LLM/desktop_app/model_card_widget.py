"""Custom widget for displaying model cards with rich visual design"""
from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class ModelCard(QFrame):
    """Beautiful large card widget matching Streamlit design"""
    
    download_clicked = Signal(str)  # Emits model ID
    
    def __init__(self, model_name: str, model_id: str, description: str, size: str, 
                 icons: str, is_downloaded: bool = False, is_new: bool = False, parent=None):
        super().__init__(parent)
        self.model_id = model_id
        self.is_downloaded = is_downloaded
        self.is_dark = True
        
        self.setMinimumHeight(160)
        self.setMaximumHeight(180)
        self.setFrameShape(QFrame.Box)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(8)
        
        # Top row: Model name + NEW badge
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)
        
        name_label = QLabel(model_name)
        name_font = QFont()
        name_font.setPointSize(12)
        name_font.setBold(True)
        name_label.setFont(name_font)
        name_label.setWordWrap(False)
        top_layout.addWidget(name_label)
        
        if is_new:
            new_badge = QLabel("NEW")
            new_badge.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff6b6b, stop:1 #ee5a6f);
                    color: white;
                    padding: 3px 8px;
                    border-radius: 4px;
                    font-size: 9px;
                    font-weight: bold;
                }
            """)
            new_badge.setMaximumWidth(50)
            top_layout.addWidget(new_badge)
        
        top_layout.addStretch(1)
        layout.addLayout(top_layout)
        
        # Description
        if description:
            desc_label = QLabel(description)
            desc_font = QFont()
            desc_font.setPointSize(9)
            desc_label.setFont(desc_font)
            desc_label.setWordWrap(True)
            desc_label.setMaximumHeight(40)
            layout.addWidget(desc_label)
        
        # Size and icons row
        middle_layout = QHBoxLayout()
        
        size_label = QLabel(f"📦 {size}")
        size_font = QFont()
        size_font.setPointSize(9)
        size_label.setFont(size_font)
        middle_layout.addWidget(size_label)
        
        middle_layout.addStretch(1)
        
        icons_label = QLabel(icons)
        icons_font = QFont()
        icons_font.setPointSize(16)
        icons_label.setFont(icons_font)
        middle_layout.addWidget(icons_label)
        
        layout.addLayout(middle_layout)
        
        # Model ID
        id_label = QLabel(f"📂 {model_id}")
        id_font = QFont()
        id_font.setPointSize(8)
        id_label.setFont(id_font)
        id_label.setStyleSheet("color: #888;")
        layout.addWidget(id_label)
        
        layout.addStretch(1)
        
        # Download button
        self.download_btn = QPushButton("📥 Download" if not is_downloaded else "✓ Downloaded")
        self.download_btn.setEnabled(not is_downloaded)
        self.download_btn.clicked.connect(lambda: self.download_clicked.emit(model_id))
        layout.addWidget(self.download_btn)
        
        self._apply_style()
    
    def _apply_style(self):
        """Apply card styling"""
        if self.is_dark:
            self.setStyleSheet("""
                ModelCard {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                                stop:0 #1e1e2e, stop:1 #16213e);
                    border: 2px solid;
                    border-radius: 10px;
                    border-color: #667eea;
                }
                ModelCard:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                                stop:0 #262740, stop:1 #1a2540);
                    border-color: #764ba2;
                }
                QLabel {
                    background: transparent;
                    color: #fafafa;
                    border: none;
                }
            """)
        else:
            self.setStyleSheet("""
                ModelCard {
                    background: white;
                    border: 2px solid #e0e0e0;
                    border-radius: 10px;
                }
                ModelCard:hover {
                    border-color: #667eea;
                }
                QLabel {
                    background: transparent;
                    color: #262730;
                    border: none;
                }
            """)
    
    def set_theme(self, dark_mode: bool):
        """Update theme"""
        self.is_dark = dark_mode
        self._apply_style()


class DownloadedModelCard(QFrame):
    """Card for already downloaded models"""
    
    selected = Signal(str)
    
    def __init__(self, model_name: str, model_path: str, size: str, icons: str, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.is_dark = True
        
        self.setMinimumHeight(100)
        self.setMaximumHeight(110)
        self.setFrameShape(QFrame.Box)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(6)
        
        # Model name
        name_label = QLabel(model_name)
        name_font = QFont()
        name_font.setPointSize(10)
        name_font.setBold(True)
        name_label.setFont(name_font)
        name_label.setWordWrap(True)
        layout.addWidget(name_label)
        
        # Size and icons
        bottom_layout = QHBoxLayout()
        
        size_label = QLabel(f"📦 {size}")
        size_font = QFont()
        size_font.setPointSize(9)
        size_label.setFont(size_font)
        bottom_layout.addWidget(size_label)
        
        bottom_layout.addStretch(1)
        
        icons_label = QLabel(icons)
        icons_font = QFont()
        icons_font.setPointSize(14)
        icons_label.setFont(icons_font)
        bottom_layout.addWidget(icons_label)
        
        layout.addLayout(bottom_layout)
        
        self._apply_style()
    
    def _apply_style(self):
        if self.is_dark:
            self.setStyleSheet("""
                DownloadedModelCard {
                    background: #1a1a2e;
                    border: 1px solid #667eea;
                    border-radius: 8px;
                }
                DownloadedModelCard:hover {
                    background: #262740;
                    border: 2px solid #764ba2;
                }
                QLabel {
                    background: transparent;
                    color: #fafafa;
                    border: none;
                }
            """)
        else:
            self.setStyleSheet("""
                DownloadedModelCard {
                    background: #f8f9fa;
                    border: 1px solid #d0d0d0;
                    border-radius: 8px;
                }
                DownloadedModelCard:hover {
                    background: #e8e9ea;
                    border: 2px solid #667eea;
                }
                QLabel {
                    background: transparent;
                    color: #262730;
                    border: none;
                }
            """)
    
    def set_theme(self, dark_mode: bool):
        self.is_dark = dark_mode
        self._apply_style()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selected.emit(self.model_path)
        super().mousePressEvent(event)


