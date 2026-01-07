"""Custom widget for displaying model cards with rich visual design"""
from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class ModelCard(QFrame):
    """Beautiful large card widget matching Streamlit design"""
    
    download_clicked = Signal(str)  # Emits model ID
    
    def __init__(self, model_name: str, model_id: str, description: str, size: str, 
                 icons: str, is_downloaded: bool = False, is_new: bool = False, 
                 downloads: str = "", likes: str = "", author_icon: str = "", parent=None):
        super().__init__(parent)
        self.model_id = model_id
        self.is_downloaded = is_downloaded
        self.is_dark = True
        
        self.setMinimumHeight(220)
        # Remove setMaximumHeight to allow automatic sizing based on content
        self.setFrameShape(QFrame.Box)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(8)
        
        # Header layout (Icon + Title + Stats)
        header_layout = QHBoxLayout()
        header_layout.setSpacing(15)
        
        # Author/Model Icon
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(50, 50)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self._set_model_icon(model_id, author_icon)
        header_layout.addWidget(self.icon_label)
        
        # Title and Stats Column
        title_stats_layout = QVBoxLayout()
        title_stats_layout.setSpacing(2)
        
        # Top row: Model name + NEW badge
        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        
        name_label = QLabel(model_name)
        name_font = QFont()
        name_font.setPointSize(14)
        name_font.setBold(True)
        name_label.setFont(name_font)
        name_label.setWordWrap(False)
        top_row.addWidget(name_label)
        
        if is_new:
            new_badge = QLabel("NEW")
            new_badge.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff6b6b, stop:1 #ee5a6f);
                    color: white;
                    padding: 3px 8px;
                    border-radius: 4px;
                    font-size: 13px;
                    font-weight: bold;
                }
            """)
            new_badge.setMaximumWidth(50)
            top_row.addWidget(new_badge)
        
        top_row.addStretch(1)
        title_stats_layout.addLayout(top_row)
        
        # Stats row (likes/downloads)
        stats_row = QHBoxLayout()
        stats_row.setSpacing(15)
        if downloads:
            dl_stat = QLabel(f"📥 {downloads} downloads")
            dl_stat.setStyleSheet("color: #888; font-size: 11px;")
            stats_row.addWidget(dl_stat)
        if likes:
            like_stat = QLabel(f"❤️ {likes} likes")
            like_stat.setStyleSheet("color: #888; font-size: 11px;")
            stats_row.addWidget(like_stat)
        stats_row.addStretch(1)
        title_stats_layout.addLayout(stats_row)
        
        header_layout.addLayout(title_stats_layout, 1)
        layout.addLayout(header_layout)
        
        # Description
        if description:
            desc_label = QLabel(description)
            desc_font = QFont()
            desc_font.setPointSize(13)
            desc_label.setFont(desc_font)
            desc_label.setWordWrap(True)
            # Remove setMaximumHeight to allow text to flow naturally
            layout.addWidget(desc_label)
        
        # Size and icons row
        middle_layout = QHBoxLayout()
        
        size_label = QLabel(f"📦 {size}")
        size_font = QFont()
        size_font.setPointSize(13)
        size_label.setFont(size_font)
        middle_layout.addWidget(size_label)
        
        middle_layout.addStretch(1)
        
        icons_label = QLabel(icons)
        icons_font = QFont()
        icons_font.setPointSize(18)
        icons_label.setFont(icons_font)
        middle_layout.addWidget(icons_label)
        
        layout.addLayout(middle_layout)
        
        # Model ID
        id_label = QLabel(f"📂 {model_id}")
        id_font = QFont()
        id_font.setPointSize(11)
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
    
    def _set_model_icon(self, model_id, author_icon):
        """Set a visual icon based on model family or author"""
        # Default icon if nothing else matches
        icon_text = "🤖"
        bg_color = "#3498db"
        
        mid_lower = model_id.lower()
        if "llama" in mid_lower:
            icon_text = "🦙"
            bg_color = "#4CAF50" # Meta Green
        elif "qwen" in mid_lower:
            icon_text = "💮"
            bg_color = "#9C27B0" # Alibaba Purple
        elif "mistral" in mid_lower or "mixtral" in mid_lower:
            icon_text = "🌬️"
            bg_color = "#FF9800" # Mistral Orange
        elif "gemma" in mid_lower or "google" in mid_lower:
            icon_text = "💎"
            bg_color = "#4285F4" # Google Blue
        elif "phi" in mid_lower or "microsoft" in mid_lower:
            icon_text = "Φ"
            bg_color = "#00A4EF" # MS Blue
        elif "deepseek" in mid_lower:
            icon_text = "🐳"
            bg_color = "#0000FF" # DeepSeek Blue
        
        self.icon_label.setText(icon_text)
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                color: white;
                border-radius: 25px;
                font-size: 24px;
                font-weight: bold;
                border: 2px solid rgba(255, 255, 255, 0.2);
            }}
        """)
    
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
    delete_clicked = Signal(str)  # Emits model path
    
    def __init__(self, model_name: str, model_path: str, size: str, icons: str, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.model_name = model_name
        self.is_dark = True
        
        self.setMinimumHeight(140)
        # Remove setMaximumHeight to allow automatic sizing
        self.setFrameShape(QFrame.Box)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(6)
        
        # Top row: Model name + Delete button
        top_layout = QHBoxLayout()
        
        name_label = QLabel(model_name)
        name_font = QFont()
        name_font.setPointSize(14)
        name_font.setBold(True)
        name_label.setFont(name_font)
        name_label.setWordWrap(True)
        top_layout.addWidget(name_label, 1)
        
        self.delete_btn = QPushButton("🗑️")
        self.delete_btn.setToolTip("Delete this model")
        self.delete_btn.setFixedSize(30, 30)
        self.delete_btn.setCursor(Qt.ArrowCursor)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 0, 0, 0.1);
                border: 1px solid rgba(255, 0, 0, 0.3);
                border-radius: 4px;
                font-size: 16px;
            }
            QPushButton:hover {
                background: rgba(255, 0, 0, 0.3);
                border: 1px solid rgba(255, 0, 0, 0.5);
            }
        """)
        self.delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self.model_path))
        top_layout.addWidget(self.delete_btn)
        
        layout.addLayout(top_layout)
        
        # Size and icons
        bottom_layout = QHBoxLayout()
        
        size_label = QLabel(f"📦 {size}")
        size_font = QFont()
        size_font.setPointSize(13)
        size_label.setFont(size_font)
        bottom_layout.addWidget(size_label)
        
        bottom_layout.addStretch(1)
        
        icons_label = QLabel(icons)
        icons_font = QFont()
        icons_font.setPointSize(16)
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


