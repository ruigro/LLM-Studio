"""Training metrics and visualization widgets"""
from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class MetricCard(QFrame):
    """Card widget for displaying training metrics"""
    
    def __init__(self, title: str, icon: str, value: str = "--", parent=None):
        super().__init__(parent)
        self.is_dark = True
        self.value_label = None
        
        self.setMinimumHeight(100)
        self.setMaximumHeight(110)
        self.setFrameShape(QFrame.Box)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(5)
        
        # Title with icon
        title_layout = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_font = QFont()
        icon_font.setPointSize(12)
        icon_label.setFont(icon_font)
        title_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(9)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        title_layout.addStretch(1)
        layout.addLayout(title_layout)
        
        layout.addStretch(1)
        
        # Value
        self.value_label = QLabel(value)
        value_font = QFont()
        value_font.setPointSize(24)
        value_font.setBold(True)
        self.value_label.setFont(value_font)
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)
        
        layout.addStretch(1)
        
        self._apply_style()
    
    def set_value(self, value: str):
        """Update the metric value"""
        self.value_label.setText(value)
    
    def _apply_style(self):
        if self.is_dark:
            self.setStyleSheet("""
                MetricCard {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                                stop:0 #1e2936, stop:1 #16213e);
                    border: 1px solid #3a4a5a;
                    border-radius: 8px;
                }
                QLabel {
                    background: transparent;
                    color: #fafafa;
                    border: none;
                }
            """)
        else:
            self.setStyleSheet("""
                MetricCard {
                    background: white;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
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

