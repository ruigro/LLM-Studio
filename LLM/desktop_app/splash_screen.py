"""
Professional splash screen for LLM Fine-tuning Studio
Shows system detection progress with app's signature gradient style
"""
from PySide6.QtWidgets import QSplashScreen, QVBoxLayout, QLabel, QWidget, QProgressBar, QTextEdit, QScrollArea
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QPainter, QColor, QLinearGradient, QFont, QTextCursor


class SplashScreen(QSplashScreen):
    def __init__(self):
        # Create a custom pixmap with gradient background
        pixmap = QPixmap(550, 350)
        pixmap.fill(Qt.transparent)
        
        super().__init__(pixmap, Qt.WindowStaysOnTopHint | Qt.WindowStaysOnTopHint)
        
        # Enable mouse events for scrolling
        self.setMouseTracking(True)
        
        # Create overlay widget for content
        self.content_widget = QWidget(self)
        self.content_widget.setGeometry(0, 0, 550, 350)
        
        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(8)
        
        # Title (compact)
        self.title = QLabel("🎯 OWLLM")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18pt;
                font-weight: bold;
                background: transparent;
                padding: 5px;
            }
        """)
        layout.addWidget(self.title)
        
        # Progress bar (compact)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p% - Detecting system...")
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                background: rgba(0, 0, 0, 0.3);
                color: white;
                font-size: 9pt;
                text-align: center;
                min-height: 22px;
                max-height: 22px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress)
        
        # Scrollable details (TextEdit instead of Label)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.details.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details.setStyleSheet("""
            QTextEdit {
                color: rgba(255, 255, 255, 0.95);
                font-size: 9pt;
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.25);
                border-radius: 6px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QScrollBar:vertical {
                background: rgba(0, 0, 0, 0.2);
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.5);
            }
        """)
        layout.addWidget(self.details, 1)  # Give it stretch factor
        
        # Version/footer (compact)
        self.footer = QLabel("v2.0 - Hardware-Adaptive")
        self.footer.setAlignment(Qt.AlignCenter)
        self.footer.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.5);
                font-size: 8pt;
                background: transparent;
                padding: 2px;
            }
        """)
        layout.addWidget(self.footer)
        
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)
    
    def update_progress(self, value: int, status: str, details: str = ""):
        """Update splash screen with detection progress"""
        self.progress.setValue(value)
        self.progress.setFormat(f"{value}% - {status}")
        
        if details:
            # Append to existing details
            self.details.append(details)
            # Auto-scroll to bottom
            cursor = self.details.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.details.setTextCursor(cursor)
        
        # Force repaint and process events for mouse interactivity
        self.repaint()
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
    def set_checking(self, component: str):
        """Mark a component as being checked"""
        self.details.append(f"⏳ Checking {component}...")
        cursor = self.details.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.details.setTextCursor(cursor)
        self.repaint()
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
    
    def set_result(self, component: str, result: str, is_ok: bool = True):
        """Update the last line with result"""
        # Move to last line and replace it
        cursor = self.details.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.LineUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()  # Remove the newline
        
        # Insert updated line
        icon = '✅' if is_ok else '⚠️'
        self.details.append(f"{icon} {component}: {result}")
        
        # Auto-scroll to bottom
        cursor = self.details.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.details.setTextCursor(cursor)
        
        self.repaint()
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

