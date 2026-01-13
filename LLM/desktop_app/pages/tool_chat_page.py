"""
Dedicated Tool Chat page for single-model conversations with tool use.

Provides a cleaner interface focused on tool-enabled chat with one model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTextEdit, QCheckBox, QSplitter, QFrame,
    QScrollArea, QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt, QProcess, QTimer
from PySide6.QtGui import QFont

from desktop_app.config.config_manager import ConfigManager
from core.models import list_local_downloads
from core.model_capabilities import get_tool_system_prompt


class ToolChatPage(QWidget):
    """Dedicated chat interface with tool calling support"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_manager = ConfigManager()
        self.current_process = None
        self.inference_buffer = ""
        self.tool_log = []
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Title
        title = QLabel("🔧 Tool Chat")
        title.setProperty("class", "page_title")
        layout.addWidget(title)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Model selector
        controls_layout.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(300)
        controls_layout.addWidget(self.model_selector, 1)
        
        # Enable tools checkbox
        self.enable_tools_check = QCheckBox("Enable Tools")
        self.enable_tools_check.setChecked(True)
        self.enable_tools_check.setStyleSheet("color: white; font-weight: bold;")
        controls_layout.addWidget(self.enable_tools_check)
        
        # Clear chat button
        clear_btn = QPushButton("Clear Chat")
        clear_btn.clicked.connect(self._clear_chat)
        controls_layout.addWidget(clear_btn)
        
        layout.addLayout(controls_layout)
        
        # Main splitter (chat + tool log)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Chat
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(8)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background: rgba(20, 20, 30, 0.8);
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 8px;
                color: white;
                padding: 12px;
                font-size: 11pt;
            }
        """)
        chat_layout.addWidget(self.chat_display, 1)
        
        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background: rgba(30, 30, 40, 0.9);
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setSpacing(8)
        
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.setMaximumHeight(100)
        self.input_field.setStyleSheet("""
            QTextEdit {
                background: rgba(40, 40, 50, 0.9);
                border: 1px solid rgba(102, 126, 234, 0.2);
                border-radius: 4px;
                color: white;
                padding: 8px;
                font-size: 11pt;
            }
        """)
        input_layout.addWidget(self.input_field)
        
        send_layout = QHBoxLayout()
        send_layout.addStretch()
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setMinimumWidth(120)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(102, 126, 234, 0.8), stop:1 rgba(118, 75, 162, 0.8));
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(102, 126, 234, 1.0), stop:1 rgba(118, 75, 162, 1.0));
            }
            QPushButton:disabled {
                background: rgba(100, 100, 100, 0.5);
            }
        """)
        self.send_btn.clicked.connect(self._send_message)
        send_layout.addWidget(self.send_btn)
        
        input_layout.addLayout(send_layout)
        chat_layout.addWidget(input_frame)
        
        splitter.addWidget(chat_widget)
        
        # Right side: Tool execution log
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(8)
        
        log_title = QLabel("Tool Execution Log")
        log_title.setStyleSheet("color: white; font-weight: bold; font-size: 12pt;")
        log_layout.addWidget(log_title)
        
        self.tool_log_display = QTextEdit()
        self.tool_log_display.setReadOnly(True)
        self.tool_log_display.setStyleSheet("""
            QTextEdit {
                background: rgba(20, 20, 30, 0.8);
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 8px;
                color: #cccccc;
                padding: 12px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
            }
        """)
        log_layout.addWidget(self.tool_log_display, 1)
        
        splitter.addWidget(log_widget)
        
        # Set initial splitter sizes (70% chat, 30% log)
        splitter.setSizes([700, 300])
        
        layout.addWidget(splitter, 1)
        
        # Load models
        QTimer.singleShot(100, self._load_models)
    
    def _load_models(self):
        """Load available models into selector"""
        try:
            from pathlib import Path
            from core.models import get_app_root
            
            models = list_local_downloads()
            self.model_selector.clear()
            
            if not models:
                self.model_selector.addItem("(No models downloaded)", None)
            else:
                download_root = get_app_root() / "models"
                for model_name in models:
                    model_path = download_root / model_name
                    self.model_selector.addItem(model_name, str(model_path))
        except Exception as e:
            self.model_selector.addItem(f"(Error loading models: {e})", None)
    
    def _clear_chat(self):
        """Clear chat history"""
        self.chat_display.clear()
        self.tool_log_display.clear()
        self.tool_log = []
        self._append_to_chat("=== Chat cleared ===\n", color="#888")
    
    def _append_to_chat(self, text: str, color: str = "white", bold: bool = False):
        """Append text to chat display"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        
        format_str = f'<span style="color: {color};'
        if bold:
            format_str += ' font-weight: bold;'
        format_str += f'">{text}</span>'
        
        cursor.insertHtml(format_str)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()
    
    def _append_to_tool_log(self, text: str):
        """Append text to tool log"""
        self.tool_log_display.append(text)
    
    def _send_message(self):
        """Send message and run inference"""
        message = self.input_field.toPlainText().strip()
        if not message:
            return
        
        # Check if model is selected
        model_path = self.model_selector.currentData()
        if not model_path or model_path == "(No models downloaded)":
            QMessageBox.warning(self, "No Model", "Please select a model first.")
            return
        
        # Disable input
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        # Display user message
        self._append_to_chat(f"\n[User]\n", color="#667eea", bold=True)
        self._append_to_chat(f"{message}\n")
        
        # Display AI response header
        self._append_to_chat(f"\n[Assistant]\n", color="#764ba2", bold=True)
        
        # Run inference (with or without tools)
        if self.enable_tools_check.isChecked():
            self._run_with_tools(model_path, message)
        else:
            self._run_without_tools(model_path, message)
    
    def _run_without_tools(self, model_path: str, prompt: str):
        """Run normal inference without tools"""
        from core.inference import InferenceConfig, run_inference
        
        try:
            cfg = InferenceConfig(
                prompt=prompt,
                base_model=model_path,
                max_new_tokens=512,
                temperature=0.7
            )
            
            output = run_inference(cfg)
            
            # Display output
            self._append_to_chat(output + "\n")
            
        except Exception as e:
            self._append_to_chat(f"\n[ERROR] {str(e)}\n", color="#F44336")
        finally:
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
    
    def _run_with_tools(self, model_path: str, prompt: str):
        """Run inference with tool calling"""
        from core.inference import ToolEnabledInferenceConfig, run_inference_with_tools
        from core.model_capabilities import get_tool_system_prompt
        from desktop_app.widgets.tool_approval_dialog import ToolApprovalDialog
        from core.tool_calling import ToolApprovalManager
        
        try:
            # Get tool configuration
            config = self.config_manager.load()
            tool_config = config.get("tool_calling", {})
            
            # Get system prompt for model
            system_prompt = get_tool_system_prompt(model_path)
            
            # Create config
            cfg = ToolEnabledInferenceConfig(
                prompt=prompt,
                base_model=model_path,
                max_new_tokens=512,
                temperature=0.7,
                enable_tools=True,
                tool_server_url=f"http://{config.get('host', '127.0.0.1')}:{config.get('port', 8763)}",
                tool_server_token=config.get('token', ''),
                auto_execute_safe_tools=tool_config.get('auto_execute_safe', True),
                max_tool_iterations=tool_config.get('max_iterations', 5),
                system_prompt=system_prompt
            )
            
            # Tool callback
            def tool_callback(tool_name, args, result):
                self._append_to_tool_log(f"✓ {tool_name}({json.dumps(args)})")
                self._append_to_tool_log(f"  Result: {json.dumps(result)[:200]}")
                self._append_to_tool_log("")
            
            # Approval callback
            approval_manager = ToolApprovalManager(cfg.auto_execute_safe_tools)
            
            def approval_callback(tool_name, args):
                danger_level = approval_manager.get_danger_level(tool_name)
                approved, remember = ToolApprovalDialog.request_approval(
                    tool_name, args, danger_level, self
                )
                if approved and remember:
                    approval_manager.approve(tool_name, remember=True)
                elif not approved and remember:
                    approval_manager.deny(tool_name, remember=True)
                return approved
            
            # Run inference
            self._append_to_tool_log(f"=== New Inference (Tools Enabled) ===")
            output, tool_log = run_inference_with_tools(
                cfg,
                tool_callback=tool_callback,
                approval_callback=approval_callback
            )
            
            # Display output
            self._append_to_chat(output + "\n")
            
            # Log tool executions
            if tool_log:
                self._append_to_tool_log(f"\nTotal tool calls: {len(tool_log)}")
            
        except Exception as e:
            import traceback
            self._append_to_chat(f"\n[ERROR] {str(e)}\n", color="#F44336")
            self._append_to_tool_log(f"ERROR: {str(e)}")
            self._append_to_tool_log(traceback.format_exc())
        finally:
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
