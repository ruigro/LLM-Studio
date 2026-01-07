"""
Server page for managing the tool server.
"""
from __future__ import annotations

import json
import secrets
import socket
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal, Qt, QUrl, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QCheckBox, QTextEdit, QFileDialog, QGroupBox, QFrame, QMessageBox, QApplication
)
from PySide6.QtGui import QDesktopServices, QClipboard

from tool_server.server import Server, ToolContext
from desktop_app.config.config_manager import ConfigManager


class ServerThread(QThread):
    """Thread for running the server without blocking UI"""
    log_output = Signal(str)
    server_started = Signal(str)  # host:port
    server_stopped = Signal()
    error = Signal(str)

    def __init__(self, host: str, port: int, ctx: ToolContext):
        super().__init__()
        self.host = host
        self.port = port
        self.ctx = ctx
        self.server: Optional[Server] = None
        self._running = False

    def run(self):
        try:
            self._running = True
            self.server = Server(self.host, self.port, self.ctx)
            self.log_output.emit(f"[server] root={self.ctx.root}")
            self.log_output.emit(f"[server] listening http://{self.host}:{self.port}")
            self.server_started.emit(f"{self.host}:{self.port}")
            self.server.serve_forever()
        except Exception as e:
            if self._running:
                self.error.emit(str(e))
        finally:
            self._running = False
            self.server_stopped.emit()

    def stop(self):
        self._running = False
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass


class ServerPage(QWidget):
    """Server management page"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.server_thread: Optional[ServerThread] = None
        self.config_manager = None  # Defer initialization
        self._server_address = ""
        self._loading_config = False
        self._setup_ui()
        # Defer config loading to avoid blocking during page creation
        QTimer.singleShot(100, self._initialize_config)
    
    def closeEvent(self, event):
        """Stop server when page/widget is closed."""
        self._stop_server_on_close()
        super().closeEvent(event)
    
    def _stop_server_on_close(self):
        """Stop the server thread if running."""
        if self.server_thread is not None:
            try:
                # Check if thread is still valid and running
                if self.server_thread.isRunning():
                    # Stop the server
                    self.server_thread.stop()
                    # Wait briefly for thread to stop (non-blocking check)
                    if not self.server_thread.wait(1000):  # 1 second timeout
                        # Thread didn't stop in time, but continue anyway
                        pass
            except RuntimeError:
                # C++ object already deleted, nothing to do
                pass
            except Exception:
                # Other error, continue anyway
                pass
            finally:
                self.server_thread = None

    def request_stop(self, on_done=None, timeout_ms: int = 5000) -> bool:
        """
        Ask the server to stop and invoke `on_done` once it's actually stopped.
        Returns True if a stop was requested (server was running), else False.
        """
        thread = self.server_thread
        if thread is None:
            print("[DEBUG] request_stop: No server thread found")
            if callable(on_done):
                QTimer.singleShot(0, on_done)
            return False

        try:
            running = thread.isRunning()
        except RuntimeError:
            print("[DEBUG] request_stop: Server thread C++ object already deleted")
            self.server_thread = None
            if callable(on_done):
                QTimer.singleShot(0, on_done)
            return False

        if not running:
            print("[DEBUG] request_stop: Server thread not running")
            if callable(on_done):
                QTimer.singleShot(0, on_done)
            return False

        print("[DEBUG] request_stop: Sending stop signal to server thread...")
        done_called = {"v": False}

        def _done():
            if done_called["v"]:
                return
            done_called["v"] = True
            print("[DEBUG] request_stop: Server stop confirmed or timed out")
            if callable(on_done):
                try:
                    on_done()
                except Exception:
                    pass

        # When the thread signals stopped, proceed.
        try:
            thread.server_stopped.connect(_done)
        except Exception as e:
            print(f"[DEBUG] request_stop: Could not connect to server_stopped: {e}")

        # Timeout safety: don't hang app close forever if something is stuck.
        QTimer.singleShot(timeout_ms, _done)

        # Request stop (the same procedure as clicking the button)
        try:
            thread.stop()
            print("[DEBUG] request_stop: thread.stop() called")
        except RuntimeError:
            print("[DEBUG] request_stop: RuntimeError during thread.stop()")
            self.server_thread = None
            _done()
        except Exception as e:
            print(f"[DEBUG] request_stop: Error calling thread.stop(): {e}")
            _done()

        return True

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("🖧 Server")
        title.setProperty("class", "page_title")
        layout.addWidget(title)

        cols = QHBoxLayout()
        cols.setSpacing(12)

        # LEFT: Controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        control_group = QGroupBox("Server Control")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(10)

        self.start_stop_btn = QPushButton("Start Server")
        self.start_stop_btn.clicked.connect(self._toggle_server)
        control_layout.addWidget(self.start_stop_btn)

        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.port_edit = QLineEdit("8765")
        port_layout.addWidget(self.port_edit)
        control_layout.addLayout(port_layout)

        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("Token:"))
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText("Auth token for /tools and /call")
        token_layout.addWidget(self.token_edit, 1)
        
        # Generate token button
        generate_token_btn = QPushButton("🎲 Generate")
        generate_token_btn.setToolTip("Generate a random secure token")
        generate_token_btn.clicked.connect(self._generate_token)
        token_layout.addWidget(generate_token_btn)
        
        control_layout.addLayout(token_layout)

        root_layout = QHBoxLayout()
        root_layout.addWidget(QLabel("Root:"))
        self.root_edit = QLineEdit(str(Path.cwd()))
        root_layout.addWidget(self.root_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._select_root)
        root_layout.addWidget(browse_btn)
        control_layout.addLayout(root_layout)

        self.expose_to_lan_check = QCheckBox("Expose to LAN (0.0.0.0)")
        control_layout.addWidget(self.expose_to_lan_check)

        left_layout.addWidget(control_group)

        # Permissions
        perm_group = QGroupBox("Permissions")
        perm_layout = QVBoxLayout(perm_group)
        self.allow_shell_check = QCheckBox("Allow Shell Commands")
        self.allow_write_check = QCheckBox("Allow File Write")
        self.allow_git_check = QCheckBox("Allow Git Operations")
        self.allow_git_check.setChecked(True)
        self.allow_network_check = QCheckBox("Allow Network Access")
        perm_layout.addWidget(self.allow_shell_check)
        perm_layout.addWidget(self.allow_write_check)
        perm_layout.addWidget(self.allow_git_check)
        perm_layout.addWidget(self.allow_network_check)
        left_layout.addWidget(perm_group)

        # Config buttons
        config_layout = QHBoxLayout()
        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self._save_config)
        config_layout.addWidget(save_btn)
        left_layout.addLayout(config_layout)

        self.config_path_label = QLabel("Config: -")
        self.config_path_label.setWordWrap(True)
        left_layout.addWidget(self.config_path_label)

        left_layout.addStretch()
        cols.addWidget(left)

        # RIGHT: Status & Logs
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        status_group = QGroupBox("Status")
        self.status_layout = QVBoxLayout(status_group)  # Store as instance variable
        self.status_label = QLabel("Status: Stopped")
        self.address_label = QLabel("Address: -")
        self.status_layout.addWidget(self.status_label)
        self.status_layout.addWidget(self.address_label)
        
        # LAN address label and copy button (created dynamically when needed)
        self.lan_address_label = None
        self.copy_lan_btn = None

        self.health_btn = QPushButton("Check Health")
        self.health_btn.clicked.connect(self._check_health)
        self.health_btn.setEnabled(False)
        self.status_layout.addWidget(self.health_btn)
        right_layout.addWidget(status_group)

        log_group = QGroupBox("Server Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        log_layout.addWidget(self.log_text)

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_btn)
        right_layout.addWidget(log_group)

        cols.addWidget(right)
        layout.addLayout(cols)
    
    def _initialize_config(self):
        """Initialize config manager and load config (deferred)."""
        if self._loading_config:
            return
        self._loading_config = True
        try:
            if self.config_manager is None:
                self.config_manager = ConfigManager()
            self._load_config()
        except Exception as e:
            # Non-critical - continue without config
            pass
        finally:
            self._loading_config = False

    def _select_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select Workspace Root", self.root_edit.text())
        if path:
            self.root_edit.setText(path)
    
    def _generate_token(self):
        """Generate a random secure token."""
        # Generate a 32-byte token (64 hex characters)
        token = secrets.token_urlsafe(32)
        self.token_edit.setText(token)
        self.token_edit.setEchoMode(QLineEdit.Normal)  # Show it briefly
        QTimer.singleShot(2000, lambda: self.token_edit.setEchoMode(QLineEdit.Password))  # Hide after 2s

    def _load_config(self):
        """Load configuration from file."""
        if self.config_manager is None:
            return
        try:
            config = self.config_manager.load()
            self.port_edit.setText(str(config.get("port", 8765)))
            self.token_edit.setText(config.get("token", ""))
            self.root_edit.setText(config.get("workspace_root", str(Path.cwd())))
            self.allow_shell_check.setChecked(config.get("allow_shell", False))
            self.allow_write_check.setChecked(config.get("allow_write", False))
            self.allow_git_check.setChecked(config.get("allow_git", True))
            self.allow_network_check.setChecked(config.get("allow_network", False))
            self.expose_to_lan_check.setChecked(config.get("expose_to_lan", False))
            self.config_path_label.setText(f"Config: {self.config_manager.get_config_path()}")
        except Exception:
            # Non-critical - use defaults
            pass

    def _save_config(self):
        if self.config_manager is None:
            QMessageBox.warning(self, "Error", "Config manager not initialized. Please wait a moment and try again.")
            return
        try:
            existing = self.config_manager.load()
            expose = self.expose_to_lan_check.isChecked()
            config = {
                "host": "0.0.0.0" if expose else "127.0.0.1",
                "port": int(self.port_edit.text() or "8765"),
                "token": self.token_edit.text().strip(),
                "workspace_root": self.root_edit.text().strip() or str(Path.cwd()),
                "allow_shell": self.allow_shell_check.isChecked(),
                "allow_write": self.allow_write_check.isChecked(),
                "allow_git": self.allow_git_check.isChecked(),
                "allow_network": self.allow_network_check.isChecked(),
                "expose_to_lan": expose,
                "enabled_tools": existing.get("enabled_tools", {}),
            }
            self.config_manager.save(config)
            QMessageBox.information(self, "Saved", f"Config saved to:\n{self.config_manager.get_config_path()}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save: {e}")

    def _save_config_silent(self):
        if self.config_manager is None:
            return
        try:
            existing = self.config_manager.load()
            expose = self.expose_to_lan_check.isChecked()
            config = {
                "host": "0.0.0.0" if expose else "127.0.0.1",
                "port": int(self.port_edit.text() or "8765"),
                "token": self.token_edit.text().strip(),
                "workspace_root": self.root_edit.text().strip() or str(Path.cwd()),
                "allow_shell": self.allow_shell_check.isChecked(),
                "allow_write": self.allow_write_check.isChecked(),
                "allow_git": self.allow_git_check.isChecked(),
                "allow_network": self.allow_network_check.isChecked(),
                "expose_to_lan": expose,
                "enabled_tools": existing.get("enabled_tools", {}),
            }
            self.config_manager.save(config)
        except Exception:
            pass

    def _toggle_server(self):
        # Prevent rapid clicking - disable button immediately
        if not self.start_stop_btn.isEnabled():
            return  # Already processing
        
        # Check if thread exists and is valid (C++ object not deleted)
        thread_valid = False
        thread_running = False
        if self.server_thread is not None:
            try:
                # Try to access the thread - will raise RuntimeError if C++ object is deleted
                thread_running = self.server_thread.isRunning()
                thread_valid = True
            except RuntimeError:
                # C++ object was deleted, set to None
                self.server_thread = None
                thread_valid = False
        
        if thread_valid and thread_running:
            # Stop the server
            try:
                self.server_thread.stop()
            except RuntimeError:
                # Thread was deleted, reset state
                self.server_thread = None
                self._on_stopped()
                return
            # Don't wait() here as it blocks the UI thread
            self.start_stop_btn.setText("Stopping...")
            self.start_stop_btn.setEnabled(False)
            # Schedule cleanup check
            QTimer.singleShot(500, self._check_thread_cleanup)
        else:
            # Check if previous thread is still cleaning up
            if thread_valid and not thread_running:
                # Thread finished but not yet deleted - wait a bit
                QTimer.singleShot(200, self._start_server)
            else:
                self._start_server()
    
    def _check_thread_cleanup(self):
        """Check if thread cleanup is complete."""
        if self.server_thread is None:
            # Thread was deleted, reset state
            self._on_stopped()
            return
        
        try:
            if not self.server_thread.isRunning():
                # Thread has stopped, reset button state
                self._on_stopped()
            else:
                # Still stopping, check again
                QTimer.singleShot(500, self._check_thread_cleanup)
        except RuntimeError:
            # C++ object was deleted, reset state
            self.server_thread = None
            self._on_stopped()
    
    def _start_server(self):
        """Start the server (called after ensuring previous thread is cleaned up)."""
        try:
            # Clean up previous thread if it exists (before creating new one)
            old_thread = self.server_thread
            if old_thread is not None:
                # Disconnect all signals to prevent stale callbacks
                try:
                    # Check if C++ object is still valid
                    if old_thread.isRunning():
                        # Thread still running, can't clean up yet
                        QMessageBox.warning(self, "Error", "Previous server thread is still running. Please wait.")
                        return
                    
                    # Safely disconnect all signals
                    for signal in [old_thread.log_output, old_thread.server_started, 
                                 old_thread.server_stopped, old_thread.error, old_thread.finished]:
                        try:
                            signal.disconnect()
                        except (RuntimeError, TypeError, Exception):
                            pass
                            
                    # Delete old thread
                    old_thread.deleteLater()
                except RuntimeError:
                    # C++ object already deleted, just clear reference
                    pass
                except Exception:
                    # Other error, try to continue
                    pass
                finally:
                    self.server_thread = None
            
            port = int(self.port_edit.text() or "8765")
            token = self.token_edit.text().strip() or "CHANGE_ME"
            root = Path(self.root_edit.text().strip() or ".")
            expose = self.expose_to_lan_check.isChecked()
            host = "0.0.0.0" if expose else "127.0.0.1"

            # Keep start lightweight: skip config save/load here
            ctx = ToolContext(
                root=root,
                token=token,
                allow_shell=self.allow_shell_check.isChecked(),
                allow_write=self.allow_write_check.isChecked(),
                allow_git=self.allow_git_check.isChecked(),
                allow_network=self.allow_network_check.isChecked(),
                require_token_for_openai=False,
                backend="mock",
                model_path="",
                backend_url="",
                enabled_tools={},
                require_auth_for_tools_list=expose,
            )

            self.server_thread = ServerThread(host, port, ctx)
            self.server_thread.log_output.connect(self._append_log)
            self.server_thread.server_started.connect(self._on_started)
            self.server_thread.server_stopped.connect(self._on_stopped)
            self.server_thread.error.connect(self._on_error)
            self.server_thread.finished.connect(self.server_thread.deleteLater)

            self.start_stop_btn.setText("Starting...")
            self.start_stop_btn.setEnabled(False)
            self.server_thread.start()
        except Exception as e:
            self._append_log(f"[ERROR] {e}")
            QMessageBox.critical(self, "Error", str(e))
            self.start_stop_btn.setText("Start Server")
            self.start_stop_btn.setEnabled(True)

    def _on_started(self, address: str):
        # Normalize for internal connections
        connection_address = self._normalize_connection_address(address)
        self._server_address = connection_address
        
        # Detect LAN IP if binding to 0.0.0.0
        lan_url = None
        if address.startswith("0.0.0.0:"):
            port = address.split(":")[1]
            lan_ip = self._get_lan_ip()
            if lan_ip:
                lan_url = f"http://{lan_ip}:{port}"
        
        # Update UI
        self.start_stop_btn.setText("Stop Server")
        self.start_stop_btn.setEnabled(True)
        self.status_label.setText("Status: Running")
        
        # Display localhost address (never show 0.0.0.0)
        self.address_label.setText(f"Local: http://{connection_address}")
        
        # Add LAN address display and copy button if available
        if lan_url:
            # Create LAN label if it doesn't exist
            if self.lan_address_label is None:
                self.lan_address_label = QLabel()
                self.lan_address_label.setStyleSheet("color: #4CAF50; background: transparent;")
                # Insert before health button (which is at the end)
                self.status_layout.insertWidget(self.status_layout.count() - 1, self.lan_address_label)
                
                self.copy_lan_btn = QPushButton("📋 Copy LAN URL")
                self.copy_lan_btn.setStyleSheet("""
                    QPushButton {
                        background: rgba(76, 175, 80, 0.6);
                        color: white;
                        border: none;
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 9pt;
                    }
                    QPushButton:hover {
                        background: rgba(76, 175, 80, 0.8);
                    }
                """)
                self.status_layout.insertWidget(self.status_layout.count() - 1, self.copy_lan_btn)
            
            self.lan_address_label.setText(f"LAN: {lan_url}")
            self.lan_address_label.setVisible(True)
            self.copy_lan_btn.setVisible(True)
            # Update lambda to use current lan_url
            
            # Disconnect all existing connections (if any) before connecting new one
            try:
                # In PySide6, disconnect() raises a RuntimeError if no slots are connected.
                # It may also emit a RuntimeWarning which we can't easily suppress without the 'warnings' module.
                # However, calling it inside a try-except is standard.
                self.copy_lan_btn.clicked.disconnect()
            except (TypeError, RuntimeError, Exception):
                # No connections to disconnect or already disconnected - ignore safely
                pass
                
            self.copy_lan_btn.clicked.connect(lambda checked=False, url=lan_url: self._copy_lan_url(url))
        else:
            # Hide LAN UI if not available
            if self.lan_address_label is not None:
                self.lan_address_label.setVisible(False)
            if self.copy_lan_btn is not None:
                self.copy_lan_btn.setVisible(False)
        
        self.health_btn.setEnabled(True)

    def _on_stopped(self):
        self.start_stop_btn.setText("Start Server")
        self.start_stop_btn.setEnabled(True)
        self.status_label.setText("Status: Stopped")
        self.address_label.setText("Address: -")
        if self.lan_address_label is not None:
            self.lan_address_label.setVisible(False)
        if self.copy_lan_btn is not None:
            self.copy_lan_btn.setVisible(False)
        self.health_btn.setEnabled(False)

    def _on_error(self, error: str):
        self._append_log(f"[ERROR] {error}")
        self.start_stop_btn.setText("Start Server")
        self.start_stop_btn.setEnabled(True)
        
        # Only show the message box if we aren't in the middle of shutting down the whole app
        main_window = self.window()
        is_shutting_down = False
        if main_window:
            try:
                is_shutting_down = getattr(main_window, "_shutdown_in_progress", False)
            except Exception:
                pass
                
        if not is_shutting_down:
            QMessageBox.warning(self, "Server Error", error)

    def _append_log(self, text: str):
        self.log_text.append(text)
    
    def _get_lan_ip(self) -> Optional[str]:
        """Get local network IPv4 address (not loopback)."""
        try:
            # Connect to a remote address to determine local IP
            # Doesn't actually send data, just determines route
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # Google DNS
            ip = s.getsockname()[0]
            s.close()
            # Only return non-loopback addresses
            if ip and ip != "127.0.0.1":
                return ip
            return None
        except Exception:
            return None
    
    def _normalize_connection_address(self, address: str) -> str:
        """Convert server bind address to client connection address."""
        if address.startswith("0.0.0.0:"):
            return address.replace("0.0.0.0:", "127.0.0.1:", 1)
        return address
    
    def _copy_lan_url(self, url: str):
        """Copy LAN URL to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(url)
        QMessageBox.information(self, "Copied", f"LAN URL copied to clipboard:\n{url}")
    
    def get_server_connection_info(self) -> Optional[Dict[str, str]]:
        """Get server connection info for MCP connections."""
        if not self._server_address:
            return None  # Server not running
        
        # Get token from UI or config
        token = self.token_edit.text().strip() or "CHANGE_ME"
        if self.config_manager:
            try:
                config = self.config_manager.load()
                token = config.get("token", token)
            except Exception:
                pass
        
        return {
            "url": f"http://{self._server_address}",
            "token": token,
            "name": "Local Tool Server"
        }

    def _check_health(self):
        if not self._server_address:
            return
        
        self.health_btn.setEnabled(False)
        self.health_btn.setText("Checking...")
        
        import threading
        def worker():
            try:
                import urllib.request
                url = f"http://{self._server_address}/health"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = resp.read().decode()
                    # Update UI in main thread
                    QTimer.singleShot(0, lambda: self._on_health_check_result(f"[health] {data}"))
            except Exception as e:
                # Update UI in main thread
                QTimer.singleShot(0, lambda: self._on_health_check_result(f"[health] Error: {e}"))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _on_health_check_result(self, message: str):
        self.health_btn.setEnabled(True)
        self.health_btn.setText("Check Health")
        self._append_log(message)
