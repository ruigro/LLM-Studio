"""
LLM Server Manager
Manages lifecycle of persistent LLM inference servers.
Handles starting, health checking, and monitoring server processes.

PHASE 1 REFACTOR: Uses StateStore for runtime state instead of rewriting YAML.
YAML is now static config only; ports are allocated at runtime and stored in DB.

THREAD SAFETY FIX: Added threading locks to prevent race conditions when
multiple chat threads access the server manager concurrently.
"""
import yaml
import subprocess
import requests
import time
import socket
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, IO

from core.state_store import get_state_store

logger = logging.getLogger(__name__)


class LLMServerManager:
    """Manages persistent LLM inference servers"""
    
    def __init__(self, config_path: Path):
        """
        Initialize server manager.
        
        Args:
            config_path: Path to llm_backends.yaml configuration file (static config only)
        """
        self.config_path = config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self._config_mtime: Optional[float] = None
        self._load_config()
        
        # StateStore for runtime state (PHASE 1: single source of truth)
        self.state_store = get_state_store()
        
        # Import environment registry
        from core.envs.env_registry import EnvRegistry
        self.env_registry = EnvRegistry()
        
        # THREAD SAFETY: Lock for all server operations
        # Prevents race conditions when multiple chat threads access manager
        self._server_lock = threading.RLock()
        
        # Track running server processes
        # Tuple: (process, log_file_handle, log_file_path)
        # log_file_handle and log_file_path are None after successful startup
        self.running_servers: Dict[str, Tuple[subprocess.Popen, Optional[IO], Optional[str]]] = {}
        
        # Warmup timeout (seconds to wait for server to become READY).
        # Large models can take a long time on first load (esp. after install / cold cache).
        self.warmup_timeout = 1800
        try:
            self.warmup_timeout = int(os.getenv("LLM_SERVER_WARMUP_TIMEOUT", str(self.warmup_timeout)))
        except Exception:
            self.warmup_timeout = 1800

    def _load_config(self) -> None:
        """(Re)load llm_backends.yaml into self.config if present/valid."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if "models" not in cfg or not isinstance(cfg["models"], dict):
            raise ValueError(f"Invalid config: 'models' key not found in {self.config_path}")

        self.config = cfg
        try:
            self._config_mtime = self.config_path.stat().st_mtime
        except Exception:
            self._config_mtime = None

    def _reload_config_if_changed(self) -> None:
        """
        Reload config if file changed on disk.
        This prevents stale config when users edit llm_backends.yaml while the app is running.
        """
        try:
            mtime = self.config_path.stat().st_mtime
        except Exception:
            return

        if self._config_mtime is None or mtime != self._config_mtime:
            try:
                self._load_config()
                logger.info(f"Reloaded LLM config from disk: {self.config_path}")
            except Exception as e:
                # Keep previous config if reload fails to avoid breaking running servers.
                logger.warning(f"Failed to reload LLM config: {e}")

    def _save_config(self) -> None:
        """
        PHASE 1: DEPRECATED - YAML is now static config only.
        Runtime state (ports, PIDs) is stored in StateStore.
        This method is kept for backward compatibility but does nothing.
        """
        logger.warning("_save_config() called but YAML rewriting is deprecated. Use StateStore instead.")
        # NO-OP: Do not rewrite YAML at runtime

    def _find_free_port(self, start_port: int, used_ports: Optional[set] = None, max_tries: int = 200) -> Optional[int]:
        """Find a free localhost port not in used_ports."""
        used_ports = used_ports or set()
        p = max(1, int(start_port))
        for _ in range(max_tries):
            if p not in used_ports and self._check_port_available(p):
                return p
            p += 1
        return None
    
    def ensure_server_running(self, model_id: str, log_callback=None) -> str:
        """
        Ensure server is running for given model_id, start if needed.
        THREAD SAFE: Uses lock to prevent concurrent starts.
        
        Args:
            model_id: Model identifier from config
            log_callback: Optional function to call with log messages
            
        Returns:
            Base URL of the running server
            
        Raises:
            ValueError: If model_id not found in config
            RuntimeError: If port is in use or server fails to start
            TimeoutError: If server doesn't become healthy in time
        """
        # THREAD SAFETY: Acquire lock for entire operation
        with self._server_lock:
            def log(msg):
                if log_callback:
                    log_callback(msg)
                logger.info(msg)

            # Always reload before resolving model_id. The file is small and this avoids
            # Windows timestamp resolution / cached config edge cases.
            try:
                self._load_config()
            except Exception as e:
                logger.warning(f"Failed to reload LLM config before start: {e}")

            if model_id not in self.config["models"]:
                raise ValueError(
                    f"Model '{model_id}' not found in config. "
                    f"Available: {list(self.config['models'].keys())}"
                )
            
            # Check if already running and healthy
            if model_id in self.running_servers:
                process, _, _ = self.running_servers[model_id]
                if process.poll() is None:  # Process is alive
                    if self._check_health(model_id):
                        log(f"Server '{model_id}' already running and healthy")
                        return self._get_server_url(model_id)
                    else:
                        log(f"Server '{model_id}' process alive but not healthy, restarting")
                        process.kill()
                        del self.running_servers[model_id]
                else:
                    log(f"Server '{model_id}' process died, restarting")
                    del self.running_servers[model_id]
            
            # Start new server
            self._start_server(model_id, log_callback=log_callback)
            return self._get_server_url(model_id)
    
    def _check_port_available(self, port: int) -> bool:
        """
        Check if port is available.
        
        Args:
            port: Port number to check
            
        Returns:
            True if available, False if in use
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', port))
            sock.close()
            return True
        except OSError:
            sock.close()
            return False
    
    def _start_server(self, model_id: str, log_callback=None):
        """
        Start server in correct environment with warmup polling.
        PHASE 1: Uses StateStore for port allocation instead of rewriting YAML.
        
        Args:
            model_id: Model identifier from config
            log_callback: Optional function to call with log messages
            
        Raises:
            RuntimeError: If port is in use or server process dies
            TimeoutError: If server doesn't become healthy in time
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            logger.info(msg)

        model_cfg = self.config["models"][model_id]
        base_model = model_cfg["base_model"]
        
        # PHASE 1: Check StateStore for existing server first
        server_state = self.state_store.get_server(model_id)
        if server_state and server_state['status'] == 'RUNNING':
            # Try to reuse existing server
            port = server_state['port']
            status, reported_model = self._check_health(model_id, return_status=True)
            if isinstance(status, str) and status in {"ok", "loading"}:
                log(f"Reusing existing server on port {port} (status={status})")
                return
        
        # PHASE 1: Get preferred port from YAML or allocate new one
        preferred_port = model_cfg.get("port", 10500)  # Default if not specified
        port = preferred_port
        
        log(f"Starting server for model '{model_id}' on port {port}")
        
        # Check port availability (with retry for TIME_WAIT state)
        max_retries = 3
        for attempt in range(max_retries):
            if self._check_port_available(port):
                log(f"Port {port} is available")
                break
            
            # Try to check if it's our server already running (ready or still loading)
            status, reported_model = self._check_health(model_id, return_status=True)
            if isinstance(status, str) and status in {"ok", "loading"}:
                # If /health speaks our API, prefer reusing the server. Older servers may report
                # model="local-llm" (generic) so don't treat that as a conflict.
                if reported_model and reported_model not in {model_id, "local-llm"}:
                    log(f"Port {port} has a different server (model={reported_model}, status={status}); will reassign port")
                else:
                    log(f"Port {port} already has a server (status={status}), using it")
                    # PHASE 1: Update StateStore
                    self.state_store.upsert_server(model_id, None, port, "RUNNING")
                    return
            
            if attempt < max_retries - 1:
                log(f"Port {port} appears in use, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                # PHASE 1: Auto-reassign to a free port and persist to StateStore (not YAML)
                # Get all ports currently in use from StateStore
                all_servers = self.state_store.list_servers()
                used_ports = {s['port'] for s in all_servers}
                # Also check YAML ports as a hint
                used_ports.update({cfg.get("port") for cfg in self.config.get("models", {}).values() 
                                 if isinstance(cfg, dict) and cfg.get("port")})
                
                new_port = self._find_free_port(port + 1, used_ports=used_ports)
                if new_port is None:
                    raise RuntimeError(
                        f"Port {port} is in use but not responding to health checks, and no free port was found.\n"
                        f"Stop the process using port {port} or choose a different port.\n"
                        f"Use 'netstat -ano | findstr :{port}' to find the process"
                    )
                log(f"Port {port} is in use and not our server; switching '{model_id}' to port {new_port}")
                port = int(new_port)
                log(f"Allocated runtime port: {model_id} -> {port} (stored in StateStore)")
                break
        
        # Get environment for this model
        log(f"Getting environment for model: {base_model}")
        env_spec = self.env_registry.get_env_for_model(base_model, log_callback=log_callback)
        log(f"Using environment: {env_spec.key}")
        log(f"Python executable: {env_spec.python_executable}")
        
        # Get app root for cwd
        from core.inference import get_app_root
        app_root = get_app_root()
        
        # Construct launcher script path
        launcher_script = app_root / "scripts" / "llm_server_start.py"
        
        if not launcher_script.exists():
            raise FileNotFoundError(f"Launcher script not found: {launcher_script}")
        
        # Launch server using environment's python
        log(f"Launching server: {env_spec.python_executable} {launcher_script} {model_id}")
        log(f"Working directory: {app_root}")
        
        # PHASE 1: Record server starting in StateStore
        from datetime import datetime
        self.state_store.upsert_server(
            model_id=model_id,
            pid=None,  # Will update after process starts
            port=port,
            status="STARTING",
            started_at=datetime.utcnow().isoformat()
        )
        
        # Create log directory and file for startup capture
        log_dir = app_root / "logs" / "server_startup"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log file path with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_model_id = model_id.replace("/", "__").replace("\\", "__")
        log_path = log_dir / f"{safe_model_id}_{timestamp}.log"
        
        # Open log file for writing (will be closed after startup completes or fails)
        log_file = open(log_path, 'w', encoding='utf-8', errors='replace')
        
        # Prepare Windows subprocess flags to hide CMD window
        subprocess_kwargs = {
            'cwd': str(app_root),
            # Capture output to log file during startup for debugging
            # After successful startup, we'll close the file and switch to DEVNULL
            'stdout': log_file,
            'stderr': subprocess.STDOUT,  # Merge stderr into stdout
            'text': True,
            'encoding': 'utf-8',
            'errors': 'replace'
        }
        
        # Windows-specific: Hide CMD window and prevent blocking
        if os.name == 'nt':  # Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            subprocess_kwargs['startupinfo'] = startupinfo
            subprocess_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        
        try:
            process = subprocess.Popen(
                [str(env_spec.python_executable), 
                 str(launcher_script), 
                 model_id],
                **subprocess_kwargs
            )
            # Store process with log file handle and path
            self.running_servers[model_id] = (process, log_file, str(log_path))
            log(f"Server process started with PID: {process.pid}")
            log(f"Startup logs being captured to: {log_path}")
            
            # PHASE 1: Update StateStore with PID
            self.state_store.upsert_server(
                model_id=model_id,
                pid=process.pid,
                port=port,
                status="STARTING"
            )
        except Exception as e:
            import traceback
            error_msg = f"Failed to launch server process: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            # PHASE 1: Record failure in StateStore
            self.state_store.upsert_server(
                model_id=model_id,
                pid=None,
                port=port,
                status="FAILED",
                last_error=error_msg[:500]
            )
            raise RuntimeError(error_msg)
        
        # Warmup: poll /health with timeout
        server_url = self._get_server_url(model_id)
        start_time = time.time()
        last_error = None
        
        log(f"Waiting for server to become healthy (timeout: {self.warmup_timeout}s)...")
        log(f"This involves loading the model into GPU memory. Please wait...")
        
        while time.time() - start_time < self.warmup_timeout:
            # Check if process died
            if process.poll() is not None:
                # Process died - read log file for error details
                log_output = ""
                if model_id in self.running_servers:
                    _, log_file_handle, log_file_path = self.running_servers[model_id]
                    
                    # Close and flush log file
                    try:
                        if log_file_handle:
                            log_file_handle.flush()
                            log_file_handle.close()
                    except Exception:
                        pass
                    
                    # Read log file contents (last 2000 lines or 100KB)
                    try:
                        if log_file_path and os.path.exists(log_file_path):
                            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                                lines = f.readlines()
                                # Get last 2000 lines or all if less
                                if len(lines) > 2000:
                                    lines = lines[-2000:]
                                log_output = "".join(lines)
                                
                                # Limit to 100KB to avoid huge error messages
                                if len(log_output) > 100000:
                                    log_output = "... (truncated) ...\n" + log_output[-100000:]
                    except Exception as e:
                        log_output = f"(Failed to read log file: {e})"
                    
                    # Clean up
                    del self.running_servers[model_id]
                    
                    # Delete log file after reading
                    try:
                        if log_file_path and os.path.exists(log_file_path):
                            os.remove(log_file_path)
                    except Exception:
                        pass
                
                # PHASE 1: Record failure in StateStore
                self.state_store.upsert_server(
                    model_id=model_id,
                    pid=process.pid,
                    port=port,
                    status="FAILED",
                    stopped_at=datetime.utcnow().isoformat(),
                    last_error=f"Process died during startup (exit code {process.returncode})"
                )
                
                error_msg = (
                    f"Server process for '{model_id}' died during startup.\n"
                    f"Exit code: {process.returncode}\n"
                    f"Port: {port}\n"
                    f"Python: {env_spec.python_executable}\n"
                    f"Script: {launcher_script}\n"
                    f"\nServer output:\n{log_output if log_output else '(no output captured)'}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Try health check (server may be up while model is still loading)
            try:
                response = requests.get(f"{server_url}/health", timeout=2)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        status = str(data.get("status", "")).lower()
                        if status == "ok":
                            elapsed = time.time() - start_time
                            logger.info(f"Server '{model_id}' is ready at {server_url} (took {elapsed:.1f}s)")
                            
                            # Close and clean up log file after successful startup
                            if model_id in self.running_servers:
                                _, log_file_handle, log_file_path = self.running_servers[model_id]
                                try:
                                    if log_file_handle:
                                        log_file_handle.flush()
                                        log_file_handle.close()
                                except Exception:
                                    pass
                                
                                # Delete log file (no longer needed after successful startup)
                                try:
                                    if log_file_path and os.path.exists(log_file_path):
                                        os.remove(log_file_path)
                                except Exception:
                                    pass
                                
                                # Update to remove log file references (keep only process)
                                self.running_servers[model_id] = (process, None, None)
                            
                            # PHASE 1: Record success in StateStore
                            self.state_store.upsert_server(
                                model_id=model_id,
                                pid=process.pid,
                                port=port,
                                status="RUNNING"
                            )
                            return
                        # Still loading; keep waiting
                        last_error = f"Server up, model status={status}"
                    except Exception:
                        # If JSON parsing fails, assume not ready yet.
                        last_error = "Server up, invalid /health JSON"
            except requests.exceptions.RequestException as e:
                last_error = str(e)
            except Exception as e:
                last_error = str(e)
            
            time.sleep(2)
        
        # Timeout reached - kill process and raise error
        logger.error(f"Server '{model_id}' failed to become healthy within {self.warmup_timeout}s")
        
        # Read log file for error details
        log_output = ""
        if model_id in self.running_servers:
            _, log_file_handle, log_file_path = self.running_servers[model_id]
            
            # Close and flush log file
            try:
                if log_file_handle:
                    log_file_handle.flush()
                    log_file_handle.close()
            except Exception:
                pass
            
            # Read log file contents (last 2000 lines or 100KB)
            try:
                if log_file_path and os.path.exists(log_file_path):
                    with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                        lines = f.readlines()
                        # Get last 2000 lines or all if less
                        if len(lines) > 2000:
                            lines = lines[-2000:]
                        log_output = "".join(lines)
                        
                        # Limit to 100KB to avoid huge error messages
                        if len(log_output) > 100000:
                            log_output = "... (truncated) ...\n" + log_output[-100000:]
            except Exception as e:
                log_output = f"(Failed to read log file: {e})"
        
        # Kill process
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            pass
        
        # Clean up
        if model_id in self.running_servers:
            del self.running_servers[model_id]
            
            # Delete log file after reading
            try:
                if log_file_path and os.path.exists(log_file_path):
                    os.remove(log_file_path)
            except Exception:
                pass
        
        # Build error message
        if log_output:
            output_text = log_output
        else:
            output_text = "(no output captured - process may have failed to start)"
        
        error_msg = (
            f"Server '{model_id}' failed to become healthy within {self.warmup_timeout}s.\n"
            f"Port: {port}\n"
            f"Last health check error: {last_error or 'Connection refused'}\n"
            f"\nServer output:\n{output_text}"
        )
        logger.error(error_msg)
        raise TimeoutError(error_msg)
    
    def _check_health(self, model_id: str, return_status: bool = False):
        """
        Check if server is healthy.
        
        Args:
            model_id: Model identifier
            return_status: If True, return (status, model_name) or (False, None) on failure.
            
        Returns:
            True if ready/ok, False otherwise (default behavior).
        """
        try:
            url = self._get_server_url(model_id)
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code != 200:
                return False if not return_status else (False, None)
            try:
                data = response.json()
            except Exception:
                # A 200 without JSON is not our API; treat as not healthy.
                return False if not return_status else ("unknown", None)

            status = str(data.get("status", "")).lower().strip()
            model_name = str(data.get("model", "")).strip() if isinstance(data, dict) else ""
            if return_status:
                return (status or "unknown", model_name or None)
            return status == "ok"
        except Exception:
            return False if not return_status else (False, None)
    
    def _get_server_url(self, model_id: str) -> str:
        """
        Get base URL for model server.
        PHASE 1: Checks StateStore first, falls back to YAML.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Base URL (e.g., "http://127.0.0.1:9100")
        """
        # PHASE 1: Check StateStore for runtime port first
        server_state = self.state_store.get_server(model_id)
        if server_state:
            port = server_state['port']
        else:
            # Fallback to YAML port
            port = self.config["models"][model_id].get("port", 10500)
        return f"http://127.0.0.1:{port}"
    
    def shutdown_server(self, model_id: str):
        """
        Shutdown server for given model.
        THREAD SAFE: Uses lock to prevent concurrent shutdown.
        
        Args:
            model_id: Model identifier
        """
        # THREAD SAFETY: Acquire lock for shutdown operation
        with self._server_lock:
            if model_id in self.running_servers:
                process, log_file_handle, log_file_path = self.running_servers[model_id]
                
                # Close log file if still open
                try:
                    if log_file_handle:
                        log_file_handle.flush()
                        log_file_handle.close()
                except Exception:
                    pass
                
                logger.info(f"Shutting down server '{model_id}'")
                
                # Graceful shutdown with timeout
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    try:
                        process.wait(timeout=2)  # Wait 2 seconds for graceful shutdown
                        logger.info(f"Server '{model_id}' terminated gracefully")
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful shutdown failed
                        logger.warning(f"Server '{model_id}' didn't terminate gracefully within 2s, force killing")
                        try:
                            process.kill()
                            process.wait(timeout=1)  # Wait briefly for kill to complete
                            logger.info(f"Server '{model_id}' force killed")
                        except subprocess.TimeoutExpired:
                            logger.error(f"Server '{model_id}' could not be killed")
                        except Exception as e:
                            logger.error(f"Error killing server '{model_id}': {e}")
                
                # Delete log file if it exists
                try:
                    if log_file_path and os.path.exists(log_file_path):
                        os.remove(log_file_path)
                except Exception:
                    pass
                
                del self.running_servers[model_id]
                
                # PHASE 1: Update StateStore
                from datetime import datetime
                self.state_store.upsert_server(
                    model_id=model_id,
                    pid=None,
                    port=0,  # Mark as stopped
                    status="STOPPED",
                    stopped_at=datetime.utcnow().isoformat()
                )
    
    def shutdown_all(self):
        """Shutdown all running servers"""
        if not self.running_servers:
            logger.info("No servers to shutdown")
            return
        
        logger.info(f"Shutting down all {len(self.running_servers)} running servers")
        model_ids = list(self.running_servers.keys())
        for model_id in model_ids:
            try:
                self.shutdown_server(model_id)
            except Exception as e:
                logger.error(f"Error shutting down server '{model_id}': {e}")
        logger.info("All servers shutdown complete")

# Global instance with thread-safe singleton pattern
_global_manager: Optional[LLMServerManager] = None
_manager_lock = threading.Lock()


def get_global_server_manager() -> LLMServerManager:
    """
    Get or create global server manager instance.
    THREAD SAFE: Uses double-checked locking pattern.
    
    Returns:
        Global LLMServerManager instance
    """
    global _global_manager
    
    # Fast path: manager already exists
    if _global_manager is not None:
        return _global_manager
    
    # Slow path: need to create manager with lock
    with _manager_lock:
        # Double-check inside lock (another thread may have created it)
        if _global_manager is None:
            from core.inference import get_app_root
            config_path = get_app_root() / "configs" / "llm_backends.yaml"
            _global_manager = LLMServerManager(config_path)
        
        return _global_manager
