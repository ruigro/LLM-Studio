"""
LLM Server Manager
Manages lifecycle of persistent LLM inference servers.
Handles starting, health checking, and monitoring server processes.
"""
import yaml
import subprocess
import requests
import time
import socket
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class LLMServerManager:
    """Manages persistent LLM inference servers"""
    
    def __init__(self, config_path: Path):
        """
        Initialize server manager.
        
        Args:
            config_path: Path to llm_backends.yaml configuration file
        """
        self.config_path = config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self._config_mtime: Optional[float] = None
        self._load_config()
        
        # Import environment registry
        from core.envs.env_registry import EnvRegistry
        self.env_registry = EnvRegistry()
        
        # Track running server processes
        self.running_servers: Dict[str, subprocess.Popen] = {}
        
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
        """Persist current config to disk (best-effort)."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, sort_keys=False, default_flow_style=False)
            try:
                self._config_mtime = self.config_path.stat().st_mtime
            except Exception:
                self._config_mtime = None
        except Exception as e:
            logger.warning(f"Failed to save LLM config: {e}")

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
            process = self.running_servers[model_id]
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
        port = model_cfg["port"]
        
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
                    return
            
            if attempt < max_retries - 1:
                log(f"Port {port} appears in use, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                # Last attempt failed: auto-reassign to a free port and persist it.
                used_ports = {cfg.get("port") for cfg in self.config.get("models", {}).values() if isinstance(cfg, dict)}
                new_port = self._find_free_port(port + 1, used_ports=used_ports)
                if new_port is None:
                    raise RuntimeError(
                        f"Port {port} is in use but not responding to health checks, and no free port was found.\n"
                        f"Stop the process using port {port} or choose a different port.\n"
                        f"Use 'netstat -ano | findstr :{port}' to find the process"
                    )
                log(f"Port {port} is in use and not our server; switching '{model_id}' to port {new_port}")
                self.config["models"][model_id]["port"] = int(new_port)
                self._save_config()
                port = int(new_port)
                log(f"Updated config: {model_id}.port = {port}")
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
        
        try:
            process = subprocess.Popen(
                [str(env_spec.python_executable), 
                 str(launcher_script), 
                 model_id],
                cwd=str(app_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1  # Line buffered
            )
            self.running_servers[model_id] = process
            log(f"Server process started with PID: {process.pid}")
        except Exception as e:
            import traceback
            error_msg = f"Failed to launch server process: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
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
                # Process died - read all output
                try:
                    stdout, _ = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, _ = process.communicate()
                
                # Clean up
                if model_id in self.running_servers:
                    del self.running_servers[model_id]
                
                error_msg = (
                    f"Server process for '{model_id}' died during startup.\n"
                    f"Exit code: {process.returncode}\n"
                    f"Port: {port}\n"
                    f"Python: {env_spec.python_executable}\n"
                    f"Script: {launcher_script}\n"
                    f"\nServer output:\n{stdout if stdout else '(no output)'}"
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
        
        # Kill and get final output
        stdout = ""
        try:
            process.kill()
            stdout, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                stdout, _ = process.communicate(timeout=2)
            except:
                pass
        except Exception:
            pass
        
        # Clean up
        if model_id in self.running_servers:
            del self.running_servers[model_id]
        
        # Build error message
        if stdout:
            output_lines = stdout.splitlines() if isinstance(stdout, str) else stdout
            output_text = "\n".join(output_lines[-20:])
        else:
            output_text = "(no output captured - process may have failed to start)"
        
        error_msg = (
            f"Server '{model_id}' failed to become healthy within {self.warmup_timeout}s.\n"
            f"Port: {port}\n"
            f"Last health check error: {last_error or 'Connection refused'}\n"
            f"\nLast 20 lines of server output:\n{output_text}"
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
        
        Args:
            model_id: Model identifier
            
        Returns:
            Base URL (e.g., "http://127.0.0.1:9100")
        """
        port = self.config["models"][model_id]["port"]
        return f"http://127.0.0.1:{port}"
    
    def shutdown_server(self, model_id: str):
        """
        Shutdown server for given model.
        
        Args:
            model_id: Model identifier
        """
        if model_id in self.running_servers:
            process = self.running_servers[model_id]
            logger.info(f"Shutting down server '{model_id}'")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"Server '{model_id}' didn't terminate gracefully, killing")
                process.kill()
            del self.running_servers[model_id]
    
    def shutdown_all(self):
        """Shutdown all running servers"""
        model_ids = list(self.running_servers.keys())
        for model_id in model_ids:
            self.shutdown_server(model_id)


# Global instance
_global_manager: Optional[LLMServerManager] = None


def get_global_server_manager() -> LLMServerManager:
    """
    Get or create global server manager instance.
    
    Returns:
        Global LLMServerManager instance
    """
    global _global_manager
    if _global_manager is None:
        from core.inference import get_app_root
        config_path = get_app_root() / "configs" / "llm_backends.yaml"
        _global_manager = LLMServerManager(config_path)
    return _global_manager
