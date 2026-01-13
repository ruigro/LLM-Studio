"""
Environment Registry - Bridges EnvironmentManager and LLM Server System
Provides environment specifications with validated Python paths for each model.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import json
from typing import Optional


@dataclass
class EnvSpec:
    """Environment specification with validated python executable"""
    key: str  # Environment identifier
    python_executable: Path  # Path to python.exe in the environment (validated)
    metadata: dict  # Environment metadata from EnvironmentManager


class EnvRegistry:
    """Registry for managing per-model Python environments"""
    
    def __init__(self):
        from core.environment_manager import EnvironmentManager
        self.env_manager = EnvironmentManager()
    
    def get_env_for_model(self, model_path: str) -> EnvSpec:
        """
        Get environment spec for a model, creating if needed.
        
        Args:
            model_path: Path to the model (base model path)
            
        Returns:
            EnvSpec with validated python executable path
            
        Raises:
            RuntimeError: If environment is corrupted or python not found
        """
        # Get environment path - this returns the directory path
        env_dir = self.env_manager.get_environment_path(model_path=model_path)
        
        # Check if environment exists
        env_created = False
        if not self.env_manager.environment_exists(model_path=model_path):
            # Create the environment
            import logging
            logging.info(f"Environment for {model_path} doesn't exist, creating...")
            try:
                # Create environment (this installs packages too)
                success, error = self.env_manager.create_environment(
                    model_path=model_path
                )
                if not success:
                    raise RuntimeError(f"Failed to create environment: {error}")
                env_created = True
            except Exception as e:
                raise RuntimeError(f"Failed to create environment: {e}")
        
        # Get Python executable path
        python_exe = self.env_manager.get_python_executable(model_path=model_path)
        
        # Install dependencies if environment was just created or if they're missing
        if env_created or not self._has_server_dependencies(python_exe):
            import logging
            import subprocess
            from pathlib import Path
            
            # Find requirements.txt
            llm_dir = Path(__file__).parent.parent.parent
            requirements_file = llm_dir / "requirements.txt"
            
            logging.info(f"Installing dependencies for server environment...")
            try:
                # First install server framework (fast, needed first)
                logging.info(f"Installing server framework (uvicorn, fastapi)...")
                result = subprocess.run(
                    [str(python_exe), "-m", "pip", "install", 
                     "uvicorn[standard]", "fastapi", "pydantic", "pyyaml", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode != 0:
                    logging.warning(f"Failed to install server framework: {result.stderr}")
                
                # Then install ML dependencies from requirements.txt if it exists
                if requirements_file.exists():
                    logging.info(f"Installing ML dependencies from {requirements_file}...")
                    result2 = subprocess.run(
                        [str(python_exe), "-m", "pip", "install", 
                         "-r", str(requirements_file), "-q"],
                        capture_output=True,
                        text=True,
                        timeout=1800  # 30 min timeout for full ML stack
                    )
                    if result2.returncode != 0:
                        logging.warning(f"Failed to install some ML dependencies: {result2.stderr}")
                else:
                    # Fallback: install minimal ML dependencies
                    logging.info(f"requirements.txt not found, installing minimal ML dependencies...")
                    result2 = subprocess.run(
                        [str(python_exe), "-m", "pip", "install", 
                         "transformers", "peft", "torch", "accelerate", "-q"],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    if result2.returncode != 0:
                        logging.warning(f"Failed to install ML dependencies: {result2.stderr}")
            except Exception as e:
                logging.warning(f"Failed to install dependencies: {e}")
                # Non-fatal - might already be installed
        
        # VALIDATE: Ensure python executable exists
        if python_exe is None or not python_exe.exists():
            raise RuntimeError(
                f"Environment python not found: {python_exe}\n"
                f"Environment may be corrupted. Try recreating it.\n"
                f"Environment directory: {env_dir}"
            )
        
        # Validate it's actually executable (on Windows, check it's a file)
        if not python_exe.is_file():
            raise RuntimeError(
                f"Python path exists but is not a file: {python_exe}\n"
                f"Environment may be corrupted."
            )
        
        # Load metadata
        metadata_file = env_dir / "environment_metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
            except Exception as e:
                # Non-fatal: use empty metadata if we can't load it
                metadata = {"error": f"Failed to load metadata: {e}"}
        else:
            metadata = {"warning": "No metadata file found"}
        
        return EnvSpec(
            key=env_dir.name,
            python_executable=python_exe,
            metadata=metadata
        )
    
    def _has_server_dependencies(self, python_exe: Path) -> bool:
        """
        Check if server dependencies are installed.
        
        Args:
            python_exe: Path to Python executable
            
        Returns:
            True if uvicorn and fastapi are available
        """
        try:
            import subprocess
            result = subprocess.run(
                [str(python_exe), "-c", "import uvicorn, fastapi"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def validate_env_spec(self, env_spec: EnvSpec) -> bool:
        """
        Validate an environment specification.
        
        Args:
            env_spec: The environment spec to validate
            
        Returns:
            True if valid, False otherwise
        """
        return (
            env_spec.python_executable.exists() and
            env_spec.python_executable.is_file()
        )
