#!/usr/bin/env python3
"""
Model Integrity Checker
Detects missing or incomplete models and provides download links/options
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelStatus:
    """Status of a model installation"""
    model_path: Path
    model_name: str
    has_config: bool
    has_weights: bool
    is_complete: bool
    missing_files: List[str]
    model_id: Optional[str] = None  # HuggingFace model ID
    estimated_size_mb: Optional[float] = None


class ModelIntegrityChecker:
    """Checks model installations for completeness"""
    
    # Essential files that indicate a valid model
    ESSENTIAL_FILES = [
        'config.json',
        'tokenizer_config.json',
    ]
    
    # Weight file patterns (at least one must exist)
    WEIGHT_PATTERNS = [
        'model.safetensors',
        'pytorch_model.bin',
        'model.safetensors.index.json',  # For sharded models
    ]
    
    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize the checker
        
        Args:
            models_dir: Directory containing models. Defaults to LLM/models
        """
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        self.models_dir = Path(models_dir)
        
        # Also check hf_models directory
        self.hf_models_dir = Path(__file__).parent / "hf_models"
    
    def check_all_models(self) -> List[ModelStatus]:
        """Check all models in the models directories
        
        Returns:
            List of ModelStatus objects for each model
        """
        statuses = []
        
        # Check both directories
        for base_dir in [self.models_dir, self.hf_models_dir]:
            if not base_dir.exists():
                continue
            
            for model_dir in base_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                # Skip hidden directories and cache directories
                if model_dir.name.startswith('.') or model_dir.name == '__pycache__':
                    continue
                
                status = self.check_model(model_dir)
                statuses.append(status)
        
        return statuses
    
    def check_model(self, model_path: Path) -> ModelStatus:
        """Check if a model is complete
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            ModelStatus object with detailed status
        """
        model_path = Path(model_path)
        model_name = model_path.name
        
        # List all files in the model directory
        try:
            files = [f.name for f in model_path.iterdir() if f.is_file()]
        except Exception as e:
            logger.warning(f"Could not list files in {model_path}: {e}")
            files = []
        
        # Check for config
        has_config = 'config.json' in files
        
        # Check for weights
        has_weights = any(
            any(f == pattern or f.startswith(pattern.replace('.safetensors', ''))
                for pattern in self.WEIGHT_PATTERNS)
            for f in files
        )
        
        # Check for sharded models (multiple safetensors files)
        if not has_weights:
            has_weights = any('.safetensors' in f or '.bin' in f for f in files)
        
        # Determine missing files
        missing_files = []
        for essential in self.ESSENTIAL_FILES:
            if essential not in files:
                missing_files.append(essential)
        
        if not has_weights:
            missing_files.append("model weights (*.safetensors or *.bin)")
        
        # Model is complete if it has config and weights
        is_complete = has_config and has_weights
        
        # Try to extract model ID from README or config
        model_id = self._extract_model_id(model_path, model_name)
        
        # Estimate size
        estimated_size = self._estimate_model_size(model_path)
        
        return ModelStatus(
            model_path=model_path,
            model_name=model_name,
            has_config=has_config,
            has_weights=has_weights,
            is_complete=is_complete,
            missing_files=missing_files,
            model_id=model_id,
            estimated_size_mb=estimated_size
        )
    
    def _extract_model_id(self, model_path: Path, model_name: str) -> Optional[str]:
        """Try to extract the HuggingFace model ID
        
        Args:
            model_path: Path to model directory
            model_name: Directory name
            
        Returns:
            Model ID if found, None otherwise
        """
        # Try to read from config.json
        config_file = model_path / 'config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Some configs have _name_or_path
                    if '_name_or_path' in config:
                        return config['_name_or_path']
            except Exception:
                pass
        
        # Try to read from README
        readme_file = model_path / 'README.md'
        if readme_file.exists():
            try:
                with open(readme_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for model card header
                    if '---' in content:
                        header = content.split('---')[1] if content.count('---') >= 2 else ''
                        if 'model-index' in header or 'model_name' in header:
                            # Parse basic YAML-like structure
                            for line in header.split('\n'):
                                if 'name:' in line:
                                    return line.split('name:')[1].strip().strip('"\'')
            except Exception:
                pass
        
        # Fallback: Convert directory name back to model ID
        # unsloth__llama-3.2-1b-instruct -> unsloth/llama-3.2-1b-instruct
        if '__' in model_name:
            return model_name.replace('__', '/', 1)
        
        # nvidia_Llama-3.1-Nemotron -> nvidia/Llama-3.1-Nemotron
        if '_' in model_name and not model_name.startswith('.'):
            return model_name.replace('_', '/', 1)
        
        return None
    
    def _estimate_model_size(self, model_path: Path) -> Optional[float]:
        """Estimate the size of model files in MB
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Size in MB, or None if cannot determine
        """
        try:
            total_size = 0
            for file in model_path.rglob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
            
            # Convert to MB
            return total_size / (1024 * 1024)
        except Exception:
            return None
    
    def get_incomplete_models(self) -> List[ModelStatus]:
        """Get list of incomplete models
        
        Returns:
            List of ModelStatus objects for incomplete models
        """
        all_models = self.check_all_models()
        return [m for m in all_models if not m.is_complete]
    
    def generate_download_instructions(self, status: ModelStatus) -> str:
        """Generate instructions for downloading a missing model
        
        Args:
            status: ModelStatus object
            
        Returns:
            Human-readable instructions
        """
        instructions = []
        instructions.append(f"Model: {status.model_name}")
        instructions.append(f"Location: {status.model_path}")
        instructions.append(f"Status: {'✓ Complete' if status.is_complete else '✗ Incomplete'}")
        
        if not status.is_complete:
            instructions.append(f"\nMissing files:")
            for missing in status.missing_files:
                instructions.append(f"  - {missing}")
            
            if status.model_id:
                instructions.append(f"\nTo download this model:")
                instructions.append(f"  Model ID: {status.model_id}")
                instructions.append(f"\nOptions:")
                instructions.append(f"  1. Use the Models tab in the GUI to search and download")
                instructions.append(f"  2. Use Python:")
                instructions.append(f"     from huggingface_hub import snapshot_download")
                instructions.append(f"     snapshot_download(")
                instructions.append(f"         repo_id='{status.model_id}',")
                instructions.append(f"         local_dir='{status.model_path}',")
                instructions.append(f"         local_dir_use_symlinks=False")
                instructions.append(f"     )")
                instructions.append(f"  3. Use command line:")
                instructions.append(f"     huggingface-cli download {status.model_id} --local-dir {status.model_path}")
        
        return "\n".join(instructions)
    
    def create_models_readme(self, output_file: Optional[Path] = None) -> str:
        """Create a README file documenting all models
        
        Args:
            output_file: Path to write README. Defaults to MODELS_README.md in the models directory
            
        Returns:
            The README content
        """
        if output_file is None:
            output_file = self.models_dir.parent / "MODELS_README.md"
        
        all_models = self.check_all_models()
        incomplete_models = [m for m in all_models if not m.is_complete]
        complete_models = [m for m in all_models if m.is_complete]
        
        lines = []
        lines.append("# Model Installation Status")
        lines.append("")
        lines.append(f"**Total Models:** {len(all_models)}")
        lines.append(f"**Complete:** {len(complete_models)}")
        lines.append(f"**Incomplete:** {len(incomplete_models)}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        if complete_models:
            lines.append("## ✓ Complete Models")
            lines.append("")
            for model in sorted(complete_models, key=lambda m: m.model_name):
                size_str = f"{model.estimated_size_mb:.1f} MB" if model.estimated_size_mb else "Unknown size"
                lines.append(f"### {model.model_name}")
                if model.model_id:
                    lines.append(f"- **Model ID:** `{model.model_id}`")
                lines.append(f"- **Location:** `{model.model_path.relative_to(self.models_dir.parent)}`")
                lines.append(f"- **Size:** {size_str}")
                lines.append(f"- **Status:** ✓ Ready to use")
                lines.append("")
        
        if incomplete_models:
            lines.append("## ✗ Incomplete Models (Need Download)")
            lines.append("")
            lines.append("These models have directory structures but are missing the actual model weights.")
            lines.append("They need to be downloaded before use.")
            lines.append("")
            
            for model in sorted(incomplete_models, key=lambda m: m.model_name):
                lines.append(f"### {model.model_name}")
                if model.model_id:
                    lines.append(f"- **Model ID:** `{model.model_id}`")
                lines.append(f"- **Location:** `{model.model_path.relative_to(self.models_dir.parent)}`")
                lines.append(f"- **Missing:** {', '.join(model.missing_files)}")
                lines.append("")
                lines.append("**Download via GUI:**")
                lines.append("1. Open LLM Studio")
                lines.append("2. Go to Models tab")
                if model.model_id:
                    lines.append(f"3. Search for: `{model.model_id}`")
                lines.append("4. Click Download")
                lines.append("")
                
                if model.model_id:
                    lines.append("**Download via Command Line:**")
                    lines.append("```bash")
                    lines.append(f"huggingface-cli download {model.model_id} \\")
                    lines.append(f"  --local-dir {model.model_path} \\")
                    lines.append(f"  --local-dir-use-symlinks False")
                    lines.append("```")
                    lines.append("")
                    
                    lines.append("**Download via Python:**")
                    lines.append("```python")
                    lines.append("from huggingface_hub import snapshot_download")
                    lines.append(f"snapshot_download(")
                    lines.append(f"    repo_id='{model.model_id}',")
                    lines.append(f"    local_dir='{model.model_path}',")
                    lines.append(f"    local_dir_use_symlinks=False")
                    lines.append(")")
                    lines.append("```")
                    lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append("- Model weights (`.safetensors`, `.bin` files) are excluded from git due to their large size")
        lines.append("- When cloning this repository on a new machine, you must download the model weights")
        lines.append("- The GUI provides the easiest way to download missing models")
        lines.append("- Alternatively, use `huggingface-cli` or Python as shown above")
        lines.append("")
        lines.append(f"**Generated:** {self._get_timestamp()}")
        
        content = "\n".join(lines)
        
        # Write to file
        try:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Model README written to {output_file}")
        except Exception as e:
            logger.error(f"Could not write README: {e}")
        
        return content
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """CLI interface for model integrity checker"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check model installation integrity")
    parser.add_argument('--models-dir', type=Path, help="Directory containing models")
    parser.add_argument('--generate-readme', action='store_true', 
                       help="Generate MODELS_README.md file")
    parser.add_argument('--check-incomplete', action='store_true',
                       help="Only show incomplete models")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging with UTF-8 support
    import sys
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(message)s'
    )
    
    # Initialize checker
    checker = ModelIntegrityChecker(args.models_dir)
    
    if args.generate_readme:
        print("Generating model README...")
        content = checker.create_models_readme()
        print(f"\n✓ README generated successfully")
        return
    
    # Check models
    print("🔍 Checking model installations...\n")
    
    if args.check_incomplete:
        models = checker.get_incomplete_models()
        if not models:
            print("✓ All models are complete!")
            return
        print(f"Found {len(models)} incomplete models:\n")
    else:
        models = checker.check_all_models()
        print(f"Found {len(models)} total models:\n")
    
    # Display results
    for status in sorted(models, key=lambda m: (not m.is_complete, m.model_name)):
        icon = "✓" if status.is_complete else "✗"
        size_str = f" ({status.estimated_size_mb:.1f} MB)" if status.estimated_size_mb else ""
        print(f"{icon} {status.model_name}{size_str}")
        
        if not status.is_complete:
            print(f"   Missing: {', '.join(status.missing_files)}")
            if status.model_id:
                print(f"   Model ID: {status.model_id}")
        print()
    
    # Summary
    complete = sum(1 for m in models if m.is_complete)
    incomplete = len(models) - complete
    
    print(f"\n{'='*60}")
    print(f"Summary: {complete} complete, {incomplete} incomplete")
    
    if incomplete > 0:
        print(f"\n💡 Tip: Run with --generate-readme to create download instructions")
        print(f"   Or use the GUI Models tab to download missing models")


if __name__ == "__main__":
    main()

