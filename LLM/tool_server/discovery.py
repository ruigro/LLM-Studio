"""
Automatic tool discovery from tool modules.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def discover_tools() -> None:
    """
    Discover and import all tools from LLM.tool_server.tools package.
    This will trigger tool registration via decorators.
    """
    try:
        # Import the tools package
        tools_package = "tool_server.tools"
        
        try:
            package = importlib.import_module(tools_package)
        except ImportError as e:
            import sys
            sys.stderr.write(f"[tool discovery] Could not import {tools_package}: {e}\n")
            return
        
        # Get the package path
        if not hasattr(package, "__path__"):
            return
        
        package_path = Path(package.__path__[0])

        for finder, name, ispkg in pkgutil.iter_modules([str(package_path)]):
            if not ispkg and not name.startswith("_"):
                try:
                    module_name = f"{tools_package}.{name}"
                    importlib.import_module(module_name)
                except Exception as e:
                    # Log but don't fail - some modules might have optional dependencies
                    import sys
                    sys.stderr.write(f"[tool discovery] Could not import {name}: {e}\n")
    except Exception as e:
        import sys
        sys.stderr.write(f"[tool discovery] Error during discovery: {e}\n")
