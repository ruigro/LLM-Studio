"""
Tool calling infrastructure for LLM inference.

PHASE 4 REFACTOR: Strict JSON-only tool calling with schema validation.
Removed XML/Python parsers. Single envelope: {"tool": "name", "args": {}, "id": "uuid"}
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from enum import Enum


class ToolCallFormat(Enum):
    """Format of tool call in LLM output"""
    JSON = "json"  # PHASE 4: Only JSON supported
    UNKNOWN = "unknown"


@dataclass
class ToolCall:
    """Represents a detected tool call (PHASE 4: JSON only)"""
    name: str
    arguments: Dict[str, Any]
    format: ToolCallFormat
    raw_text: str  # Original text from LLM
    call_id: str  # Unique call ID


@dataclass
class ToolResult:
    """Result of tool execution"""
    success: bool
    result: Any
    error: Optional[str] = None


class ToolCallDetector:
    """
    Parse LLM output to detect tool calls (PHASE 4: JSON-only with schema validation).
    
    Expected format:
    {"tool": "calculator", "args": {"expression": "42*17"}, "id": "call_123"}
    """
    
    def __init__(self):
        """Initialize detector with JSON schema"""
        self.schema = self._load_schema()
    
    def _load_schema(self) -> dict:
        """Load tool call JSON schema"""
        try:
            schema_path = Path(__file__).parent.parent / "tools" / "schema.json"
            if schema_path.exists():
                return json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        
        # Fallback schema if file not found
        return {
            "type": "object",
            "required": ["tool", "args", "id"],
            "properties": {
                "tool": {"type": "string", "pattern": "^[a-zA-Z][a-zA-Z0-9_]*$"},
                "args": {"type": "object"},
                "id": {"type": "string"}
            }
        }
    
    def detect(self, text: str) -> List[ToolCall]:
        """
        Detect tool calls in LLM output (PHASE 4: JSON-only).
        
        Args:
            text: LLM output text
        
        Returns:
            List of detected tool calls
        """
        calls = []
        
        # Extract first valid JSON object
        tool_call_obj = self._extract_json_object(text)
        
        if tool_call_obj:
            # Validate against schema
            if self._validate_schema(tool_call_obj):
                calls.append(ToolCall(
                    name=tool_call_obj["tool"],
                    arguments=tool_call_obj["args"],
                    format=ToolCallFormat.JSON,
                    raw_text=json.dumps(tool_call_obj),
                    call_id=tool_call_obj["id"]
                ))
        
        return calls
    
    def _extract_json_object(self, text: str) -> Optional[dict]:
        """
        Extract first valid JSON object from text.
        
        Args:
            text: Text potentially containing JSON
        
        Returns:
            Parsed JSON object or None
        """
        # Try to find JSON object with brace matching
        brace_level = 0
        start_idx = None
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_level == 0:
                    start_idx = i
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0 and start_idx is not None:
                    # Found complete JSON object
                    json_str = text[start_idx:i+1]
                    try:
                        obj = json.loads(json_str)
                        # Check if it looks like a tool call (has required keys)
                        if "tool" in obj and "args" in obj and "id" in obj:
                            return obj
                    except json.JSONDecodeError:
                        # Try jsonfix repair
                        obj = self._repair_json(json_str)
                        if obj:
                            return obj
                    
                    # Reset for next object
                    start_idx = None
        
        return None
    
    def _repair_json(self, json_str: str) -> Optional[dict]:
        """
        Attempt to repair malformed JSON (ONE attempt).
        
        Args:
            json_str: Potentially malformed JSON string
        
        Returns:
            Parsed object or None if repair failed
        """
        try:
            # Try jsonfix if available
            import jsonfix
            obj = jsonfix.loads(json_str)
            
            # Validate it has required keys
            if isinstance(obj, dict) and "tool" in obj and "args" in obj and "id" in obj:
                return obj
        except ImportError:
            # jsonfix not available, try basic repairs
            try:
                # Common fixes: trailing commas, single quotes
                repaired = json_str.replace("'", '"')  # Single to double quotes
                repaired = re.sub(r',\s*}', '}', repaired)  # Trailing commas in objects
                repaired = re.sub(r',\s*]', ']', repaired)  # Trailing commas in arrays
                
                obj = json.loads(repaired)
                if isinstance(obj, dict) and "tool" in obj and "args" in obj and "id" in obj:
                    return obj
            except:
                pass
        except Exception:
            pass
        
        return None
    
    def _validate_schema(self, obj: dict) -> bool:
        """
        Validate JSON object against schema.
        
        Args:
            obj: JSON object to validate
        
        Returns:
            True if valid
        """
        try:
            # Try jsonschema if available
            import jsonschema
            jsonschema.validate(instance=obj, schema=self.schema)
            return True
        except ImportError:
            # Manual validation if jsonschema not available
            required = self.schema.get("required", [])
            for key in required:
                if key not in obj:
                    return False
            
            # Check tool name pattern
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', obj.get("tool", "")):
                return False
            
            # Check args is dict
            if not isinstance(obj.get("args"), dict):
                return False
            
            return True
        except Exception:
            return False


class ToolExecutor:
    """Execute tools by calling the tool server"""
    
    def __init__(self, server_url: str, auth_token: Optional[str] = None):
        """
        Args:
            server_url: Base URL of tool server (e.g., http://127.0.0.1:8763)
            auth_token: Optional authentication token
        """
        self.server_url = server_url.rstrip('/')
        self.auth_token = auth_token
    
    def execute(self, tool_call: ToolCall, timeout: int = 30) -> ToolResult:
        """
        Execute a tool call.
        
        Args:
            tool_call: Tool call to execute
            timeout: Timeout in seconds
            
        Returns:
            ToolResult with success status and result/error
        """
        try:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["X-Auth-Token"] = self.auth_token
            
            payload = {
                "name": tool_call.name,
                "args": tool_call.arguments
            }
            
            req = urllib.request.Request(
                f"{self.server_url}/call",
                headers=headers,
                data=json.dumps(payload).encode("utf-8"),
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                
                if result.get("ok"):
                    return ToolResult(
                        success=True,
                        result=result.get("result")
                    )
                else:
                    return ToolResult(
                        success=False,
                        result=None,
                        error=result.get("error", "Unknown error")
                    )
        
        except urllib.error.HTTPError as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"HTTP {e.code}: {e.reason}"
            )
        except urllib.error.URLError as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Connection error: {e.reason}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )


class ToolApprovalManager:
    """Manage user approval for tool execution"""
    
    # Danger levels from tool server
    DANGER_LEVELS = {
        "safe": ["read_file", "list_dir", "git_status"],
        "warning": ["write_file"],
        "dangerous": ["run_shell"]
    }
    
    def __init__(self, auto_execute_safe: bool = True):
        """
        Args:
            auto_execute_safe: If True, auto-execute safe tools without asking
        """
        self.auto_execute_safe = auto_execute_safe
        self.session_approvals: Dict[str, bool] = {}  # tool_name -> approved
    
    def requires_approval(self, tool_name: str) -> bool:
        """Check if tool requires user approval"""
        # Check session approvals first
        if tool_name in self.session_approvals:
            return not self.session_approvals[tool_name]
        
        # Auto-execute safe tools if enabled
        if self.auto_execute_safe and tool_name in self.DANGER_LEVELS["safe"]:
            return False
        
        # All other tools require approval
        return True
    
    def get_danger_level(self, tool_name: str) -> str:
        """Get danger level of tool"""
        for level, tools in self.DANGER_LEVELS.items():
            if tool_name in tools:
                return level
        return "unknown"
    
    def approve(self, tool_name: str, remember: bool = False):
        """Approve a tool for execution"""
        if remember:
            self.session_approvals[tool_name] = True
    
    def deny(self, tool_name: str, remember: bool = False):
        """Deny a tool for execution"""
        if remember:
            self.session_approvals[tool_name] = False
    
    def clear_session_approvals(self):
        """Clear all session approvals"""
        self.session_approvals.clear()


def format_tool_result_for_llm(tool_call: ToolCall, result: ToolResult) -> str:
    """
    Format tool result for feeding back to LLM.
    
    Args:
        tool_call: The original tool call
        result: Result of execution
        
    Returns:
        Formatted string to append to conversation
    """
    if result.success:
        result_text = json.dumps(result.result, indent=2) if result.result else "Success"
        return f"\n<tool_result tool=\"{tool_call.name}\">\n{result_text}\n</tool_result>\n"
    else:
        return f"\n<tool_result tool=\"{tool_call.name}\" error=\"true\">\n{result.error}\n</tool_result>\n"
