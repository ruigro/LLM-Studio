"""
Tool calling infrastructure for LLM inference.

Supports multiple formats:
1. Native JSON function calling
2. XML-style tags (prompted)
3. Python-style calls (alternative)
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable
from enum import Enum


class ToolCallFormat(Enum):
    """Format of tool call in LLM output"""
    NATIVE_JSON = "native_json"
    XML_STYLE = "xml_style"
    PYTHON_STYLE = "python_style"
    UNKNOWN = "unknown"


@dataclass
class ToolCall:
    """Represents a detected tool call"""
    name: str
    arguments: Dict[str, Any]
    format: ToolCallFormat
    raw_text: str  # Original text from LLM


@dataclass
class ToolResult:
    """Result of tool execution"""
    success: bool
    result: Any
    error: Optional[str] = None


class ToolCallDetector:
    """Parse LLM output to detect tool calls in various formats"""
    
    # Regex patterns for different formats
    XML_PATTERN = r'<tool_call>(.*?)</tool_call>'
    PYTHON_PATTERN = r'(\w+)\((.*?)\)'
    
    @staticmethod
    def detect(text: str) -> List[ToolCall]:
        """
        Detect tool calls in LLM output.
        Returns list of detected tool calls.
        """
        calls = []
        
        # Try native JSON format first
        calls.extend(ToolCallDetector._detect_json(text))
        
        # Try XML-style tags
        calls.extend(ToolCallDetector._detect_xml(text))
        
        # Try Python-style calls (only if no other format found)
        if not calls:
            calls.extend(ToolCallDetector._detect_python(text))
        
        return calls
    
    @staticmethod
    def _detect_json(text: str) -> List[ToolCall]:
        """Detect native JSON function calls"""
        calls = []
        
        # Look for JSON objects with "name" and "arguments" keys
        # This matches OpenAI function calling format
        json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
        
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                full_match = match.group(0)
                obj = json.loads(full_match)
                name = obj.get("name")
                args = obj.get("arguments", {})
                
                if name:
                    calls.append(ToolCall(
                        name=name,
                        arguments=args if isinstance(args, dict) else {},
                        format=ToolCallFormat.NATIVE_JSON,
                        raw_text=full_match
                    ))
            except json.JSONDecodeError:
                continue
        
        return calls
    
    @staticmethod
    def _detect_xml(text: str) -> List[ToolCall]:
        """Detect XML-style tool calls: <tool_call>tool_name(arg="val")</tool_call>"""
        calls = []
        
        for match in re.finditer(ToolCallDetector.XML_PATTERN, text, re.DOTALL):
            content = match.group(1).strip()
            
            # Parse the content as Python-style call
            python_match = re.match(r'(\w+)\((.*?)\)', content, re.DOTALL)
            if python_match:
                tool_name = python_match.group(1)
                args_str = python_match.group(2)
                
                # Parse arguments
                args = ToolCallDetector._parse_arguments(args_str)
                
                calls.append(ToolCall(
                    name=tool_name,
                    arguments=args,
                    format=ToolCallFormat.XML_STYLE,
                    raw_text=match.group(0)
                ))
        
        return calls
    
    @staticmethod
    def _detect_python(text: str) -> List[ToolCall]:
        """Detect Python-style calls (last resort, prone to false positives)"""
        # Only look for known tool names to avoid false positives
        known_tools = ['read_file', 'write_file', 'list_dir', 'run_shell', 'git_status']
        calls = []
        
        for tool_name in known_tools:
            pattern = rf'\b{tool_name}\((.*?)\)'
            for match in re.finditer(pattern, text, re.DOTALL):
                args_str = match.group(1)
                args = ToolCallDetector._parse_arguments(args_str)
                
                calls.append(ToolCall(
                    name=tool_name,
                    arguments=args,
                    format=ToolCallFormat.PYTHON_STYLE,
                    raw_text=match.group(0)
                ))
        
        return calls
    
    @staticmethod
    def _parse_arguments(args_str: str) -> Dict[str, Any]:
        """Parse argument string into dictionary"""
        args = {}
        
        if not args_str.strip():
            return args
        
        # Try parsing as JSON first
        try:
            parsed = json.loads(f'{{{args_str}}}')
            if isinstance(parsed, dict):
                return parsed
        except:
            pass
        
        # Parse key=value pairs
        # Supports: key="value", key='value', key=value
        arg_pattern = r'(\w+)\s*=\s*(["\'])(.*?)\2|(\w+)\s*=\s*(\S+)'
        
        for match in re.finditer(arg_pattern, args_str):
            if match.group(1):  # Quoted value
                key = match.group(1)
                value = match.group(3)
            else:  # Unquoted value
                key = match.group(4)
                value = match.group(5)
            
            # Try to parse value as JSON (for booleans, numbers, etc.)
            try:
                value = json.loads(value)
            except:
                pass  # Keep as string
            
            args[key] = value
        
        return args


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
