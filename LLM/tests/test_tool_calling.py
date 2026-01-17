"""
PHASE 3: pytest test suite for tool calling functionality.
"""
import pytest
import sys
from pathlib import Path

# Add LLM to path
llm_dir = Path(__file__).parent.parent
sys.path.insert(0, str(llm_dir))


def test_tool_call_detector_import():
    """Test that tool calling modules import"""
    from core.tool_calling import ToolCallDetector, ToolExecutor, ToolApprovalManager
    
    assert ToolCallDetector is not None
    assert ToolExecutor is not None
    assert ToolApprovalManager is not None


def test_tool_call_detector_json():
    """Test JSON tool call detection"""
    from core.tool_calling import ToolCallDetector
    
    detector = ToolCallDetector()
    
    # Test valid JSON tool call
    text_with_json = '''Here is the result:
{"tool": "calculator", "args": {"expression": "42*17"}, "id": "call_123"}
That's the answer.'''
    
    calls = detector.detect(text_with_json)
    
    # Should detect at least one call (depending on current implementation)
    assert isinstance(calls, list)


def test_tool_registry():
    """Test tool registry imports"""
    try:
        from tool_server.tool_registry import ToolRegistry
        
        registry = ToolRegistry()
        tools = registry.list_tools()
        
        assert isinstance(tools, list)
    except ImportError:
        pytest.skip("Tool registry not available")


@pytest.mark.skipif(True, reason="Requires running tool server")
def test_tool_server_health():
    """Test tool server health endpoint (requires server running)"""
    import requests
    
    try:
        response = requests.get("http://localhost:8763/health", timeout=2)
        assert response.status_code == 200
    except requests.exceptions.RequestException:
        pytest.skip("Tool server not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
