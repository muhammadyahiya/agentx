"""Tools: MCP tool loading + a few built-in tools."""
from .mcp import load_mcp_tools
from .builtin import make_web_search_tool

__all__ = ["load_mcp_tools", "make_web_search_tool"]
