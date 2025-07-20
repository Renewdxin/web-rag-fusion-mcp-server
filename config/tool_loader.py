"""Tool configuration loader for MCP server."""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from mcp.types import Tool


class ToolConfigLoader:
    """Loads tool configurations from YAML files."""
    
    def __init__(self, config_path: Path = None):
        """Initialize the tool config loader.
        
        Args:
            config_path: Path to the tools.yaml file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "tools.yaml"
        self.config_path = config_path
        self._tools_config = None
    
    def load_tools_config(self) -> Dict[str, Any]:
        """Load tools configuration from YAML file."""
        if self._tools_config is None:
            try:
                with open(self.config_path, 'r') as f:
                    self._tools_config = yaml.safe_load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load tools config from {self.config_path}: {e}")
        return self._tools_config
    
    def get_tool_definitions(self) -> List[Tool]:
        """Get tool definitions as MCP Tool objects."""
        config = self.load_tools_config()
        tools = []
        
        for tool_name, tool_config in config.get("tools", {}).items():
            tool = Tool(
                name=tool_name,
                description=tool_config.get("description", ""),
                inputSchema=tool_config.get("inputSchema", {})
            )
            tools.append(tool)
        
        return tools
    
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get the input schema for a specific tool."""
        config = self.load_tools_config()
        tools = config.get("tools", {})
        
        if tool_name not in tools:
            raise ValueError(f"Tool '{tool_name}' not found in configuration")
        
        return tools[tool_name].get("inputSchema", {})