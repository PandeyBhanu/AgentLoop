"""
Tool registry for managing available tools.
"""
from typing import Dict, Optional
from app.tools.base import BaseTool


class ToolRegistry:
    """
    Registry for managing and retrieving tools.
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Instance of BaseTool to register
            
        Raises:
            ValueError: If a tool with the same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def get(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            BaseTool instance if found, None otherwise
        """
        return self._tools.get(tool_name)
    
    def get_all(self) -> Dict[str, BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of all registered tools
        """
        return self._tools.copy()
    
    def get_all_definitions(self) -> list[Dict[str, any]]:
        """
        Get all tool definitions.
        
        Returns:
            List of tool definition dictionaries
        """
        return [tool.to_definition() for tool in self._tools.values()]
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool is registered, False otherwise
        """
        return tool_name in self._tools
    
    def list_tool_names(self) -> list[str]:
        """
        Get list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
