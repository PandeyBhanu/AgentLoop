"""
Tool registry for managing available tools.
"""
from typing import Any, Dict, Optional
from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError
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
        if not isinstance(tool, BaseTool):
            raise TypeError("Tool must implement BaseTool")
        if not isinstance(tool.name, str) or not tool.name or not isinstance(tool.description, str) or not tool.description.strip():
            raise ValueError("Tool name and description are required")
        if tool.risk_level not in ("low", "medium", "high") or not isinstance(tool.timeout, (int, float)) or tool.timeout <= 0:
            raise ValueError("Tool risk level and positive timeout are required")
        if not isinstance(tool.metadata, dict):
            raise ValueError("Tool metadata must be an object")
        for schema in (tool.input_schema, tool.output_schema):
            try:
                Draft7Validator.check_schema(schema)
            except (SchemaError, TypeError) as exc:
                raise ValueError(f"Invalid tool schema: {exc}") from exc
            if schema.get("type") != "object":
                raise ValueError("Tool input and output schemas must describe objects")
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
    
    def get_all_definitions(self) -> list[Dict[str, Any]]:
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
