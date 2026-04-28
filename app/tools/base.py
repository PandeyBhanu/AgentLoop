"""
Base class for all tools in the agent system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel
import json


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    All tools must inherit from this class and implement the execute method.
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = self.get_description()
        self.input_schema = self.get_input_schema()
    
    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        Execute the tool with the given arguments.
        
        Args:
            arguments: Dictionary of arguments for the tool
            
        Returns:
            The result of the tool execution
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of what this tool does.
        
        Returns:
            String description of the tool
        """
        pass
    
    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get the JSON Schema for tool inputs.
        
        Returns:
            Dictionary representing the JSON Schema
        """
        pass
    
    def to_definition(self) -> Dict[str, Any]:
        """
        Convert tool to a definition dictionary.
        
        Returns:
            Dictionary with tool name, description, and input schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }
