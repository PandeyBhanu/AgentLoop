"""
Base class for all tools in the agent system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


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
    
    def get_output_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    @property
    def output_schema(self) -> Dict[str, Any]:
        return self.get_output_schema()

    @property
    def timeout(self) -> int:
        return 30

    @property
    def risk_level(self) -> str:
        return "low"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {}

    def to_definition(self) -> Dict[str, Any]:
        """
        Convert tool to a definition dictionary.
        
        Returns:
            Dictionary with tool name, description, and input schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "timeout": self.timeout,
            "risk_level": self.risk_level,
            "metadata": self.metadata
        }
