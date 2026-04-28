"""
Pydantic schemas for tool definitions and validation.
"""
from pydantic import BaseModel, Field
from typing import Any, Dict
from jsonschema import Draft7Validator


class ToolDefinition(BaseModel):
    """Definition of a tool including its schema."""
    name: str = Field(..., description="Unique name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    input_schema: Dict[str, Any] = Field(..., description="JSON Schema for tool inputs")


class ToolValidationResult(BaseModel):
    """Result of validating tool inputs against schema."""
    valid: bool = Field(..., description="Whether the inputs are valid")
    errors: list[str] = Field(default_factory=list, description="Validation error messages")


class ToolInput(BaseModel):
    """Base class for tool input validation."""
    pass


def validate_tool_inputs(arguments: Dict[str, Any], schema: Dict[str, Any]) -> ToolValidationResult:
    """
    Validate tool arguments against JSON schema.
    
    Args:
        arguments: The arguments to validate
        schema: The JSON schema to validate against
        
    Returns:
        ToolValidationResult with validation status and errors
    """
    validator = Draft7Validator(schema)
    errors = []
    
    for error in validator.iter_errors(arguments):
        errors.append(f"{error.path}: {error.message}")
    
    return ToolValidationResult(valid=len(errors) == 0, errors=errors)
