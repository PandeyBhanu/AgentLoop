"""
Calculator tool for basic arithmetic operations.
"""
from typing import Any, Dict
from app.tools.base import BaseTool
import operator
import math


class Calculator(BaseTool):
    """
    Calculator tool for performing basic arithmetic operations.
    """
    
    def get_description(self) -> str:
        return "Perform basic arithmetic operations: add, subtract, multiply, divide, factorial"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "factorial"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First operand (or the number for factorial)"
                },
                "b": {
                    "type": "number",
                    "description": "Second operand (not required for factorial)"
                }
            },
            "required": ["operation", "a"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        Execute the calculator operation.
        
        Args:
            arguments: Dictionary with 'operation', 'a', and optionally 'b' keys
            
        Returns:
            Result of the arithmetic operation
        """
        operation = arguments.get("operation")
        a = arguments.get("a")
        b = arguments.get("b")
        
        # Handle factorial separately (only needs one operand)
        if operation == "factorial":
            try:
                result = math.factorial(int(a))
                return {"result": result, "operation": operation}
            except Exception as e:
                return {"error": str(e)}
        
        # Handle binary operations
        if b is None:
            return {"error": f"Operation '{operation}' requires both 'a' and 'b' operands"}
        
        operations = {
            "add": operator.add,
            "subtract": operator.sub,
            "multiply": operator.mul,
            "divide": operator.truediv
        }
        
        if operation not in operations:
            raise ValueError(f"Invalid operation: {operation}")
        
        op_func = operations[operation]
        
        try:
            result = op_func(a, b)
            return {"result": result, "operation": operation}
        except ZeroDivisionError:
            return {"error": "Division by zero"}
        except Exception as e:
            return {"error": str(e)}
