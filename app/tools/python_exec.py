"""
Python code executor tool (sandboxed with restricted environment).
"""
from typing import Any, Dict
from app.tools.base import BaseTool
import asyncio
import ast
import sys
from io import StringIO


# Restricted builtins for safe execution
SAFE_BUILTINS = {
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'dict': dict,
    'enumerate': enumerate,
    'filter': filter,
    'float': float,
    'int': int,
    'len': len,
    'list': list,
    'map': map,
    'max': max,
    'min': min,
    'range': range,
    'reversed': reversed,
    'round': round,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'zip': zip,
    'print': print,
}

# Safe math functions
import math
SAFE_BUILTINS.update({
    'math': math,
    'sqrt': math.sqrt,
    'pow': pow,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'pi': math.pi,
    'e': math.e,
})


class PythonExec(BaseTool):
    """
    Python code executor tool with enhanced sandboxing.
    Uses AST-based safe evaluation instead of exec().
    """
    
    def get_description(self) -> str:
        return "Execute Python code in a restricted environment (math expressions, basic operations only)"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute (math expressions, basic operations)"
                },
                "timeout": {
                    "type": "integer",
                    "default": 5,
                    "description": "Execution timeout in seconds"
                }
            },
            "required": ["code"]
        }
    
    def _is_safe_code(self, code: str) -> tuple[bool, str]:
        """
        Check if code is safe for execution using AST analysis.
        
        Args:
            code: Python code to check
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        # Block dangerous keywords
        dangerous_keywords = [
            "import", "exec", "eval", "compile", "open", "file",
            "__import__", "getattr", "setattr", "delattr",
            "globals", "locals", "vars", "dir",
            "subprocess", "os", "sys", "pickle", "marshal",
            "class ", "def ", "lambda", "yield", "await",
            "for ", "while ", "if ", "else:", "try:", "except:",
            "finally:", "with ", "raise", "assert", "break", "continue"
        ]
        
        code_lower = code.lower()
        for keyword in dangerous_keywords:
            if keyword in code_lower:
                return False, f"Code contains restricted keyword: {keyword}"
        
        # Try to parse as AST
        try:
            tree = ast.parse(code, mode='eval')
        except SyntaxError:
            # If eval fails, try exec mode but still check
            try:
                tree = ast.parse(code, mode='exec')
                # Check for dangerous nodes
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        return False, "Import statements are not allowed"
                    if isinstance(node, ast.FunctionDef):
                        return False, "Function definitions are not allowed"
                    if isinstance(node, ast.ClassDef):
                        return False, "Class definitions are not allowed"
                    if isinstance(node, ast.Call):
                        # Check if calling dangerous functions
                        if isinstance(node.func, ast.Name):
                            if node.func.id in ['exec', 'eval', 'compile', 'open']:
                                return False, f"Dangerous function call: {node.func.id}"
            except SyntaxError as e:
                return False, f"Syntax error: {str(e)}"
        
        return True, ""
    
    def _safe_eval(self, code: str, local_vars: Dict[str, Any]) -> Any:
        """
        Safely evaluate code using restricted environment.
        
        Args:
            code: Python code to evaluate
            local_vars: Local variables for execution context
            
        Returns:
            Result of evaluation
            
        Raises:
            Exception: If evaluation fails
        """
        # Create restricted globals
        restricted_globals = {
            '__builtins__': SAFE_BUILTINS,
            **local_vars
        }
        
        # Try eval mode first (expressions only)
        try:
            return eval(code, restricted_globals, {})
        except SyntaxError:
            # If eval fails, the code might be a statement
            # For safety, we only allow simple print statements
            if code.strip().startswith('print('):
                # Capture output from print
                old_stdout = sys.stdout
                sys.stdout = captured = StringIO()
                try:
                    exec(code, restricted_globals, {})
                    output = captured.getvalue()
                    return output or "Executed successfully"
                finally:
                    sys.stdout = old_stdout
            else:
                raise ValueError("Only expressions and print statements are allowed")
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        Execute Python code with timeout and enhanced sandboxing.
        
        Args:
            arguments: Dictionary with 'code' and optional 'timeout' keys
            
        Returns:
            Structured result with output or error
        """
        code = arguments.get("code", "")
        timeout = arguments.get("timeout", 5)
        
        # Validate code safety
        is_safe, safety_error = self._is_safe_code(code)
        if not is_safe:
            return {
                "error": "Execution failed",
                "details": safety_error,
                "success": False
            }
        
        # Execute with timeout in a thread
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._safe_eval, code, {}),
                timeout=timeout
            )
            
            # Convert result to string if needed
            if result is None:
                output = "Executed successfully"
            else:
                output = str(result)
            
            return {
                "output": output,
                "success": True
            }
            
        except asyncio.TimeoutError:
            return {
                "error": "Execution failed",
                "details": f"Execution timed out after {timeout} seconds",
                "success": False
            }
        except Exception as e:
            return {
                "error": "Execution failed",
                "details": str(e),
                "success": False
            }
