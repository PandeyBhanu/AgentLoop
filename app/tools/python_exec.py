"""
Python code evaluator tool.

SECURITY NOTE: this tool evaluates LLM-supplied expressions inside the
agent process. A restricted namespace is NOT a secure sandbox — Python
object introspection can escape built-in restrictions. For that reason:

- The tool is DISABLED BY DEFAULT. Set AGENTLOOP_ENABLE_PYTHON_EXEC=1
  (or 'true'/'yes') to enable it.
- When enabled, code is validated with an AST *whitelist* (not a keyword
  blacklist): only expression nodes, constants, whitelisted names, and
  whitelisted call targets are permitted. Attribute access is limited to
  the `math` module and dunder attributes are always rejected.
- Execution runs in a thread with a timeout, stdout is captured, and
  output is truncated.

This is defense-in-depth for math/utility expressions, not a boundary
against malicious code. Do NOT enable on untrusted deployments.
"""
from typing import Any, Dict
from app.tools.base import BaseTool
import asyncio
import ast
import math
import os
import sys
from io import StringIO


# Restricted builtins for safe evaluation
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

MAX_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 10_000

# AST node types permitted in evaluated expressions.
_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.Name, ast.Load, ast.Store,
    ast.Attribute,
    ast.Tuple, ast.List, ast.Set, ast.Dict,
    ast.Subscript, ast.Slice, ast.Starred,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.comprehension, ast.keyword,
    # operator / comparator nodes
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
    ast.UAdd, ast.USub, ast.Invert, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
)


class PythonExec(BaseTool):
    """
    Expression evaluator with AST whitelisting.

    Disabled unless AGENTLOOP_ENABLE_PYTHON_EXEC is set, because a
    restricted namespace alone is not a secure sandbox.
    """

    def get_description(self) -> str:
        return "Evaluate Python math/utility expressions in a restricted environment"

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python expression to evaluate (math expressions, basic operations)"
                },
                "timeout": {
                    "type": "integer",
                    "default": 5,
                    "description": f"Execution timeout in seconds (max {MAX_TIMEOUT_SECONDS})"
                }
            },
            "required": ["code"]
        }

    @staticmethod
    def is_enabled() -> bool:
        return os.getenv("AGENTLOOP_ENABLE_PYTHON_EXEC", "").lower() in ("1", "true", "yes")

    def _is_safe_code(self, code: str) -> tuple[bool, str]:
        """
        Validate code with an AST whitelist.

        Permitted: expressions over constants, whitelisted builtin names,
        whitelisted calls, math.* attributes, literals/comprehensions.
        Rejected: statements (import/def/class/assign/...), attribute access
        outside `math`, dunder attributes, and any non-whitelisted name.
        """
        try:
            tree = ast.parse(code, mode='eval')
        except SyntaxError:
            return False, "Only expressions are allowed (statements are not permitted)"
        except (ValueError, MemoryError) as e:
            return False, f"Could not parse code: {e}"

        # Names introduced by comprehension targets (e.g. `x` in
        # `sum(x*x for x in range(10))`) are legal load names.
        comp_targets = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.comprehension):
                for t in ast.walk(node.target):
                    if isinstance(t, ast.Name):
                        comp_targets.add(t.id)

        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                return False, f"Disallowed syntax: {type(node).__name__}"

            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    continue  # comprehension target
                if node.id not in SAFE_BUILTINS and node.id not in comp_targets:
                    return False, f"Unknown or disallowed name: {node.id}"

            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    return False, "Access to private/dunder attributes is not allowed"
                if not (isinstance(node.value, ast.Name) and node.value.id == "math"):
                    return False, "Attribute access is only allowed on 'math'"

            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    if func.id not in SAFE_BUILTINS:
                        return False, f"Disallowed function call: {func.id}"
                elif isinstance(func, ast.Attribute):
                    if not (isinstance(func.value, ast.Name) and func.value.id == "math"):
                        return False, "Only math.* calls are allowed"
                else:
                    return False, "Only direct calls to safe functions are allowed"

        return True, ""

    def _safe_eval(self, code: str, local_vars: Dict[str, Any]) -> Any:
        """
        Evaluate code in a restricted namespace, capturing stdout so that
        print() output is returned to the caller instead of leaking to the
        server's stdout.
        """
        restricted_globals = {
            '__builtins__': SAFE_BUILTINS,
            **local_vars,
        }

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            result = eval(code, restricted_globals, {})
        finally:
            sys.stdout = old_stdout

        printed = captured.getvalue()
        if result is None:
            return printed or "Executed successfully"
        return printed + str(result) if printed else result

    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        Evaluate an expression with timeout, structured errors, and an
        output cap. No-ops with a structured error when disabled.
        """
        if not self.is_enabled():
            return {
                "error": "Execution disabled",
                "details": "PythonExec is disabled by default. Set "
                           "AGENTLOOP_ENABLE_PYTHON_EXEC=1 to enable it.",
                "success": False
            }

        code = arguments.get("code", "")
        try:
            timeout = int(arguments.get("timeout", 5))
        except (TypeError, ValueError):
            timeout = 5
        timeout = max(1, min(timeout, MAX_TIMEOUT_SECONDS))

        is_safe, safety_error = self._is_safe_code(code)
        if not is_safe:
            return {
                "error": "Execution failed",
                "details": safety_error,
                "success": False
            }

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._safe_eval, code, {}),
                timeout=timeout
            )

            if result is None:
                output = "Executed successfully"
            else:
                output = str(result)
            if len(output) > MAX_OUTPUT_CHARS:
                output = output[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"

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
            # Return the exception class + message only — no internals.
            return {
                "error": "Execution failed",
                "details": f"{type(e).__name__}: {e}",
                "success": False
            }
