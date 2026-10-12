"""Disabled-by-default Python expression execution in a disposable Docker container."""
import asyncio
import ast
import os
import shutil
import uuid
from typing import Any, Dict

from app.tools.base import BaseTool

MAX_TIMEOUT_SECONDS = 30
MAX_OUTPUT_BYTES = 10_000
MAX_CODE_CHARS = 4000
WORKER_IMAGE = "agentloop-python-worker:local"

_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.Call, ast.Name, ast.Load, ast.Store, ast.Attribute,
    ast.Tuple, ast.List, ast.Set, ast.Dict, ast.Subscript, ast.Slice, ast.Starred,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.keyword, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.FloorDiv, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
    ast.UAdd, ast.USub, ast.Invert, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
)
_ALLOWED_NAMES = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "range", "reversed",
    "round", "sorted", "str", "sum", "tuple", "zip", "print", "math",
    "sqrt", "pow", "sin", "cos", "tan", "pi", "e",
}

_WORKER_SCRIPT = """import sys,math
safe={name:getattr(__builtins__,name) for name in ('abs','all','any','bool','dict','enumerate','filter','float','int','len','list','map','max','min','range','reversed','round','sorted','str','sum','tuple','zip','print')}
safe.update(math=math,sqrt=math.sqrt,pow=pow,sin=math.sin,cos=math.cos,tan=math.tan,pi=math.pi,e=math.e)
code=sys.stdin.read(4001)
result=eval(compile(code,'<expression>','eval'),{'__builtins__':safe},{})
if result is not None: print(result)
"""


class PythonExec(BaseTool):
    @property
    def risk_level(self) -> str:
        return "high"

    @property
    def timeout(self) -> int:
        return MAX_TIMEOUT_SECONDS

    def get_description(self) -> str:
        return "Evaluate a restricted Python expression in a disposable container (disabled by default)"

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "minLength": 1, "maxLength": MAX_CODE_CHARS},
                "timeout": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_SECONDS, "default": 15},
            },
            "required": ["code"],
            "additionalProperties": False,
        }

    @staticmethod
    def is_enabled() -> bool:
        return os.getenv("AGENTLOOP_ENABLE_PYTHON_EXEC", "").lower() in ("1", "true", "yes")

    def _is_safe_code(self, code: str) -> tuple[bool, str]:
        if len(code) > MAX_CODE_CHARS:
            return False, "Expression too long"
        try:
            tree = ast.parse(code, mode="eval")
        except (SyntaxError, ValueError, MemoryError):
            return False, "Only valid expressions are allowed"
        comp_targets = {
            target.id
            for node in ast.walk(tree) if isinstance(node, ast.comprehension)
            for target in ast.walk(node.target) if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                return False, f"Disallowed syntax: {type(node).__name__}"
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store):
                if node.id not in _ALLOWED_NAMES and node.id not in comp_targets:
                    return False, f"Unknown or disallowed name: {node.id}"
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_") or not (isinstance(node.value, ast.Name) and node.value.id == "math"):
                    return False, "Only public math attributes are allowed"
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id not in _ALLOWED_NAMES:
                    return False, "Disallowed call"
                if not isinstance(func, (ast.Name, ast.Attribute)):
                    return False, "Only direct calls are allowed"
        return True, ""

    async def _stop_container(self, name: str, proc: asyncio.subprocess.Process) -> None:
        proc.kill()
        await proc.wait()
        try:
            killer = await asyncio.create_subprocess_exec(
                "docker", "kill", name,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), 5)
        except (OSError, asyncio.TimeoutError):
            pass

    @staticmethod
    def _docker_args(name: str) -> list[str]:
        return [
            "docker", "run", "--rm", "--name", name, "-i",
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "16",
            "--memory", "128m", "--memory-swap", "128m", "--cpus", "0.5",
            "--user", "65534:65534", "--workdir", "/tmp",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--ulimit", "nofile=32:32", "--ulimit", "fsize=1048576:1048576",
            WORKER_IMAGE,
        ]

    async def _run_container(self, code: str, timeout: int) -> str:
        name = f"agentloop-{uuid.uuid4().hex}"
        proc = await asyncio.create_subprocess_exec(
            *self._docker_args(name), "python", "-I", "-B", "-c", _WORKER_SCRIPT,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            proc.stdin.write(code.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
            output = await asyncio.wait_for(proc.stdout.read(MAX_OUTPUT_BYTES + 1), timeout)
            if len(output) > MAX_OUTPUT_BYTES:
                await self._stop_container(name, proc)
                return "Output limit exceeded"
            await asyncio.wait_for(proc.wait(), max(1, timeout))
            if proc.returncode:
                return "Execution failed inside worker"
            return output.decode("utf-8", errors="replace").strip() or "Executed successfully"
        except BaseException:
            if proc.returncode is None:
                await asyncio.shield(self._stop_container(name, proc))
            raise

    async def execute(self, arguments: Dict[str, Any]) -> Any:
        if not self.is_enabled():
            return {"error": "Execution disabled", "details": "PythonExec is disabled by default", "success": False}
        code = arguments.get("code", "")
        timeout = arguments.get("timeout", 15)
        if not isinstance(code, str) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
            return {"error": "Invalid input", "success": False}
        safe, reason = self._is_safe_code(code)
        if not safe:
            return {"error": "Execution failed", "details": reason, "success": False}
        if not shutil.which("docker"):
            return {"error": "Execution unavailable", "details": "Docker CLI is required", "success": False}
        try:
            output = await self._run_container(code, timeout)
            if output in ("Output limit exceeded", "Execution failed inside worker"):
                return {"error": "Execution failed", "details": output, "success": False}
            return {"output": output, "success": True}
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return {"error": "Execution failed", "details": f"Execution timed out after {timeout} seconds", "success": False}
        except OSError:
            return {"error": "Execution unavailable", "details": "Could not start isolated worker", "success": False}
