"""
Tests for the tool system: registry, validation, and each tool's execution.
"""
import asyncio
import time

import pytest

from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry
from app.tools.calculator import Calculator
from app.tools.web_search import WebSearch
from app.tools.file_reader import FileReader
from app.tools.python_exec import PythonExec
from app.schemas.tools import validate_tool_inputs


class DummyTool(BaseTool):
    """Minimal tool for registry/timeout tests."""

    def get_description(self):
        return "A dummy tool"

    def get_input_schema(self):
        return {"type": "object", "properties": {}}

    async def execute(self, arguments):
        return {"ok": True}


class SlowTool(BaseTool):
    def __init__(self, delay=5.0):
        self._delay = delay
        super().__init__()

    def get_description(self):
        return "A tool that sleeps"

    def get_input_schema(self):
        return {"type": "object", "properties": {}}

    async def execute(self, arguments):
        await asyncio.sleep(self._delay)
        return {"ok": True}


class FailingTool(BaseTool):
    def get_description(self):
        return "A tool that raises"

    def get_input_schema(self):
        return {"type": "object", "properties": {}}

    async def execute(self, arguments):
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_register_and_lookup(self, registry):
        assert registry.has_tool("Calculator")
        assert registry.get("Calculator").name == "Calculator"

    def test_unknown_tool_returns_none(self, registry):
        assert registry.get("NoSuchTool") is None
        assert not registry.has_tool("NoSuchTool")

    def test_duplicate_registration_raises(self, registry):
        with pytest.raises(ValueError, match="already registered"):
            registry.register(Calculator())

    def test_list_tool_names(self, registry):
        registry.register(DummyTool())
        assert sorted(registry.list_tool_names()) == ["Calculator", "DummyTool"]

    def test_get_all_returns_copy(self, registry):
        tools = registry.get_all()
        tools["Injected"] = DummyTool()
        assert not registry.has_tool("Injected")

    def test_get_all_definitions_shape(self, registry):
        defs = registry.get_all_definitions()
        assert len(defs) == 1
        d = defs[0]
        assert d["name"] == "Calculator"
        assert isinstance(d["description"], str)
        assert d["input_schema"]["type"] == "object"

    def test_definitions_filterable_by_allowed_tools(self, registry):
        """Simulates the allowed_tools filtering done in AgentLoop.run."""
        registry.register(DummyTool())
        defs = [
            d for d in registry.get_all_definitions() if d["name"] in ["Calculator"]
        ]
        assert [d["name"] for d in defs] == ["Calculator"]


# ---------------------------------------------------------------------------
# JSON Schema validation (validate_tool_inputs)
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_valid_arguments(self, registry):
        tool = registry.get("Calculator")
        result = validate_tool_inputs(
            {"operation": "add", "a": 1, "b": 2}, tool.input_schema
        )
        assert result.valid
        assert result.errors == []

    def test_missing_required_field(self, registry):
        tool = registry.get("Calculator")
        result = validate_tool_inputs({"a": 1, "b": 2}, tool.input_schema)
        assert not result.valid
        assert any("operation" in e for e in result.errors)

    def test_wrong_type(self, registry):
        tool = registry.get("Calculator")
        result = validate_tool_inputs(
            {"operation": "add", "a": "not-a-number", "b": 2}, tool.input_schema
        )
        assert not result.valid

    def test_enum_violation(self, registry):
        tool = registry.get("Calculator")
        result = validate_tool_inputs(
            {"operation": "power", "a": 1, "b": 2}, tool.input_schema
        )
        assert not result.valid

    def test_extra_properties_allowed(self, registry):
        # Schema does not set additionalProperties: false
        tool = registry.get("Calculator")
        result = validate_tool_inputs(
            {"operation": "add", "a": 1, "b": 2, "extra": "x"}, tool.input_schema
        )
        assert result.valid


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class TestCalculator:
    @pytest.fixture
    def calc(self):
        return Calculator()

    async def test_add(self, calc):
        r = await calc.execute({"operation": "add", "a": 2, "b": 3})
        assert r["result"] == 5

    async def test_subtract(self, calc):
        r = await calc.execute({"operation": "subtract", "a": 10, "b": 4})
        assert r["result"] == 6

    async def test_multiply(self, calc):
        r = await calc.execute({"operation": "multiply", "a": 7, "b": 6})
        assert r["result"] == 42

    async def test_divide(self, calc):
        r = await calc.execute({"operation": "divide", "a": 10, "b": 4})
        assert r["result"] == 2.5

    async def test_divide_by_zero(self, calc):
        r = await calc.execute({"operation": "divide", "a": 1, "b": 0})
        assert "error" in r
        assert "zero" in r["error"].lower()

    async def test_factorial(self, calc):
        r = await calc.execute({"operation": "factorial", "a": 5})
        assert r["result"] == 120

    async def test_factorial_negative_returns_error(self, calc):
        r = await calc.execute({"operation": "factorial", "a": -1})
        assert "error" in r

    async def test_missing_b_operand(self, calc):
        r = await calc.execute({"operation": "add", "a": 1})
        assert "error" in r

    async def test_invalid_operation_raises(self, calc):
        # NOTE: inconsistent with other errors which return {"error": ...}
        with pytest.raises(ValueError, match="Invalid operation"):
            await calc.execute({"operation": "bogus", "a": 1, "b": 2})


# ---------------------------------------------------------------------------
# WebSearch (mock)
# ---------------------------------------------------------------------------

class TestWebSearch:
    async def test_returns_results(self):
        r = await WebSearch().execute({"query": "test query", "num_results": 3})
        assert r["query"] == "test query"
        assert len(r["results"]) == 3
        assert r["total_results"] == 3
        assert all("test query" in res["title"] for res in r["results"])

    async def test_caps_at_ten_results(self):
        r = await WebSearch().execute({"query": "q", "num_results": 50})
        assert len(r["results"]) == 10

    async def test_default_num_results(self):
        r = await WebSearch().execute({"query": "q"})
        assert len(r["results"]) == 5


# ---------------------------------------------------------------------------
# FileReader
# ---------------------------------------------------------------------------

class TestFileReader:
    """FileReader is jailed to a workspace root; all tests run inside tmp_path."""

    @pytest.fixture
    def reader(self, tmp_path):
        return FileReader(workspace_root=str(tmp_path))

    @pytest.fixture
    def sample_file(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("hello world", encoding="utf-8")
        return f

    async def test_reads_file(self, reader, sample_file):
        r = await reader.execute({"file_path": "sample.txt"})
        assert r["content"] == "hello world"
        assert r["size"] == len("hello world")

    async def test_reads_file_via_absolute_path_inside_root(self, reader, sample_file):
        r = await reader.execute({"file_path": str(sample_file)})
        assert r["content"] == "hello world"

    async def test_file_not_found(self, reader):
        r = await reader.execute({"file_path": "nope.txt"})
        assert "error" in r
        assert "not found" in r["error"].lower()

    async def test_directory_path(self, reader, tmp_path):
        (tmp_path / "subdir").mkdir()
        r = await reader.execute({"file_path": "subdir"})
        assert "error" in r
        assert "not a file" in r["error"].lower()

    async def test_truncates_large_files(self, reader, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 20000, encoding="utf-8")
        r = await reader.execute({"file_path": "big.txt"})
        assert "truncated" in r["content"]
        assert len(r["content"]) < 20000

    async def test_bad_encoding(self, reader, tmp_path):
        f = tmp_path / "bin.txt"
        f.write_bytes(b"\xff\xfe\x00\x01\x80")
        r = await reader.execute({"file_path": "bin.txt", "encoding": "utf-8"})
        assert "error" in r

    async def test_path_traversal_blocked(self, reader, tmp_path):
        # A file outside the workspace must not be readable.
        outside = tmp_path.parent / "outside_secret.txt"
        outside.write_text("SECRET", encoding="utf-8")
        try:
            r = await reader.execute({"file_path": "../outside_secret.txt"})
            assert "error" in r
            assert "denied" in r["error"].lower() or "outside" in r["error"].lower()
        finally:
            outside.unlink()


# ---------------------------------------------------------------------------
# PythonExec
# ---------------------------------------------------------------------------

class TestPythonExec:
    @pytest.fixture
    def pyexec(self, monkeypatch):
        # PythonExec is disabled by default; enable it for these tests.
        monkeypatch.setenv("AGENTLOOP_ENABLE_PYTHON_EXEC", "1")
        return PythonExec()

    async def test_simple_expression(self, pyexec):
        r = await pyexec.execute({"code": "2 + 2"})
        assert r["success"] is True
        assert r["output"] == "4"

    async def test_math_module(self, pyexec):
        r = await pyexec.execute({"code": "math.sqrt(16)"})
        assert r["success"] is True
        assert r["output"] == "4.0"

    async def test_print_statement(self, pyexec):
        r = await pyexec.execute({"code": 'print("hello")'})
        assert r["success"] is True
        assert "hello" in r["output"]

    async def test_import_blocked(self, pyexec):
        r = await pyexec.execute({"code": "import os"})
        assert r["success"] is False

    async def test_eval_blocked(self, pyexec):
        r = await pyexec.execute({"code": "eval('1+1')"})
        assert r["success"] is False

    async def test_statement_rejected(self, pyexec):
        r = await pyexec.execute({"code": "x = 5"})
        assert r["success"] is False

    async def test_runtime_error_structured(self, pyexec):
        r = await pyexec.execute({"code": "1 / 0"})
        assert r["success"] is False
        assert r["error"] == "Execution failed"

    async def test_execution_timeout(self, pyexec, monkeypatch):
        # Patch the evaluator to sleep; avoids a real slow computation.
        monkeypatch.setattr(
            pyexec, "_safe_eval", lambda code, local_vars: time.sleep(5)
        )
        r = await pyexec.execute({"code": "1 + 1", "timeout": 1})
        assert r["success"] is False
        assert "timed out" in r["details"].lower()

    async def test_dunder_escape_blocked(self, pyexec):
        r = await pyexec.execute({"code": "().__class__.__base__.__subclasses__()"})
        assert r["success"] is False

    async def test_cos_function_allowed(self, pyexec):
        r = await pyexec.execute({"code": "cos(0)"})
        assert r["success"] is True
        assert r["output"] == "1.0"
