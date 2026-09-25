"""
File reader tool for reading file contents.

Security model: reads are jailed to a workspace root (default: process cwd,
overridable via constructor arg or FILE_READER_ROOT env var). The requested
path is resolved (symlinks included) and rejected unless it stays inside the
workspace. Hidden files/dirs (any path component starting with '.', e.g.
.env, .git/) and known binary extensions are rejected. Error messages are
generic and never echo resolved filesystem paths.
"""
from typing import Any, Dict, Optional
from app.tools.base import BaseTool
import os
import aiofiles
from pathlib import Path


MAX_FILE_BYTES = 1_000_000      # hard cap on file size
MAX_OUTPUT_CHARS = 10_000       # cap on returned content length

# Extensions that are never served (binary/executable content is not useful
# to the agent and may carry secrets, e.g. .sqlite/.db keystores).
BLOCKED_EXTENSIONS = {
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
    ".pyc", ".pyo", ".class", ".jar", ".o",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".pdf", ".woff", ".woff2", ".ttf", ".otf",
    ".sqlite", ".sqlite3", ".db", ".pem", ".key", ".pfx", ".p12",
}


class FileReader(BaseTool):
    """
    File reader tool for reading text files, jailed to a workspace root.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        root = workspace_root or os.getenv("FILE_READER_ROOT") or os.getcwd()
        self._root = Path(root).resolve()
        super().__init__()

    def get_description(self) -> str:
        return "Read the contents of a text file from the workspace"

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read, relative to the workspace root"
                },
                "encoding": {
                    "type": "string",
                    "default": "utf-8",
                    "description": "File encoding (default: utf-8)"
                }
            },
            "required": ["file_path"]
        }

    def _resolve(self, file_path: str) -> tuple[Optional[Path], Optional[str]]:
        """
        Resolve the requested path inside the workspace root.

        Returns (resolved_path, None) on success or (None, error_message).
        """
        if not file_path or not file_path.strip():
            return None, "file_path is required"

        try:
            requested = Path(file_path)
            # For absolute inputs, resolve as-is; relative inputs are anchored
            # at the workspace root. resolve() also collapses '..' and follows
            # symlinks, so the containment check below is symlink-safe.
            candidate = (
                requested.resolve() if requested.is_absolute()
                else (self._root / requested).resolve()
            )
        except (OSError, ValueError):
            return None, "Invalid file path"

        # Containment: resolved path must stay inside the workspace root.
        try:
            relative = candidate.relative_to(self._root)
        except ValueError:
            return None, "Access denied: path is outside the workspace"

        # Block hidden files/dirs anywhere in the relative path
        # (.env, .git/, .ssh/, ...).
        if any(part.startswith(".") for part in relative.parts):
            return None, "Access denied: hidden files are not readable"

        if candidate.suffix.lower() in BLOCKED_EXTENSIONS:
            return None, "Access denied: unsupported file type"

        return candidate, None

    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        Read the contents of a file inside the workspace.
        """
        file_path = arguments.get("file_path", "")
        encoding = arguments.get("encoding", "utf-8")

        candidate, error = self._resolve(file_path)
        if error:
            return {"error": error}

        if not candidate.exists():
            return {"error": "File not found"}

        if not candidate.is_file():
            return {"error": "Path is not a file"}

        # Hard size limit before reading to avoid loading huge files.
        try:
            if candidate.stat().st_size > MAX_FILE_BYTES:
                return {"error": "File too large to read"}
        except OSError:
            return {"error": "Could not stat file"}

        try:
            async with aiofiles.open(candidate, 'r', encoding=encoding) as f:
                content = await f.read()

            if len(content) > MAX_OUTPUT_CHARS:
                content = content[:MAX_OUTPUT_CHARS] + "\n\n... (content truncated)"

            return {
                "file_path": file_path,
                "content": content,
                "size": len(content)
            }
        except PermissionError:
            return {"error": "Permission denied"}
        except UnicodeDecodeError:
            return {"error": f"Could not decode file with encoding {encoding}"}
        except (OSError, ValueError):
            return {"error": "Could not read file"}
