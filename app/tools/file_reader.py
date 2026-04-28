"""
File reader tool for reading file contents.
"""
from typing import Any, Dict
from app.tools.base import BaseTool
import os
import aiofiles


class FileReader(BaseTool):
    """
    File reader tool for reading text files.
    """
    
    def get_description(self) -> str:
        return "Read the contents of a text file from the local filesystem"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "encoding": {
                    "type": "string",
                    "default": "utf-8",
                    "description": "File encoding (default: utf-8)"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        Read the contents of a file.
        
        Args:
            arguments: Dictionary with 'file_path' and optional 'encoding' keys
            
        Returns:
            File contents or error message
        """
        file_path = arguments.get("file_path", "")
        encoding = arguments.get("encoding", "utf-8")
        
        # Security check: prevent reading files outside current directory
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return {"error": f"File not found: {file_path}"}
        
        if not os.path.isfile(abs_path):
            return {"error": f"Path is not a file: {file_path}"}
        
        try:
            async with aiofiles.open(abs_path, 'r', encoding=encoding) as f:
                content = await f.read()
            
            # Truncate very large files
            max_length = 10000
            if len(content) > max_length:
                content = content[:max_length] + "\n\n... (content truncated)"
            
            return {
                "file_path": file_path,
                "content": content,
                "size": len(content)
            }
        except PermissionError:
            return {"error": f"Permission denied: {file_path}"}
        except UnicodeDecodeError:
            return {"error": f"Could not decode file with encoding {encoding}"}
        except Exception as e:
            return {"error": str(e)}
