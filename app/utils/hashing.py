"""
Hashing utilities for loop detection.
"""
import hashlib
import json
from typing import Dict, Any


def create_fingerprint(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Create a fingerprint for a tool call to detect loops.
    
    Args:
        tool_name: Name of the tool
        arguments: Arguments passed to the tool
        
    Returns:
        Hash string representing the fingerprint
    """
    # Sort arguments to ensure consistent fingerprint regardless of order
    sorted_args = json.dumps(arguments, sort_keys=True)
    
    # Create fingerprint from tool name + sorted arguments
    fingerprint_string = f"{tool_name}:{sorted_args}"
    
    # Use SHA-256 for fingerprint
    return hashlib.sha256(fingerprint_string.encode()).hexdigest()
