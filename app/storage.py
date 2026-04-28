"""
Storage module for agent run states.
In production, this should use a proper database.
"""
from typing import Dict

# Global state storage for agent runs
runs: Dict[str, Dict] = {}
