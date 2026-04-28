"""
Lightweight memory interface for agent state persistence.
"""
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime


class Memory:
    """
    Lightweight in-memory storage for agent context.
    Can be extended to use persistent storage (Redis, database, etc.).
    """
    
    def __init__(self):
        self._storage: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
    
    async def store(self, key: str, value: Any) -> None:
        """
        Store a value in memory.
        
        Args:
            key: Storage key
            value: Value to store
        """
        self._storage[key] = value
        self._timestamps[key] = datetime.utcnow()
    
    async def retrieve(self, query: str) -> Optional[Any]:
        """
        Retrieve a value from memory by key.
        
        Args:
            query: Key to retrieve
            
        Returns:
            Stored value if found, None otherwise
        """
        return self._storage.get(query)
    
    async def search(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Search for keys matching a pattern.
        
        Args:
            pattern: Pattern to match (simple substring match)
            
        Returns:
            List of matching entries with keys and values
        """
        results = []
        for key, value in self._storage.items():
            if pattern.lower() in key.lower():
                results.append({
                    "key": key,
                    "value": value,
                    "timestamp": self._timestamps.get(key).isoformat()
                })
        return results
    
    async def delete(self, key: str) -> bool:
        """
        Delete a key from memory.
        
        Args:
            key: Key to delete
            
        Returns:
            True if deleted, False if not found
        """
        if key in self._storage:
            del self._storage[key]
            del self._timestamps[key]
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all stored values."""
        self._storage.clear()
        self._timestamps.clear()
    
    async def get_all(self) -> Dict[str, Any]:
        """
        Get all stored values.
        
        Returns:
            Dictionary of all stored key-value pairs
        """
        return self._storage.copy()
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory stats
        """
        return {
            "total_keys": len(self._storage),
            "oldest_timestamp": min(self._timestamps.values()) if self._timestamps else None,
            "newest_timestamp": max(self._timestamps.values()) if self._timestamps else None
        }
