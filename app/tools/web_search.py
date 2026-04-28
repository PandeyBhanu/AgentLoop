"""
Web search tool (mock implementation).
"""
from typing import Any, Dict
from app.tools.base import BaseTool
import random


class WebSearch(BaseTool):
    """
    Web search tool for searching the internet (mock implementation).
    """
    
    def get_description(self) -> str:
        return "Search the web for information (mock implementation - returns simulated results)"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of results to return"
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        Execute a web search (mock).
        
        Args:
            arguments: Dictionary with 'query' and optional 'num_results' keys
            
        Returns:
            Mock search results
        """
        query = arguments.get("query", "")
        num_results = arguments.get("num_results", 5)
        
        # Mock search results
        mock_results = []
        for i in range(min(num_results, 10)):
            mock_results.append({
                "title": f"Mock result {i+1} for: {query}",
                "url": f"https://example.com/result/{i+1}",
                "snippet": f"This is a mock search result snippet for the query '{query}'. "
                          f"In a real implementation, this would contain actual search results."
            })
        
        return {
            "query": query,
            "results": mock_results,
            "total_results": len(mock_results)
        }
