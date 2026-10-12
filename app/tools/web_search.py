"""Provider-backed web search with bounded, validated result metadata."""
import asyncio
import ipaddress
import os
import time
from typing import Any, Dict, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

from app.tools.base import BaseTool


class SearchResult(BaseModel):
    title: str = Field(max_length=300)
    url: str
    snippet: str = Field(max_length=1000)
    source: str = Field(max_length=100)


class SearchProvider(Protocol):
    async def search(self, query: str, limit: int) -> list[SearchResult]: ...


def valid_result_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            return False
        hostname = parts.hostname.lower().rstrip(".")
        if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
            return False
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return "." in hostname
        return address.is_global
    except ValueError:
        return False


class SearxngProvider:
    def __init__(self, base_url: str, timeout: float = 8.0, transport: httpx.AsyncBaseTransport | None = None):
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo:
            raise ValueError("Invalid search service URL")
        if url.scheme != "https" and url.host not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Search service must use HTTPS outside localhost")
        self.base_url = str(url).rstrip("/")
        self.timeout = timeout
        self.transport = transport

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False, transport=self.transport) as client:
            response = await client.get(f"{self.base_url}/search", params={"q": query, "format": "json"})
            if 300 <= response.status_code < 400:
                raise ValueError("Search service redirects are not allowed")
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ValueError("Invalid search service response")
        results = []
        for item in data["results"][:min(limit * 3, 30)]:
            if not isinstance(item, dict) or not valid_result_url(item.get("url", "")):
                continue
            results.append(SearchResult(
                title=str(item.get("title", ""))[:300],
                url=item["url"],
                snippet=str(item.get("content", ""))[:1000],
                source=str(item.get("engine", "searxng"))[:100],
            ))
            if len(results) == limit:
                break
        return results


class WebSearch(BaseTool):
    def __init__(self, provider: SearchProvider | None = None, rate_per_second: float = 1.0):
        provider_name = os.getenv("SEARCH_PROVIDER", "").lower()
        if provider is None and provider_name == "searxng":
            endpoint = os.getenv("SEARCH_BASE_URL")
            if endpoint:
                provider = SearxngProvider(endpoint)
        if provider_name and provider_name != "searxng" and provider is None:
            raise ValueError(f"Unsupported search provider: {provider_name}")
        if rate_per_second <= 0:
            raise ValueError("Search rate must be positive")
        self.provider = provider
        self._interval = 1.0 / rate_per_second
        self._next_request = 0.0
        self._rate_lock = asyncio.Lock()
        super().__init__()

    @property
    def risk_level(self) -> str:
        return "medium"

    @property
    def timeout(self) -> int:
        return 12

    def get_description(self) -> str:
        return "Search the web using the configured search service; unavailable without SEARCH_PROVIDER and SEARCH_BASE_URL"

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "num_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Dict[str, Any]) -> Any:
        if self.provider is None:
            return {"error": "Web search is not configured", "results": []}
        query = arguments["query"]
        limit = arguments.get("num_results", 5)
        async with self._rate_lock:
            now = time.monotonic()
            if self._next_request > now:
                await asyncio.sleep(self._next_request - now)
            self._next_request = time.monotonic() + self._interval
        try:
            results = await self.provider.search(query, limit)
        except (httpx.HTTPError, ValueError) as exc:
            return {"error": f"Search service failure: {type(exc).__name__}", "results": []}
        validated = [item.model_dump() for item in results if valid_result_url(item.url)][:limit]
        return {"query": query, "results": validated, "total_results": len(validated)}
