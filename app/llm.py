"""
Provider-agnostic LLM wrapper supporting OpenAI, Groq, and Gemini APIs with token and cost tracking.
"""
from typing import List, Dict, Any, Optional, Literal
import httpx
import os
import time
import json
from app.schemas.messages import Message


# Cost estimation per 1K tokens
COST_PER_1K_INPUT_OPENAI = 0.03
COST_PER_1K_OUTPUT_OPENAI = 0.06
COST_PER_1K_INPUT_GROQ = 0.00059  # Groq Llama3 pricing
COST_PER_1K_OUTPUT_GROQ = 0.00079
COST_PER_1K_INPUT_GEMINI = 0.000125  # Gemini 1.5 Flash pricing
COST_PER_1K_OUTPUT_GEMINI = 0.000375


class LLMConfig:
    """
    Configuration for LLM provider selection and settings.
    
    Design decision: Centralize all provider-specific configuration in this class
    to avoid scattering provider logic across the codebase.
    """
    provider: Literal["openai", "groq", "gemini"]
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    
    def __init__(
        self,
        provider: Literal["openai", "groq", "gemini"] = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.provider = provider
        
        # Provider-specific defaults
        if provider == "groq":
            self.base_url = base_url or "https://api.groq.com/openai/v1"
            self.model = model or "llama-3.1-8b-instant"
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
        elif provider == "gemini":
            self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta"
            self.model = model or "gemini-pro"
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        else:  # openai
            self.base_url = base_url or "https://api.openai.com/v1"
            self.model = model or "gpt-4"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        self.temperature = temperature
        self.max_tokens = max_tokens


def extract_json_block(text: str) -> str:
    """
    Extract JSON block from text, handling extra text before/after JSON.
    
    This is useful for providers like Groq that may return extra text
    around JSON responses.
    """
    if not text:
        return text
    
    # Try to find JSON-like content
    # First, try to find content between { and }
    start = text.find("{")
    end = text.rfind("}")
    
    if start != -1 and end != -1 and end > start:
        json_text = text[start:end+1]
        # Try to parse it to see if it's valid
        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass
    
    # If that fails, try to find content between ```json and ```
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            json_text = text[start:end].strip()
            try:
                json.loads(json_text)
                return json_text
            except json.JSONDecodeError:
                pass
    
    # If that fails, try to find content between ``` and ```
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            json_text = text[start:end].strip()
            try:
                json.loads(json_text)
                return json_text
            except json.JSONDecodeError:
                pass
    
    # Return original text if no valid JSON found
    return text


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using simple heuristic (len(text)/4).
    
    Used for providers like Groq that may not consistently return
    token usage in their API responses.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    return max(1, len(text) // 4)


class LLMClient:
    """
    Provider-agnostic client for interacting with LLM APIs.
    
    Supports OpenAI and Groq through a unified interface.
    All provider-specific logic is isolated in this layer.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client with configuration.
        
        Args:
            config: LLMConfig object with provider settings
        """
        self.config = config
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def _send_openai_compatible(
        self,
        messages: List[Message],
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = True
    ) -> tuple[str, Dict[str, Any]]:
        """Send request to OpenAI-compatible API (OpenAI, Groq)."""
        start_time = time.time()
        
        # Convert messages to API format
        api_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        # Build request payload
        payload = {
            "model": self.config.model,
            "messages": api_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        # Add tool schemas if provided (Groq may not support function calling)
        if tool_schemas and self.config.provider == "openai":
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema["description"],
                        "parameters": schema["input_schema"]
                    }
                }
                for schema in tool_schemas
            ]
        
        # Enforce JSON mode if requested (note: Groq may not support response_format)
        if json_mode and self.config.provider == "openai":
            payload["response_format"] = {"type": "json_object"}
        
        # Make API request
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Debug: print request payload for Groq
        if self.config.provider == "groq":
            print(f"Groq request payload: {payload}")
            print(f"Groq base_url: {self.config.base_url}")
        
        try:
            response = await self.client.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            
            # Debug: print response for Groq
            if self.config.provider == "groq":
                print(f"Groq response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"Groq response body: {response.text}")
            
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            raise Exception(f"LLM API error: {str(e)}")
        except KeyError as e:
            raise Exception(f"Invalid LLM response format: {str(e)}")
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response text
        content = data["choices"][0]["message"]["content"]
        
        # For Groq, extract JSON block if in JSON mode
        if json_mode and self.config.provider == "groq":
            content = extract_json_block(content)
        
        # Extract token usage and calculate cost
        token_usage = data.get("usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        
        # Estimate tokens if provider doesn't return usage (Groq inconsistency)
        if total_tokens == 0:
            input_text = " ".join([msg.content for msg in messages])
            prompt_tokens = estimate_tokens(input_text)
            completion_tokens = estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens
        
        # Calculate cost based on provider
        if self.config.provider == "groq":
            estimated_cost = (
                (prompt_tokens / 1000) * COST_PER_1K_INPUT_GROQ +
                (completion_tokens / 1000) * COST_PER_1K_OUTPUT_GROQ
            )
        else:
            estimated_cost = (
                (prompt_tokens / 1000) * COST_PER_1K_INPUT_OPENAI +
                (completion_tokens / 1000) * COST_PER_1K_OUTPUT_OPENAI
            )
        
        metadata = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "estimated_cost": estimated_cost,
            "provider": self.config.provider,
            "model": self.config.model
        }
        
        return content, metadata
    
    async def _send_gemini(
        self,
        messages: List[Message],
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = True
    ) -> tuple[str, Dict[str, Any]]:
        """Send request to Gemini API."""
        start_time = time.time()
        
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })
        
        # Build request payload
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens
            }
        }
        
        # Make API request
        url = f"{self.config.base_url}/models/{self.config.model}:generateContent?key={self.config.api_key}"
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            raise Exception(f"LLM API error: {str(e)}")
        except KeyError as e:
            raise Exception(f"Invalid LLM response format: {str(e)}")
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response text
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Extract token usage
        usage_metadata = data.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", prompt_tokens + completion_tokens)
        
        # Estimate tokens if not provided
        if total_tokens == 0:
            input_text = " ".join([msg.content for msg in messages])
            prompt_tokens = estimate_tokens(input_text)
            completion_tokens = estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens
        
        # Calculate cost
        estimated_cost = (
            (prompt_tokens / 1000) * COST_PER_1K_INPUT_GEMINI +
            (completion_tokens / 1000) * COST_PER_1K_OUTPUT_GEMINI
        )
        
        metadata = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "estimated_cost": estimated_cost,
            "provider": self.config.provider,
            "model": self.config.model
        }
        
        return content, metadata
    
    async def send_messages(
        self,
        messages: List[Message],
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = True,
        max_retries: int = 2
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Send messages to the LLM and get response with latency and cost tracking.
        
        Args:
            messages: List of Message objects
            tool_schemas: Optional list of tool schemas to include
            json_mode: Whether to enforce JSON output
            max_retries: Maximum retry attempts for parsing failures
            
        Returns:
            Tuple of (response_text, metadata) where metadata includes:
            - prompt_tokens: Input tokens
            - completion_tokens: Output tokens
            - total_tokens: Total tokens
            - latency_ms: Request latency in milliseconds
            - estimated_cost: Estimated cost in USD
            - provider: Provider name
            - model: Model name
        """
        if self.config.provider == "gemini":
            return await self._send_gemini(messages, tool_schemas, json_mode)
        else:
            return await self._send_openai_compatible(messages, tool_schemas, json_mode)
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
