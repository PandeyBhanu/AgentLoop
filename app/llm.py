"""
Provider adapters for OpenAI, Groq, and Gemini with normalized agent responses.
"""
from typing import List, Dict, Any, Optional, Literal, Protocol
import httpx
import os
import time
import json
from app.schemas.messages import Message


COST_PER_1K_INPUT_OPENAI = 0.03
COST_PER_1K_OUTPUT_OPENAI = 0.06
COST_PER_1K_INPUT_GROQ = 0.00059
COST_PER_1K_OUTPUT_GROQ = 0.00079
COST_PER_1K_INPUT_GEMINI = 0.000125
COST_PER_1K_OUTPUT_GEMINI = 0.000375


class LLMConfig:
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
        if provider not in ("openai", "groq", "gemini"):
            raise ValueError(f"Unsupported LLM provider: {provider!r}")
        self.provider = provider

        if provider == "groq":
            self.base_url = base_url or "https://api.groq.com/openai/v1"
            self.model = model or "llama-3.1-8b-instant"
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
        elif provider == "gemini":
            self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta"
            self.model = model or "gemini-pro"
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        else:
            self.base_url = base_url or "https://api.openai.com/v1"
            self.model = model or "gpt-4"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.temperature = temperature
        self.max_tokens = max_tokens


def extract_json_block(text: str) -> str:
    if not text:
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]
        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass

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

    return text


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMHTTPError(Exception):
    def __init__(
        self,
        status_code: int | None = None,
        retry_after: float | None = None,
        transient: bool | None = None,
    ):
        self.status_code = status_code
        self.retry_after = retry_after
        self.transient = (
            status_code in (408, 409, 425, 429) or
            (status_code is not None and status_code >= 500)
        ) if transient is None else transient
        super().__init__(
            f"LLM API error: HTTP {status_code}"
            if status_code is not None else "LLM API transport error"
        )


class LLMProvider(Protocol):
    @property
    def capabilities(self) -> Dict[str, bool]: ...

    async def send_messages(
        self, messages: List[Message], tool_schemas: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = True, max_retries: int = 2
    ) -> tuple[str, Optional[Dict[str, Any]]]: ...

    async def close(self) -> None: ...


def _raise_http_error(exc: httpx.HTTPError) -> None:
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("retry-after")
        try:
            parsed_retry_after = float(retry_after) if retry_after else None
        except ValueError:
            parsed_retry_after = None
        raise LLMHTTPError(exc.response.status_code, parsed_retry_after) from exc
    raise LLMHTTPError(transient=True) from exc


def _usage_metadata(
    provider: str,
    model: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    estimated_cost: float,
) -> Dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
        "estimated_cost": estimated_cost,
        "provider": provider,
        "model": model,
    }


class OpenAICompatibleProvider:
    native_tool_calls = False
    enforced_json_mode = False
    input_cost = COST_PER_1K_INPUT_OPENAI
    output_cost = COST_PER_1K_OUTPUT_OPENAI

    def __init__(self, config: LLMConfig, client: httpx.AsyncClient):
        self.config = config
        self.client = client

    @property
    def capabilities(self) -> Dict[str, bool]:
        return {
            "native_tool_calls": self.native_tool_calls,
            "json_mode": self.enforced_json_mode,
        }

    async def send_messages(
        self,
        messages: List[Message],
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = True,
        max_retries: int = 2,
    ) -> tuple[str, Dict[str, Any]]:
        start_time = time.time()
        api_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        payload = {
            "model": self.config.model,
            "messages": api_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        if tool_schemas and self.native_tool_calls:
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

        if json_mode and self.enforced_json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = await self.client.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            _raise_http_error(e)

        latency_ms = (time.time() - start_time) * 1000
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError("Invalid LLM response format") from e

        native_calls = message.get("tool_calls") or []
        if native_calls:
            if len(native_calls) != 1 or native_calls[0].get("type") != "function":
                raise ValueError("Only one native function call per model step is supported")
            function = native_calls[0]["function"]
            arguments = json.loads(function["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("Native function arguments must be an object")
            content = json.dumps({
                "type": "action",
                "tool_name": function["name"],
                "arguments": arguments,
            })
        else:
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Provider returned neither text nor a supported function call")

        if json_mode and not self.enforced_json_mode:
            content = extract_json_block(content)

        token_usage = data.get("usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        if total_tokens == 0:
            input_text = " ".join(str(msg.content) for msg in messages)
            prompt_tokens = estimate_tokens(input_text)
            completion_tokens = estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens

        estimated_cost = (
            (prompt_tokens / 1000) * self.input_cost +
            (completion_tokens / 1000) * self.output_cost
        )
        return content, _usage_metadata(
            self.config.provider, self.config.model, latency_ms,
            prompt_tokens, completion_tokens, total_tokens, estimated_cost,
        )


class OpenAIProvider(OpenAICompatibleProvider):
    native_tool_calls = True
    enforced_json_mode = True


class GroqProvider(OpenAICompatibleProvider):
    input_cost = COST_PER_1K_INPUT_GROQ
    output_cost = COST_PER_1K_OUTPUT_GROQ


class GeminiProvider:
    def __init__(self, config: LLMConfig, client: httpx.AsyncClient):
        self.config = config
        self.client = client

    @property
    def capabilities(self) -> Dict[str, bool]:
        return {"native_tool_calls": False, "json_mode": False}

    async def send_messages(
        self,
        messages: List[Message],
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = True,
        max_retries: int = 2,
    ) -> tuple[str, Dict[str, Any]]:
        start_time = time.time()
        contents = []
        system_texts = []
        for msg in messages:
            if msg.role == "system":
                system_texts.append(str(msg.content))
                continue
            contents.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [{"text": str(msg.content)}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens
            }
        }
        if system_texts:
            payload["system_instruction"] = {
                "parts": [{"text": "\n\n".join(system_texts)}]
            }

        try:
            response = await self.client.post(
                f"{self.config.base_url}/models/{self.config.model}:generateContent",
                json=payload,
                headers={
                    "x-goog-api-key": self.config.api_key,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            _raise_http_error(e)

        latency_ms = (time.time() - start_time) * 1000
        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError("Invalid LLM response format") from e
        if not isinstance(content, str):
            raise ValueError("Provider returned non-text content")

        usage_metadata = data.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", prompt_tokens + completion_tokens)
        if total_tokens == 0:
            input_text = " ".join(str(msg.content) for msg in messages)
            prompt_tokens = estimate_tokens(input_text)
            completion_tokens = estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens

        estimated_cost = (
            (prompt_tokens / 1000) * COST_PER_1K_INPUT_GEMINI +
            (completion_tokens / 1000) * COST_PER_1K_OUTPUT_GEMINI
        )
        return content, _usage_metadata(
            "gemini", self.config.model, latency_ms,
            prompt_tokens, completion_tokens, total_tokens, estimated_cost,
        )


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=60.0)
        adapter = {
            "openai": OpenAIProvider,
            "groq": GroqProvider,
            "gemini": GeminiProvider,
        }[config.provider]
        self.provider = adapter(config, self.client)

    @property
    def capabilities(self) -> Dict[str, bool]:
        return self.provider.capabilities

    async def send_messages(
        self,
        messages: List[Message],
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = True,
        max_retries: int = 2
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        if not self.config.api_key:
            raise Exception(
                f"No API key configured for provider '{self.config.provider}'"
            )
        return await self.provider.send_messages(
            messages, tool_schemas=tool_schemas,
            json_mode=json_mode, max_retries=max_retries,
        )

    async def close(self) -> None:
        await self.client.aclose()
