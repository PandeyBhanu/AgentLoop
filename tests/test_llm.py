"""
Tests for app.llm — LLMClient provider dispatch, payload construction,
token/cost metadata, and error handling. httpx is mocked; no network.
"""
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.llm import LLMClient, LLMConfig, estimate_tokens, extract_json_block
from app.schemas.messages import Message


OPENAI_PAYLOAD = {
    "choices": [{"message": {"content": '{"type": "finish", "final_answer": "x"}'}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

GEMINI_PAYLOAD = {
    "candidates": [{"content": {"parts": [{"text": '{"type": "finish", "final_answer": "g"}'}]}}],
    "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4, "totalTokenCount": 12},
}

MSGS = [
    Message(role="system", content="sys"),
    Message(role="user", content="hi"),
]

TOOL_SCHEMAS = [
    {"name": "Calculator", "description": "math", "input_schema": {"type": "object"}}
]


def http_response(status=200, payload=None, url="http://api.test/chat"):
    req = httpx.Request("POST", url)
    return httpx.Response(status, json=payload or {}, request=req)


def make_client(provider, **kwargs):
    config = LLMConfig(provider=provider, api_key="sk-test", **kwargs)
    return LLMClient(config)


def patch_post(client, monkeypatch, response):
    mock = AsyncMock(return_value=response)
    monkeypatch.setattr(client.client, "post", mock)
    return mock


# ---------------------------------------------------------------------------
# OpenAI-compatible path
# ---------------------------------------------------------------------------

class TestOpenAIPath:
    async def test_sends_tools_and_json_mode(self, monkeypatch):
        client = make_client("openai")
        mock = patch_post(client, monkeypatch, http_response(payload=OPENAI_PAYLOAD))

        content, meta = await client.send_messages(MSGS, TOOL_SCHEMAS)

        payload = mock.call_args.kwargs["json"]
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["tools"][0]["function"]["name"] == "Calculator"
        assert payload["tools"][0]["function"]["parameters"] == {"type": "object"}
        assert content == OPENAI_PAYLOAD["choices"][0]["message"]["content"]
        assert meta["provider"] == "openai"
        assert meta["total_tokens"] == 15

    async def test_auth_header_sent(self, monkeypatch):
        client = make_client("openai")
        mock = patch_post(client, monkeypatch, http_response(payload=OPENAI_PAYLOAD))
        await client.send_messages(MSGS)
        assert mock.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"

    async def test_token_estimation_fallback(self, monkeypatch):
        payload = {
            "choices": [{"message": {"content": '{"type": "finish", "final_answer": "x"}'}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        client = make_client("openai")
        patch_post(client, monkeypatch, http_response(payload=payload))

        _, meta = await client.send_messages(MSGS)
        assert meta["total_tokens"] > 0  # estimated via len/4 heuristic

    async def test_http_error_wrapped(self, monkeypatch):
        client = make_client("openai")
        patch_post(client, monkeypatch, http_response(status=500, payload={}))
        with pytest.raises(Exception, match="LLM API error"):
            await client.send_messages(MSGS)


# ---------------------------------------------------------------------------
# Groq path
# ---------------------------------------------------------------------------

class TestGroqPath:
    async def test_omits_tools_and_json_mode(self, monkeypatch):
        client = make_client("groq")
        mock = patch_post(client, monkeypatch, http_response(payload=OPENAI_PAYLOAD))

        await client.send_messages(MSGS, TOOL_SCHEMAS, json_mode=True)
        payload = mock.call_args.kwargs["json"]
        assert "tools" not in payload
        assert "response_format" not in payload
        assert mock.call_args.args[0] == "https://api.groq.com/openai/v1/chat/completions"

    async def test_json_block_extraction(self, monkeypatch):
        payload = {
            "choices": [{
                "message": {"content":
                    'Here you go:\n{"type": "finish", "final_answer": "x"}\nDone.'}
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        client = make_client("groq")
        patch_post(client, monkeypatch, http_response(payload=payload))

        content, meta = await client.send_messages(MSGS, json_mode=True)
        assert json.loads(content)["type"] == "finish"
        assert meta["provider"] == "groq"

    async def test_groq_cost_uses_groq_pricing(self, monkeypatch):
        client = make_client("groq")
        patch_post(client, monkeypatch, http_response(payload=OPENAI_PAYLOAD))
        _, meta = await client.send_messages(MSGS)
        # 10/1000*0.00059 + 5/1000*0.00079
        assert meta["estimated_cost"] == pytest.approx(0.0000059 + 0.00000395)


# ---------------------------------------------------------------------------
# Gemini path
# ---------------------------------------------------------------------------

class TestGeminiPath:
    async def test_gemini_payload_shape(self, monkeypatch):
        client = make_client("gemini")
        mock = patch_post(client, monkeypatch, http_response(payload=GEMINI_PAYLOAD))

        content, meta = await client.send_messages(MSGS)

        payload = mock.call_args.kwargs["json"]
        # implementation maps every non-"user" role (incl. system) to "model"
        assert payload["contents"][0]["role"] == "model"
        assert payload["contents"][0]["parts"][0]["text"] == "sys"
        assert payload["contents"][1]["role"] == "user"
        assert meta["provider"] == "gemini"
        assert meta["total_tokens"] == 12

    async def test_gemini_key_in_header_not_url(self, monkeypatch):
        # Regression: the API key must travel in the x-goog-api-key header,
        # never in the URL (URLs end up in logs and exception messages).
        client = make_client("gemini")
        mock = patch_post(client, monkeypatch, http_response(payload=GEMINI_PAYLOAD))
        await client.send_messages(MSGS)
        assert "sk-test" not in mock.call_args.args[0]
        assert mock.call_args.kwargs["headers"]["x-goog-api-key"] == "sk-test"

    async def test_assistant_maps_to_model(self, monkeypatch):
        client = make_client("gemini")
        mock = patch_post(client, monkeypatch, http_response(payload=GEMINI_PAYLOAD))
        msgs = MSGS + [Message(role="assistant", content="a")]
        await client.send_messages(msgs)
        assert mock.call_args.kwargs["json"]["contents"][2]["role"] == "model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_extract_json_block_plain(self):
        assert extract_json_block('{"a": 1}') == '{"a": 1}'

    def test_extract_json_block_with_text(self):
        out = extract_json_block('prefix {"a": 1} suffix')
        assert json.loads(out) == {"a": 1}

    def test_extract_json_block_fence(self):
        out = extract_json_block('```json\n{"a": 2}\n```')
        assert json.loads(out) == {"a": 2}

    def test_extract_json_block_no_json(self):
        assert extract_json_block("no json here") == "no json here"

    def test_estimate_tokens(self):
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("") == 1
        assert estimate_tokens("x" * 100) == 25


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_provider_defaults(self):
        assert LLMConfig(provider="groq", api_key="k").base_url == \
            "https://api.groq.com/openai/v1"
        assert LLMConfig(provider="openai", api_key="k").base_url == \
            "https://api.openai.com/v1"
        assert LLMConfig(provider="gemini", api_key="k").model == "gemini-pro"

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError):
            LLMConfig(provider="not-a-provider", api_key="k")

    async def test_missing_api_key_fails_fast(self, monkeypatch):
        # Ensure no real key leaks in from the environment.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient(LLMConfig(provider="openai", api_key=None))
        with pytest.raises(Exception, match="No API key"):
            await client.send_messages(MSGS)
