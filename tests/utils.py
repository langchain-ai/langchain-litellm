from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from litellm import Router
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from openai.types.chat import ChatCompletion
from openai.types.responses import Response


def serve_http(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
) -> list[httpx.Request]:
    """Answer every request litellm sends with ``body``, in-process.

    litellm sends async calls through its own aiohttp transport, not httpx's.
    """
    requests: list[httpx.Request] = []

    def _reply(_: object, request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=body, request=request)

    async def _areply(transport: object, request: httpx.Request) -> httpx.Response:
        return _reply(transport, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _reply)
    monkeypatch.setattr(LiteLLMAiohttpTransport, "handle_async_request", _areply)
    return requests


def responses_api_reply(*output: dict[str, Any]) -> dict[str, Any]:
    """A completed Responses API reply carrying ``output``, checked by the openai SDK."""
    reply = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": list(output),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
    }
    Response.model_validate(reply)
    return reply


def message_item(*texts: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": text, "annotations": []} for text in texts
        ],
    }


def function_call_item(call_id: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "get_weather",
        "arguments": arguments,
    }


def reasoning_item(item_id: str, summary: str) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": item_id,
        "summary": [{"type": "summary_text", "text": summary}],
    }


def chat_completion_reply(*contents: str) -> dict[str, Any]:
    """A Chat Completions reply with one choice per content, checked by the openai SDK."""
    reply = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": index,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
            for index, content in enumerate(contents)
        ],
    }
    ChatCompletion.model_validate(reply)
    return reply


def make_router() -> Router:
    model_group_gpt4 = "gpt-4"
    model_group_to_test = "gpt-3.5-turbo"
    fake_model_prefix = "azure/fake-deployment-name-"
    fake_models_names = [fake_model_prefix + suffix for suffix in ["1", "2"]]
    fake_api_key = "fakekeyvalue"
    fake_api_version = "XXXX-XX-XX"
    fake_api_base = "https://faketesturl/"

    model_list = [
        {
            "model_name": model_group_gpt4,
            "litellm_params": {
                "model": fake_models_names[0],
                "api_key": fake_api_key,
                "api_version": fake_api_version,
                "api_base": fake_api_base,
            },
        },
        {
            "model_name": model_group_to_test,
            "litellm_params": {
                "model": fake_models_names[1],
                "api_key": fake_api_key,
                "api_version": fake_api_version,
                "api_base": fake_api_base,
            },
        },
    ]
    return Router(model_list)


def make_embedding_router() -> Router:
    fake_api_key = "fakekeyvalue"
    model_list = [
        {
            "model_name": "openai/text-embedding-3-small",
            "litellm_params": {
                "model": "openai/text-embedding-3-small",
                "api_key": fake_api_key,
            },
        },
        {
            "model_name": "openai/text-embedding-3-small",
            "litellm_params": {
                "model": "openai/text-embedding-3-small",
                "api_key": fake_api_key,
            },
        },
    ]
    return Router(model_list)


def mock_embedding_response(texts: list[str]) -> MagicMock:
    """Create a mock litellm embedding response."""
    mock_response = MagicMock()
    mock_response.data = [
        {"embedding": [0.1, 0.2, 0.3], "index": i} for i in range(len(texts))
    ]
    return mock_response
