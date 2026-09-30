import json
from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import MagicMock

import httpx
import litellm
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from litellm import Router
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import TypeAdapter

# Newer litellm sends manual thinking to Opus 4.7 as adaptive thinking.
OPUS_4_7_THINKS_ADAPTIVELY = (
    litellm.get_optional_params(
        model="claude-opus-4-7",
        custom_llm_provider="anthropic",
        drop_params=True,
        thinking={"type": "enabled", "budget_tokens": 1024},
    ).get("thinking")
    or {}
).get("type") == "adaptive"


def serve_http(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any],
    events: Sequence[dict[str, Any]] = (),
) -> list[httpx.Request]:
    """Answer every request litellm sends with ``body``, in-process.

    A request that asks to stream gets ``events`` as server-sent events instead.
    litellm sends async calls through its own aiohttp transport, not httpx's.
    """
    requests: list[httpx.Request] = []
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()

    def _reply(_: object, request: httpx.Request) -> httpx.Response:
        requests.append(request)
        # Gemini asks to stream in its URL rather than its body.
        if json.loads(request.content).get("stream") or request.url.path.endswith(
            ":streamGenerateContent"
        ):
            return httpx.Response(
                200,
                content=stream,
                headers={"content-type": "text/event-stream"},
                request=request,
            )
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
            # Required since openai 2.45.0; older SDKs keep it as an extra.
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
    }
    Response.model_validate(reply)
    return reply


def responses_api_events(*events: dict[str, Any]) -> list[dict[str, Any]]:
    """Responses API stream ``events`` in order, each checked by the openai SDK."""
    numbered = [
        {**event, "sequence_number": number} for number, event in enumerate(events)
    ]
    TypeAdapter(list[ResponseStreamEvent]).validate_python(numbered)
    return numbered


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


def web_search_call_item() -> dict[str, Any]:
    return {
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "weather in Paris"},
    }


def reasoning_item(
    item_id: str, summary: str, encrypted_content: str | None = None
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "reasoning",
        "id": item_id,
        "summary": [{"type": "summary_text", "text": summary}],
    }
    if encrypted_content is not None:
        item["encrypted_content"] = encrypted_content
    return item


def chat_completion_reply(
    *contents: str, usage: dict[str, int] | None = None
) -> dict[str, Any]:
    """A Chat Completions reply with one choice per content, checked by the openai SDK."""
    reply: dict[str, Any] = {
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
    if usage is not None:
        reply["usage"] = usage
    ChatCompletion.model_validate(reply)
    return reply


def chat_completion_events(content: str, usage: dict[str, int]) -> list[dict[str, Any]]:
    """``content`` streamed as Chat Completions chunks, then OpenAI's trailing usage chunk."""
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o-mini",
    }
    events = [
        {
            **chunk,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        },
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": usage},
    ]
    TypeAdapter(list[ChatCompletionChunk]).validate_python(events)
    return events


def gemini_reply(text: str, grounding: dict[str, Any]) -> dict[str, Any]:
    """A Gemini reply grounded by web search; a stream sends it as its one event."""
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}], "role": "model"},
                "finishReason": "STOP",
                "groundingMetadata": grounding,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2,
        },
    }


async def whole_reply(llm: BaseChatModel, method: str) -> BaseMessage:
    """The reply to "hi" read through ``method``, with a stream's chunks merged."""
    if method == "invoke":
        return llm.invoke("hi")
    if method == "ainvoke":
        return await llm.ainvoke("hi")
    if method == "stream":
        chunks = list(llm.stream("hi"))
    else:
        chunks = [chunk async for chunk in llm.astream("hi")]
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    return merged


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


def serve_requests(
    monkeypatch: pytest.MonkeyPatch,
    reply: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Answer every request litellm sends with ``reply(request)``, in-process.

    Unlike ``serve_http``, the answer can depend on the request.
    """

    def _reply(_: object, request: httpx.Request) -> httpx.Response:
        return reply(request)

    async def _areply(_: object, request: httpx.Request) -> httpx.Response:
        return reply(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _reply)
    monkeypatch.setattr(LiteLLMAiohttpTransport, "handle_async_request", _areply)


def stream_reply(
    request: httpx.Request, events: Sequence[dict[str, Any]]
) -> httpx.Response:
    """``events`` as the server-sent events answering ``request``."""
    return httpx.Response(
        200,
        content="".join(f"data: {json.dumps(event)}\n\n" for event in events).encode(),
        headers={"content-type": "text/event-stream"},
        request=request,
    )


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
