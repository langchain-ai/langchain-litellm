"""Signed thinking survives a tool-loop continuation on Anthropic-format routes.

Anthropic-format providers, including Bedrock and Anthropic-compatible endpoints
such as Kimi For Coding, expect a tool-call turn to come back with its signed
thinking block. litellm rebuilds that block only from the assistant message's
``thinking_blocks`` key, so the connector captures the signed blocks inbound and
hands them back outbound, and only where the destination checks them. The wire
tests patch only the transport: the connector, litellm's request transform, its
SSE parser and its clients all run.
"""

# stdlib
import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

# third-party
import httpx
import litellm
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from litellm.llms.anthropic.chat.handler import ModelResponseIterator
from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from litellm.types.utils import Message

# first-party
from langchain_litellm.chat_models import ChatLiteLLM, ChatLiteLLMRouter
from langchain_litellm.chat_models.litellm import (
    _attach_thinking_blocks,
    _convert_delta_to_message_chunk,
    _convert_message_to_dict,
    _endpoint_name,
    _keep_thinking_blocks,
    _signing_endpoint,
    _ThinkingBlockAssembler,
)

THOUGHT = ["The user wants the Paris weather. ", "I will call get_weather."]
SIGNED = {"type": "thinking", "thinking": "".join(THOUGHT), "signature": "sig-1=="}
SECOND = {"type": "thinking", "thinking": "Second.", "signature": "sig-2=="}
REDACTED = {"type": "redacted_thinking", "data": "opaque=="}
TOOL_CALL = {"name": "get_weather", "args": {"city": "Paris"}, "id": "toolu_01"}
GATEWAY = "http://gateway.internal:4000"
KIMI_BASE = "https://api.kimi.com/coding/"


def endpoint(model: str, api_base: str | None = None) -> str:
    """The stored origin of a block this model signed at this base."""
    origin = _signing_endpoint(model, None, api_base)
    assert origin is not None
    return origin


ANTHROPIC = endpoint("anthropic/claude-sonnet-4-20250514")
KIMI = endpoint("anthropic/kimi-for-coding", "https://api.kimi.com/coding/")


def signed_at(endpoint: str, *blocks: dict[str, Any]) -> list[dict[str, Any]]:
    """Blocks as stored after a response: each marked with the endpoint that signed it."""
    return [{**block, "origin": endpoint} for block in blocks]


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


# ── which endpoints check replayed thinking ──────────────────────────────────


@pytest.fixture
def _no_anthropic_base(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "api_base", None)


ARN = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc"
CLAUDE_ON_BEDROCK = "anthropic.claude-sonnet-4-20250514-v1:0"


@pytest.mark.usefixtures("_no_anthropic_base")
@pytest.mark.parametrize(
    ("model", "provider", "api_base", "base_model", "expected"),
    [
        (
            "anthropic/claude-sonnet-4-20250514",
            None,
            None,
            None,
            "anthropic||claude-sonnet-4-20250514",
        ),
        (
            "claude-sonnet-4-20250514",
            None,
            None,
            None,
            "anthropic||claude-sonnet-4-20250514",
        ),
        (
            "anthropic/kimi-for-coding",
            None,
            "https://api.kimi.com/coding/",
            None,
            "anthropic|https://api.kimi.com/coding|kimi-for-coding",
        ),
        (
            "kimi-for-coding",
            "anthropic",
            "HTTPS://API.KIMI.COM/coding",
            None,
            "anthropic|https://api.kimi.com/coding|kimi-for-coding",
        ),
        (
            "bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
            None,
            None,
            None,
            "bedrock||converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
        ),
        (
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            None,
            None,
            None,
            "bedrock||us.anthropic.claude-sonnet-4-20250514-v1:0",
        ),
        (
            f"bedrock/converse/{ARN}",
            None,
            None,
            "anthropic.claude-sonnet-4-20250514-v1:0",
            f"bedrock||converse/{ARN}",
        ),
        (f"bedrock/converse/{ARN}", None, None, None, None),
        (
            "vertex_ai/claude-3-7-sonnet@20250219",
            None,
            None,
            None,
            "vertex_ai||claude-3-7-sonnet@20250219",
        ),
        ("azure_ai/claude-sonnet-4", None, None, None, "azure_ai||claude-sonnet-4"),
        ("gemini/gemini-2.5-pro", None, None, None, None),
        ("openai/gpt-4o", None, None, None, None),
        ("deepseek/deepseek-reasoner", None, None, None, None),
        ("mistral/magistral-medium-latest", None, None, None, None),
        ("github_copilot/claude-sonnet-4", None, None, None, None),
        ("openrouter/anthropic/claude-sonnet-4", None, None, None, None),
        ("bedrock/meta.llama3-70b-instruct-v1:0", None, None, None, None),
        ("my-router-alias", None, None, None, None),
        (None, None, None, None, None),
    ],
)
def test_only_anthropic_format_routes_name_a_signing_endpoint(
    model: str | None,
    provider: str | None,
    api_base: str | None,
    base_model: str | None,
    expected: str | None,
) -> None:
    """Kimi and Anthropic both speak the Anthropic format but sign differently."""
    assert _endpoint_name(model, provider, api_base, base_model) == expected


@pytest.mark.usefixtures("_no_anthropic_base")
def test_an_unset_base_names_where_litellm_actually_sends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://gw.example/")

    assert _endpoint_name("anthropic/claude-x", None, None) == (
        "anthropic|https://gw.example|claude-x"
    )


def test_a_stored_origin_never_carries_the_api_base() -> None:
    """Traces and checkpoints keep additional_kwargs, so the origin is a digest."""
    origin = _signing_endpoint(
        "anthropic/claude-x", None, "https://user:secret@gw.example.com/v1?key=ABC"
    )

    assert origin is not None
    assert len(origin) == 16
    assert not any(part in origin for part in ("secret", "gw.example", "abc", "user"))


# ── the stored shape ─────────────────────────────────────────────────────────


def _litellm_message(**fields: Any) -> Any:
    """A litellm Message, the object a response actually carries, built untyped."""
    return Message(**fields)


def _kept(message: Any, origin: str | None) -> Any:
    """What a non-streaming call stores from a response holding ``message``."""
    response = {"choices": [{"message": message, "finish_reason": "stop"}]}
    result = ChatLiteLLM(model=CLAUDE, api_key="fake")._create_chat_result(response)
    return _keep_thinking_blocks(result, response, origin).generations[0].message


def test_non_streaming_response_stores_only_signed_blocks_with_their_origin() -> None:
    message = _kept(
        _litellm_message(
            content="",
            reasoning_content=SIGNED["thinking"],
            thinking_blocks=[
                {"type": "thinking", "thinking": "unsigned"},
                {"type": "thinking", "thinking": "blank", "signature": ""},
                SIGNED,
                REDACTED,
            ],
        ),
        KIMI,
    )

    assert message.additional_kwargs["thinking_blocks"] == signed_at(
        KIMI, SIGNED, REDACTED
    )
    assert message.content == ""


def test_response_without_signed_blocks_stores_no_key() -> None:
    message = _kept(
        {
            "role": "assistant",
            "content": "hi",
            "thinking_blocks": [{"type": "thinking", "thinking": "r"}],
        },
        ANTHROPIC,
    )

    assert "thinking_blocks" not in message.additional_kwargs


def test_blocks_are_not_captured_without_an_origin() -> None:
    """A route that never checks the signature must not collect blocks to replay."""
    message = _kept(_litellm_message(content="", thinking_blocks=[SIGNED]), None)

    assert "thinking_blocks" not in message.additional_kwargs


# ── streaming: the summed chunks equal the non-streaming list ─────────────────


def _anthropic_deltas() -> list[Any]:
    """litellm's own parser on thinking, redacted and thinking blocks."""
    events: list[dict[str, Any]] = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        *[
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": part},
            }
            for part in THOUGHT
        ],
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": SIGNED["signature"]},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "redacted_thinking", "data": REDACTED["data"]},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "thinking_delta", "thinking": "Second."},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "signature_delta", "signature": "sig-2=="},
        },
        {"type": "content_block_stop", "index": 2},
    ]
    parser = ModelResponseIterator(
        streaming_response=iter([]), sync_stream=True, json_mode=False
    )
    return [r.choices[0].delta for r in map(parser.chunk_parser, events) if r.choices]


def _bedrock_converse_events(
    blocks: list[tuple[list[str], str]],
) -> list[dict[str, Any]]:
    """Bedrock closes a block with a signature and no text, unlike Anthropic."""
    events: list[dict[str, Any]] = []
    for index, (parts, signature) in enumerate(blocks):
        events += [
            {"delta": {"reasoningContent": {"text": p}}, "contentBlockIndex": index}
            for p in parts
        ]
        events.append(
            {
                "delta": {"reasoningContent": {"signature": signature}},
                "contentBlockIndex": index,
            }
        )
    return events


def _bedrock_converse_deltas() -> list[Any]:
    decoder = AWSEventStreamDecoder(model="anthropic.claude-sonnet-4")
    events = _bedrock_converse_events(
        [(THOUGHT, SIGNED["signature"]), (["Second."], "sig-2==")]
    )
    return [
        r.choices[0].delta
        for r in map(decoder.converse_chunk_parser, events)
        if r.choices
    ]


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        (_anthropic_deltas, [SIGNED, REDACTED, SECOND]),
        (_bedrock_converse_deltas, [SIGNED, SECOND]),
    ],
    ids=["anthropic", "bedrock-converse"],
)
@pytest.mark.parametrize("dumped", [True, False], ids=["base-dumps", "router-delta"])
def test_streamed_chunks_sum_to_the_non_streaming_blocks(
    deltas: Callable[[], list[Any]], expected: list[dict[str, Any]], dumped: bool
) -> None:
    thinking = _ThinkingBlockAssembler(ANTHROPIC)
    total = None
    for delta in deltas():
        chunk = _convert_delta_to_message_chunk(
            delta.model_dump() if dumped else delta, AIMessageChunk, thinking
        )
        total = chunk if total is None else total + chunk

    assert total is not None
    assert total.additional_kwargs["thinking_blocks"] == signed_at(ANTHROPIC, *expected)


# ── the request side ─────────────────────────────────────────────────────────


def _turn(blocks: list[dict[str, Any]], content: Any = "") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[TOOL_CALL],
        additional_kwargs={"thinking_blocks": blocks},
    )


def _attached(message: AIMessage, endpoint: str | None) -> dict[str, Any]:
    message_dicts = [_convert_message_to_dict(message)]
    _attach_thinking_blocks([message], message_dicts, endpoint)
    return message_dicts[0]


def test_converting_a_message_alone_never_sends_thinking_blocks() -> None:
    """Only an entry point that knows the destination may attach them."""
    assert "thinking_blocks" not in _convert_message_to_dict(
        _turn(signed_at(ANTHROPIC, SIGNED))
    )


def test_blocks_go_back_to_their_endpoint_with_only_provider_keys() -> None:
    """Inline thinking stays stripped from content; the key is what litellm reads."""
    message = _turn(
        [
            {
                **SIGNED,
                "index": 0,
                "cache_control": {"type": "ephemeral"},
                "origin": KIMI,
            },
            {"type": "thinking", "thinking": "unsigned", "origin": KIMI},
            {**REDACTED, "origin": KIMI},
        ],
        content=[
            {"type": "thinking", "thinking": "inline"},
            {"type": "redacted_thinking", "data": "inline"},
            {"type": "text", "text": "hello"},
        ],
    )

    sent = _attached(message, KIMI)

    assert sent["content"] == [{"type": "text", "text": "hello"}]
    assert sent["thinking_blocks"] == [SIGNED, REDACTED]
    sent["thinking_blocks"][0]["thinking"] = "changed"
    assert (
        message.additional_kwargs["thinking_blocks"][0]["thinking"]
        == SIGNED["thinking"]
    )


def test_blocks_never_go_to_an_endpoint_that_did_not_sign_them() -> None:
    """Kimi's signatures are not Anthropic's, though both speak its format."""
    sent = _attached(_turn(signed_at(KIMI, SIGNED)), ANTHROPIC)

    assert "thinking_blocks" not in sent
    assert sent["content"] == ""


@pytest.mark.parametrize(
    "blocks",
    [
        [SIGNED],
        [{**SIGNED, "origin": ANTHROPIC}, {**SECOND, "origin": KIMI}],
    ],
    ids=["no-origin", "mixed-origins"],
)
def test_a_turn_goes_back_only_when_every_block_names_the_endpoint(
    blocks: list[dict[str, Any]],
) -> None:
    assert "thinking_blocks" not in _attached(_turn(blocks), ANTHROPIC)


def test_request_omits_the_key_when_nothing_is_signed() -> None:
    """litellm treats a present key as "has thinking", so an unsigned-only key would
    keep thinking enabled while every block is dropped before the request."""
    unsigned = [{"type": "thinking", "thinking": "x", "origin": ANTHROPIC}]

    assert "thinking_blocks" not in _attached(_turn(unsigned), ANTHROPIC)


def test_a_replayed_empty_tool_turn_goes_out_with_no_content() -> None:
    """litellm fills an empty string with placeholder text, editing the signed turn."""
    sent = _attached(_turn(signed_at(ANTHROPIC, SIGNED)), ANTHROPIC)

    assert sent["content"] is None
    assert sent["thinking_blocks"] == [SIGNED]


def test_a_destination_that_does_not_check_gets_the_turn_unchanged() -> None:
    sent = _attached(_turn(signed_at(ANTHROPIC, SIGNED)), None)

    assert "thinking_blocks" not in sent
    assert sent["content"] == ""


# ── the entry points decide per request ──────────────────────────────────────


def _history(endpoint: str = ANTHROPIC) -> list[Any]:
    return [
        HumanMessage("What's the weather in Paris?"),
        _turn(signed_at(endpoint, SIGNED)),
        ToolMessage("Sunny", tool_call_id="toolu_01"),
    ]


def _assistant_sent(captured: dict[str, Any]) -> dict[str, Any]:
    return next(m for m in captured["messages"] if m["role"] == "assistant")


def _reply(**message: Any) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok", **message},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _capture_calls(monkeypatch: pytest.MonkeyPatch, cls: type) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake(self: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _reply(thinking_blocks=[SIGNED])

    monkeypatch.setattr(cls, "completion_with_retry", fake)
    return captured


@pytest.mark.parametrize(
    ("kwargs", "history", "forwarded"),
    [
        ({"model": "anthropic/claude-sonnet-4-20250514"}, ANTHROPIC, True),
        ({"model": "openai/gpt-4o"}, ANTHROPIC, False),
        (
            {"model": "anthropic/claude-sonnet-4-20250514"},
            KIMI,
            False,
        ),
        (
            {
                "model": "anthropic/kimi-for-coding",
                "api_base": "https://api.kimi.com/coding/",
            },
            KIMI,
            True,
        ),
        (
            {
                "model": "anthropic/claude-sonnet-4-20250514",
                "model_kwargs": {"fallbacks": ["gemini/gemini-2.5-pro"]},
            },
            ANTHROPIC,
            False,
        ),
        (
            {"model": "anthropic/claude-sonnet-4-20250514", "api_base": GATEWAY},
            endpoint("anthropic/kimi-k2", GATEWAY),
            False,
        ),
        (
            {"model": "anthropic/claude-sonnet-4-20250514", "api_base": GATEWAY},
            endpoint("anthropic/claude-sonnet-4-20250514", GATEWAY),
            True,
        ),
    ],
    ids=[
        "same-endpoint",
        "openai",
        "kimi-signed",
        "kimi",
        "with-fallback",
        "gateway-other-model",
        "gateway-same-model",
    ],
)
def test_base_forwards_only_to_the_one_endpoint_that_signed(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    history: str,
    forwarded: bool,
) -> None:
    captured = _capture_calls(monkeypatch, ChatLiteLLM)

    ChatLiteLLM(api_key="fake", **kwargs).invoke(_history(history))

    assert ("thinking_blocks" in _assistant_sent(captured)) is forwarded


@pytest.mark.parametrize(
    ("call", "stored"),
    [
        ({"model": "openai/gpt-4o"}, None),
        ({"fallbacks": ["gemini/gemini-2.5-pro"]}, None),
        (
            {"base_url": "https://api.kimi.com/coding/"},
            endpoint(
                "anthropic/claude-sonnet-4-20250514", "https://api.kimi.com/coding/"
            ),
        ),
    ],
    ids=["redirect", "fallback", "base-url"],
)
def test_a_per_call_setting_decides_for_the_call(
    monkeypatch: pytest.MonkeyPatch, call: dict[str, Any], stored: str | None
) -> None:
    """History signed elsewhere never goes; what comes back is marked with this call's
    one endpoint, or not kept when the call could reach more than one."""
    captured = _capture_calls(monkeypatch, ChatLiteLLM)
    llm = ChatLiteLLM(model="anthropic/claude-sonnet-4-20250514", api_key="fake")

    message = llm.invoke(_history(), **call)

    assert "thinking_blocks" not in _assistant_sent(captured)
    kept = message.additional_kwargs.get("thinking_blocks")
    assert (kept[0]["origin"] if kept else None) == stored


@pytest.mark.parametrize(
    ("kwargs", "stored"),
    [
        ({"model": "anthropic/claude-sonnet-4-20250514"}, signed_at(ANTHROPIC, SIGNED)),
        ({"model": "gemini/gemini-2.5-pro"}, None),
        (
            {
                "model": "anthropic/claude-sonnet-4-20250514",
                "model_kwargs": {"fallbacks": ["gemini/gemini-2.5-pro"]},
            },
            None,
        ),
    ],
    ids=["anthropic", "gemini", "fallback-may-answer"],
)
def test_blocks_are_captured_only_when_the_signer_is_known(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], stored: Any
) -> None:
    _capture_calls(monkeypatch, ChatLiteLLM)

    message = ChatLiteLLM(api_key="fake", **kwargs).invoke("hi")

    assert message.additional_kwargs.get("thinking_blocks") == stored


def _router(deployments: list[tuple[str, str]], **settings: Any) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {"model_name": group, "litellm_params": {"model": model, "api_key": "k"}}
            for group, model in deployments
        ],
        num_retries=0,
        **settings,
    )


CLAUDE = "anthropic/claude-sonnet-4-20250514"


@pytest.mark.parametrize(
    ("deployments", "settings", "call", "forwarded"),
    [
        ([("main", CLAUDE)], {}, {}, True),
        ([("main", CLAUDE), ("main", CLAUDE)], {}, {}, True),
        (
            [
                ("main", CLAUDE),
                ("main", "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0"),
            ],
            {},
            {},
            False,
        ),
        ([("main", CLAUDE), ("main", "openai/gpt-4o")], {}, {}, False),
        (
            [("main", CLAUDE), ("backup", "openai/gpt-4o")],
            {"fallbacks": [{"main": ["backup"]}]},
            {},
            False,
        ),
        (
            [("main", CLAUDE), ("*", "*")],
            {"fallbacks": [{"main": ["gemini/gemini-2.5-pro"]}]},
            {},
            False,
        ),
        (
            [("main", CLAUDE), ("backup", CLAUDE)],
            {"default_fallbacks": ["backup"]},
            {},
            False,
        ),
        (
            [("main", CLAUDE), ("gem", "gemini/gemini-2.5-pro")],
            {},
            {"context_window_fallbacks": [{"main": ["gem"]}]},
            False,
        ),
        ([("main", CLAUDE)], {}, {"api_base": "https://api.kimi.com/coding/"}, False),
        (
            [("main", CLAUDE), ("gpt", "openai/gpt-4o")],
            {"model_group_alias": {"main": "gpt"}},
            {},
            False,
        ),
    ],
    ids=[
        "claude",
        "same-endpoint-twice",
        "anthropic-and-bedrock",
        "mixed-group",
        "falls-back-to-openai",
        "falls-back-to-a-wildcard",
        "default-fallback",
        "per-call-context-window-fallback",
        "per-call-api-base",
        "alias-shadows-the-group",
    ],
)
def test_router_forwards_only_when_every_reachable_deployment_signs_the_same(
    monkeypatch: pytest.MonkeyPatch,
    deployments: list[tuple[str, str]],
    settings: dict[str, Any],
    call: dict[str, Any],
    forwarded: bool,
) -> None:
    captured = _capture_calls(monkeypatch, ChatLiteLLMRouter)
    llm = ChatLiteLLMRouter(router=_router(deployments, **settings), model_name="main")

    llm.invoke(_history(), **call)

    assert ("thinking_blocks" in _assistant_sent(captured)) is forwarded


def test_router_counts_a_fallback_keyed_by_the_group_without_its_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_calls(monkeypatch, ChatLiteLLMRouter)
    router = _router(
        [(CLAUDE, CLAUDE), ("backup", "openai/gpt-4o")],
        fallbacks=[{"claude-sonnet-4-20250514": ["backup"]}],
    )

    ChatLiteLLMRouter(router=router, model_name=CLAUDE).invoke(_history())

    assert "thinking_blocks" not in _assistant_sent(captured)


# ── every guard, at the entry points ─────────────────────────────────────────


def test_a_subclass_overriding_create_chat_result_still_captures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_create_chat_result`` is an extension point, so its signature stays as is."""

    class Sub(ChatLiteLLM):
        def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
            return super()._create_chat_result(response)

    _capture_calls(monkeypatch, Sub)

    message = Sub(model=CLAUDE, api_key="fake").invoke("hi")

    assert message.additional_kwargs["thinking_blocks"] == signed_at(ANTHROPIC, SIGNED)


@pytest.mark.parametrize(
    ("model", "history", "reason"),
    [
        ("openai/gpt-4o", ANTHROPIC, "no single signing endpoint"),
        (CLAUDE, KIMI, "another endpoint signed them"),
    ],
    ids=["no-endpoint", "other-signer"],
)
def test_withheld_blocks_are_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    model: str,
    history: str,
    reason: str,
) -> None:
    _capture_calls(monkeypatch, ChatLiteLLM)

    with caplog.at_level(logging.DEBUG, logger="langchain_litellm.chat_models.litellm"):
        ChatLiteLLM(model=model, api_key="fake").invoke(_history(history))

    assert reason in caplog.text


def test_a_result_that_does_not_pair_with_its_choices_keeps_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _reply(thinking_blocks=[SIGNED])

    with caplog.at_level(logging.DEBUG, logger="langchain_litellm.chat_models.litellm"):
        result = _keep_thinking_blocks(ChatResult(generations=[]), response, ANTHROPIC)

    assert result.generations == []
    assert "generations and choices differ" in caplog.text


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"fallbacks": ["gemini/gemini-2.5-pro"]},
        {"context_window_fallback_dict": {CLAUDE: "gemini/gemini-2.5-pro"}},
    ],
    ids=["fallbacks", "context-window-fallback"],
)
def test_base_neither_replays_nor_keeps_when_litellm_can_fall_back(
    monkeypatch: pytest.MonkeyPatch, model_kwargs: dict[str, Any]
) -> None:
    captured = _capture_calls(monkeypatch, ChatLiteLLM)

    message = ChatLiteLLM(
        model=CLAUDE, api_key="fake", model_kwargs=model_kwargs
    ).invoke(_history())

    assert "thinking_blocks" not in _assistant_sent(captured)
    assert "thinking_blocks" not in message.additional_kwargs


def test_base_replays_nothing_under_global_model_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "model_fallbacks", ["gemini/gemini-2.5-pro"])
    captured = _capture_calls(monkeypatch, ChatLiteLLM)

    ChatLiteLLM(model=CLAUDE, api_key="fake").invoke(_history())

    assert "thinking_blocks" not in _assistant_sent(captured)


@pytest.mark.usefixtures("_no_anthropic_base")
@pytest.mark.parametrize(
    "setting", ["litellm.api_base", "ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"]
)
def test_base_names_the_base_litellm_falls_back_to(
    monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    unset = endpoint(CLAUDE)
    if setting == "litellm.api_base":
        monkeypatch.setattr(litellm, "api_base", GATEWAY)
    else:
        monkeypatch.setenv(setting, GATEWAY)
    captured = _capture_calls(monkeypatch, ChatLiteLLM)
    llm = ChatLiteLLM(model=CLAUDE, api_key="fake")

    llm.invoke(_history(unset))
    assert "thinking_blocks" not in _assistant_sent(captured)
    llm.invoke(_history(endpoint(CLAUDE, GATEWAY)))
    assert "thinking_blocks" in _assistant_sent(captured)


@pytest.mark.parametrize(
    ("model", "llm_kwargs", "call", "signed_by"),
    [
        (
            CLAUDE,
            {"api_base": KIMI_BASE},
            {"base_url": GATEWAY},
            (CLAUDE, GATEWAY, None),
        ),
        (
            "kimi-for-coding",
            {"custom_llm_provider": "anthropic", "api_base": KIMI_BASE},
            {},
            ("anthropic/kimi-for-coding", KIMI_BASE, None),
        ),
        (
            f"bedrock/converse/{ARN}",
            {"model_kwargs": {"base_model": CLAUDE_ON_BEDROCK}},
            {},
            (f"bedrock/converse/{ARN}", None, CLAUDE_ON_BEDROCK),
        ),
    ],
    ids=["base-url-beats-api-base", "provider-apart-from-model", "arn-with-base-model"],
)
def test_base_names_the_endpoint_litellm_actually_uses(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    llm_kwargs: dict[str, Any],
    call: dict[str, Any],
    signed_by: tuple[str, str | None, str | None],
) -> None:
    captured = _capture_calls(monkeypatch, ChatLiteLLM)
    origin = _signing_endpoint(signed_by[0], None, signed_by[1], signed_by[2])
    assert origin is not None

    ChatLiteLLM(model=model, api_key="fake", **llm_kwargs).invoke(
        _history(origin), **call
    )

    assert "thinking_blocks" in _assistant_sent(captured)


def test_a_turn_without_tool_calls_keeps_its_content() -> None:
    message = AIMessage(
        content="", additional_kwargs={"thinking_blocks": signed_at(ANTHROPIC, SIGNED)}
    )

    sent = _attached(message, ANTHROPIC)

    assert sent["thinking_blocks"] == [SIGNED]
    assert sent["content"] == ""


def test_a_stored_block_keeps_only_well_formed_values() -> None:
    raw: list[dict[str, Any]] = [
        {"type": "thinking", "thinking": None, "signature": "s=="},
        {"type": "redacted_thinking", "data": ""},
    ]

    assert _ThinkingBlockAssembler(ANTHROPIC).feed(raw) == signed_at(
        ANTHROPIC, {"type": "thinking", "thinking": "", "signature": "s=="}
    )
    assert _attached(_turn(signed_at(ANTHROPIC, *raw)), ANTHROPIC)[
        "thinking_blocks"
    ] == [{"type": "thinking", "thinking": "", "signature": "s=="}]


def test_a_redacted_block_ends_the_text_buffered_before_it() -> None:
    """Bedrock closes a block with no text, so leftover fragments must not carry over."""
    thinking = _ThinkingBlockAssembler(ANTHROPIC)
    thinking.feed([{"type": "thinking", "thinking": "stale "}])
    thinking.feed([REDACTED])

    closed = thinking.feed([{"type": "thinking", "thinking": "", "signature": "s=="}])

    assert closed == signed_at(
        ANTHROPIC, {"type": "thinking", "thinking": "", "signature": "s=="}
    )


def test_a_captured_origin_carries_nothing_of_the_api_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Traces and checkpoints keep additional_kwargs, so the origin is a digest."""
    _capture_calls(monkeypatch, ChatLiteLLM)
    base = "https://user:secret@gw.example.com/v1?key=ABC"

    message = ChatLiteLLM(model=CLAUDE, api_key="fake", api_base=base).invoke("hi")

    stored = repr(message.additional_kwargs)
    assert message.additional_kwargs["thinking_blocks"]
    assert not any(part in stored for part in ("secret", "gw.example", "key=", "user:"))


def _router_of(entries: list[dict[str, Any]], **settings: Any) -> litellm.Router:
    return litellm.Router(model_list=entries, num_retries=0, **settings)


def _entry(group: str, model: str, **extra: Any) -> dict[str, Any]:
    info = extra.pop("model_info", None)
    entry: dict[str, Any] = {
        "model_name": group,
        "litellm_params": {"model": model, "api_key": "k", **extra},
    }
    if info is not None:
        entry["model_info"] = info
    return entry


@pytest.mark.parametrize(
    ("entries", "settings", "history", "forwarded"),
    [
        (
            [_entry("main", CLAUDE), _entry("safe", CLAUDE)],
            {"content_policy_fallbacks": [{"main": ["safe"]}]},
            ANTHROPIC,
            False,
        ),
        (
            [_entry("main", CLAUDE, fallbacks=["gemini/gemini-2.5-pro"])],
            {},
            ANTHROPIC,
            False,
        ),
        (
            [_entry("main", CLAUDE), _entry("other", CLAUDE), _entry("x", CLAUDE)],
            {"fallbacks": [{"other": ["x"]}]},
            ANTHROPIC,
            False,
        ),
        (
            [
                _entry(
                    "main",
                    "anthropic/kimi-for-coding",
                    base_url="https://api.kimi.com/coding/",
                )
            ],
            {},
            KIMI,
            True,
        ),
        (
            [
                _entry(
                    "main",
                    "kimi-for-coding",
                    custom_llm_provider="anthropic",
                    api_base="https://api.kimi.com/coding/",
                )
            ],
            {},
            KIMI,
            True,
        ),
        (
            [
                _entry(
                    "main",
                    f"bedrock/converse/{ARN}",
                    model_info={"base_model": CLAUDE_ON_BEDROCK},
                )
            ],
            {},
            "arn",
            True,
        ),
    ],
    ids=[
        "content-policy-fallback",
        "deployment-level-fallback",
        "any-router-fallback",
        "deployment-base-url",
        "deployment-provider",
        "model-info-base-model",
    ],
)
def test_router_resolves_each_deployment_like_the_base(
    monkeypatch: pytest.MonkeyPatch,
    entries: list[dict[str, Any]],
    settings: dict[str, Any],
    history: str,
    forwarded: bool,
) -> None:
    if history == "arn":
        arn_origin = _signing_endpoint(
            f"bedrock/converse/{ARN}", None, None, CLAUDE_ON_BEDROCK
        )
        assert arn_origin is not None
        history = arn_origin
    captured = _capture_calls(monkeypatch, ChatLiteLLMRouter)
    llm = ChatLiteLLMRouter(router=_router_of(entries, **settings), model_name="main")

    llm.invoke(_history(history))

    assert ("thinking_blocks" in _assistant_sent(captured)) is forwarded


def test_a_group_whose_deployments_differ_has_no_endpoint() -> None:
    """Deterministic, whichever deployment a set would hand back first."""
    router = _router_of(
        [
            _entry("main", CLAUDE),
            _entry("main", "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0"),
        ]
    )

    llm = ChatLiteLLMRouter(router=router, model_name="main")

    assert llm._thinking_endpoint({"model": "main"}) is None


# ── end to end: the continuation's HTTP body ─────────────────────────────────


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    ).encode()


def _message_start(message_id: str) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }


def _tool_turn(stream: bool, block: dict[str, Any], tool_id: str) -> httpx.Response:
    """Claude's answer: one signed thinking block, then a tool call."""
    if not stream:
        return httpx.Response(
            200,
            json={
                "id": f"msg_{tool_id}",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [
                    block,
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    parts = [block["thinking"][:10], block["thinking"][10:]]
    events = [
        _message_start(f"msg_{tool_id}"),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        *[
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": p},
            }
            for p in parts
        ],
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": block["signature"]},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": "get_weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city": "Paris"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=_sse(events)
    )


def _text_turn(stream: bool) -> httpx.Response:
    if not stream:
        return httpx.Response(
            200,
            json={
                "id": "msg_text",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [{"type": "text", "text": "Sunny."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    events = [
        _message_start("msg_text"),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Sunny."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=_sse(events)
    )


# Claude's answer to the nth request of the loop, by how many tool results it holds.
SCRIPT = [
    lambda stream: _tool_turn(stream, SIGNED, "toolu_01"),
    lambda stream: _tool_turn(stream, SECOND, "toolu_02"),
    _text_turn,
]


@pytest.fixture
def anthropic_wire(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Answer litellm's Anthropic requests in process and record every body."""
    bodies: list[dict[str, Any]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        bodies.append(body)
        results = sum(
            b.get("type") == "tool_result"
            for m in body["messages"]
            if isinstance(m["content"], list)
            for b in m["content"]
        )
        response = SCRIPT[results](bool(body.get("stream")))
        response.request = request
        return response

    async def answer_async(self: Any, request: httpx.Request) -> httpx.Response:
        await request.aread()
        return answer(request)

    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request", lambda self, r: answer(r)
    )
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", answer_async)
    monkeypatch.setattr(LiteLLMAiohttpTransport, "handle_async_request", answer_async)
    return bodies


def _model(kind: str) -> Any:
    params: dict[str, Any] = {
        "model": "anthropic/kimi-for-coding",
        "api_base": "https://api.kimi.com/coding/",
        "api_key": "fake",
    }
    extra: dict[str, Any] = {
        "max_tokens": 1024,
        "max_retries": 1,
        "model_kwargs": {
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "allowed_openai_params": ["thinking"],
        },
    }
    if kind == "base":
        return ChatLiteLLM(**params, **extra)
    router = litellm.Router(
        model_list=[{"model_name": "kimi", "litellm_params": params}], num_retries=0
    )
    return ChatLiteLLMRouter(router=router, model_name="kimi", **extra)


def _run(llm: Any, mode: str, messages: list[Any]) -> Any:
    if mode == "invoke":
        return llm.invoke(messages)
    if mode == "ainvoke":
        return asyncio.run(llm.ainvoke(messages))
    if mode == "stream":
        chunks = list(llm.stream(messages))
    else:

        async def collect() -> list[Any]:
            return [c async for c in llm.astream(messages)]

        chunks = asyncio.run(collect())
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


@pytest.mark.parametrize("mode", ["invoke", "stream", "ainvoke", "astream"])
@pytest.mark.parametrize("kind", ["base", "router"])
def test_every_replayed_turn_goes_out_exactly_as_claude_returned_it(
    anthropic_wire: list[dict[str, Any]], kind: str, mode: str
) -> None:
    """A three-request tool loop: each earlier assistant turn goes back unedited.

    The signature covers the whole turn, so any edit counts: a missing block, a
    block moved, or placeholder text litellm writes into an empty turn.
    """
    llm = _model(kind).bind_tools([WEATHER_TOOL])
    history: list[Any] = [HumanMessage("What's the weather in Paris?")]

    for tool_id in ("toolu_01", "toolu_02"):
        history.append(_run(llm, mode, history))
        history.append(ToolMessage("Sunny", tool_call_id=tool_id))
    _run(llm, mode, history)

    replayed = [m for m in anthropic_wire[-1]["messages"] if m["role"] == "assistant"]
    assert [[b["type"] for b in m["content"]] for m in replayed] == [
        ["thinking", "tool_use"],
        ["thinking", "tool_use"],
    ]
    assert [m["content"][0] for m in replayed] == [SIGNED, SECOND]


def _stream_fakes(
    monkeypatch: pytest.MonkeyPatch, llm: Any, script: Callable[[Any], list[Any]]
) -> None:
    """Serve each call the parsed chunks `script` picks from its request."""

    def fake(self: Any, **kwargs: Any) -> Any:
        return iter(script(kwargs))

    async def afake(self: Any, **kwargs: Any) -> Any:
        chunks = script(kwargs)

        async def gen() -> Any:
            for chunk in chunks:
                # Yield control so two concurrent streams interleave chunk by chunk.
                await asyncio.sleep(0)
                yield chunk

        return gen()

    monkeypatch.setattr(type(llm), "completion_with_retry", fake)
    monkeypatch.setattr(type(llm), "acompletion_with_retry", afake)


def _bedrock_converse_chunks(blocks: list[tuple[list[str], str]]) -> list[Any]:
    decoder = AWSEventStreamDecoder(model="anthropic.claude-sonnet-4")
    return [decoder.converse_chunk_parser(e) for e in _bedrock_converse_events(blocks)]


@pytest.mark.parametrize("mode", ["stream", "astream"])
@pytest.mark.parametrize("kind", ["base", "router"])
def test_each_stream_loop_assembles_across_chunks(
    monkeypatch: pytest.MonkeyPatch, kind: str, mode: str
) -> None:
    """Bedrock's closing chunk carries no text, so only state held for the whole
    stream rebuilds the block: this fails if a loop builds its assembler per chunk
    or does not pass it."""
    llm = _model(kind)
    chunks = _bedrock_converse_chunks(
        [(THOUGHT, SIGNED["signature"]), (["Second."], "sig-2==")]
    )
    _stream_fakes(monkeypatch, llm, lambda kwargs: chunks)

    total = _run(llm, mode, [HumanMessage("hi")])

    assert total.additional_kwargs["thinking_blocks"] == signed_at(KIMI, SIGNED, SECOND)


@pytest.mark.parametrize("kind", ["base", "router"])
def test_concurrent_streams_keep_their_own_thinking(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Two streams on one instance, interleaved chunk by chunk, each keep their own
    blocks: this fails if the assembler outlives a call or is shared between calls."""
    llm = _model(kind)
    by_prompt = {
        "a": _bedrock_converse_chunks([(["alpha ", "one"], "sig-a==")]),
        "b": _bedrock_converse_chunks([(["beta ", "two"], "sig-b==")]),
    }
    _stream_fakes(
        monkeypatch, llm, lambda kwargs: by_prompt[kwargs["messages"][-1]["content"]]
    )

    async def collect(prompt: str) -> Any:
        total = None
        async for chunk in llm.astream([HumanMessage(prompt)]):
            total = chunk if total is None else total + chunk
        return total

    async def both() -> list[Any]:
        return list(await asyncio.gather(collect("a"), collect("b")))

    first, second = asyncio.run(both())

    assert first.additional_kwargs["thinking_blocks"] == signed_at(
        KIMI, {"type": "thinking", "thinking": "alpha one", "signature": "sig-a=="}
    )
    assert second.additional_kwargs["thinking_blocks"] == signed_at(
        KIMI, {"type": "thinking", "thinking": "beta two", "signature": "sig-b=="}
    )
