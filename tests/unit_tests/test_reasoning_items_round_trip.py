"""Responses API reasoning items survive a tool-loop continuation.

litellm's Responses API bridge returns each reasoning item, with its id and
encrypted content, on the assistant message as ``reasoning_items``, and rebuilds
reasoning input on the next turn only from that key. OpenAI decrypts an item only
for the organization and model that issued it, so the connector hands items back
only to the endpoint and key that produced them. The wire tests patch only the
transport: the connector, litellm's bridge, its SSE parser and its clients all run.
"""

import asyncio
import json
import re
from typing import Any

import litellm
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_litellm.chat_models import ChatLiteLLM, ChatLiteLLMRouter
from langchain_litellm.chat_models.litellm import (
    _ORIGIN,
    _convert_message_to_dict,
    _rejoin_split_reply,
)
from tests.utils import (
    chat_completion_reply,
    function_call_item,
    message_item,
    reasoning_item,
    responses_api_events,
    responses_api_reply,
    serve_http,
)

KEY = "sk-test-org-a"
OTHER_KEY = "sk-test-org-b"
MODEL = "openai/gpt-5-mini"
DEPLOYMENT = "openai/responses/gpt-5-mini"
ENCRYPTED = "gAAAAB-encrypted-reasoning"
REASONING = reasoning_item("rs_1", "Need the weather tool.", ENCRYPTED)
CALL = function_call_item("call_Qx7wP2", '{"city": "Paris"}')
REPLY = responses_api_reply(REASONING, message_item("Checking."), CALL)
EVENTS = responses_api_events(
    {
        "type": "response.created",
        "response": {**REPLY, "status": "in_progress", "output": []},
    },
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"type": "reasoning", "id": "rs_1", "summary": []},
    },
    {
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "delta": "Need the weather tool.",
    },
    {"type": "response.output_item.done", "output_index": 0, "item": REASONING},
    {
        "type": "response.output_text.delta",
        "output_index": 1,
        "item_id": "msg_1",
        "content_index": 0,
        "delta": "Checking.",
        "logprobs": [],
    },
    {
        "type": "response.output_item.added",
        "output_index": 2,
        "item": {**CALL, "arguments": ""},
    },
    {
        "type": "response.function_call_arguments.delta",
        "output_index": 2,
        "item_id": "fc_1",
        "delta": CALL["arguments"],
    },
    {"type": "response.output_item.done", "output_index": 2, "item": CALL},
    {"type": "response.completed", "response": REPLY},
)
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
MODES = ["invoke", "ainvoke", "stream", "astream"]


def _base(**kwargs: Any) -> ChatLiteLLM:
    return ChatLiteLLM(
        **{"model": MODEL, "api_key": KEY, "use_responses_api": True, **kwargs}
    )


def _router(*deployments: dict[str, Any], **router: Any) -> ChatLiteLLMRouter:
    params = deployments or ({"model": DEPLOYMENT, "api_key": KEY},)
    model_list = [{"model_name": "gpt", "litellm_params": p} for p in params]
    return ChatLiteLLMRouter(
        router=litellm.Router(model_list=model_list, **router), model_name="gpt"
    )


def _model(kind: str) -> ChatLiteLLM:
    return _base() if kind == "base" else _router()


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


def _first_turn(llm: Any, mode: str = "invoke") -> Any:
    return _run(llm.bind_tools([WEATHER_TOOL]), mode, [HumanMessage("weather?")])


def _second_turn(llm: Any, first: Any, mode: str = "invoke") -> None:
    history = [
        HumanMessage("weather?"),
        first,
        ToolMessage("Sunny", tool_call_id="call_Qx7wP2"),
    ]
    _run(llm.bind_tools([WEATHER_TOOL]), mode, history)


def _sent_reasoning(request: Any) -> list[tuple[Any, ...]]:
    """The reasoning items a Responses API request carries, in order."""
    return [
        (item["id"], item.get("encrypted_content"))
        for item in json.loads(request.content)["input"]
        if isinstance(item, dict) and item.get("type") == "reasoning"
    ]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["base", "router"])
def test_a_reply_keeps_its_reasoning_items(
    monkeypatch: pytest.MonkeyPatch, kind: str, mode: str
) -> None:
    """Each item keeps its id and encrypted content, marked with where it came from."""
    serve_http(monkeypatch, REPLY, EVENTS)

    message = _first_turn(_model(kind), mode)

    items = message.additional_kwargs["reasoning_items"]
    assert [(item["id"], item["encrypted_content"]) for item in items] == [
        ("rs_1", ENCRYPTED)
    ]
    assert all(re.fullmatch("[0-9a-f]{16}", item[_ORIGIN]) for item in items)
    assert message.additional_kwargs["reasoning_content"] == "Need the weather tool."
    assert message.tool_calls[0]["id"] == "call_Qx7wP2"


@pytest.mark.parametrize("kind", ["base", "router"])
def test_a_kept_item_never_holds_the_api_key(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The origin digests the key; a stored history must not carry it."""
    serve_http(monkeypatch, REPLY, EVENTS)

    message = _first_turn(_model(kind))

    assert KEY not in json.dumps(message.additional_kwargs, default=str)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["base", "router"])
def test_a_tool_loop_sends_the_reasoning_item_back(
    monkeypatch: pytest.MonkeyPatch, kind: str, mode: str
) -> None:
    """A fresh instance with the same endpoint and key replays the item, unmarked,
    ahead of the function call it led to."""
    requests = serve_http(monkeypatch, REPLY, EVENTS)
    first = _first_turn(_model(kind), mode)

    _second_turn(_model(kind), first, mode)

    assert _sent_reasoning(requests[-1]) == [("rs_1", ENCRYPTED)]
    types = [item.get("type") for item in json.loads(requests[-1].content)["input"]]
    assert types.index("reasoning") < types.index("function_call")
    assert _ORIGIN not in requests[-1].content.decode()


def test_an_item_the_server_stored_goes_back_by_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With store=True there is no encrypted content; the id alone resolves it."""
    stored = responses_api_reply(
        reasoning_item("rs_1", "Need the weather tool."),
        message_item("Checking."),
        CALL,
    )
    requests = serve_http(monkeypatch, stored)
    first = _first_turn(_base())

    _second_turn(_base(), first)

    assert _sent_reasoning(requests[-1]) == [("rs_1", None)]


@pytest.mark.parametrize(
    "second",
    [
        pytest.param({"api_key": OTHER_KEY}, id="another-key"),
        pytest.param({"model": "openai/gpt-5"}, id="another-model"),
        pytest.param({"api_base": "https://gateway.example/v1"}, id="another-base"),
        pytest.param({"organization": "org-other"}, id="another-organization"),
    ],
)
def test_an_item_never_goes_to_another_issuer(
    monkeypatch: pytest.MonkeyPatch, second: dict[str, Any]
) -> None:
    """OpenAI decrypts an item only for the org and model that issued it; elsewhere
    it answers 400 invalid_encrypted_content, so the item stays behind."""
    requests = serve_http(monkeypatch, REPLY)
    first = _first_turn(_base())

    _second_turn(_base(**second), first)

    assert _sent_reasoning(requests[-1]) == []


@pytest.mark.parametrize(
    ("second_key", "replayed"),
    [
        pytest.param(KEY, [("rs_1", ENCRYPTED)], id="same-env-key"),
        pytest.param(OTHER_KEY, [], id="another-env-key"),
    ],
)
def test_the_origin_follows_a_key_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, second_key: str, replayed: list[Any]
) -> None:
    """litellm falls back to OPENAI_API_KEY, so the origin must too."""
    requests = serve_http(monkeypatch, REPLY)
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    first = _first_turn(ChatLiteLLM(model=MODEL, use_responses_api=True))

    monkeypatch.setenv("OPENAI_API_KEY", second_key)
    _second_turn(ChatLiteLLM(model=MODEL, use_responses_api=True), first)

    assert _sent_reasoning(requests[-1]) == replayed


def test_a_chat_completions_call_never_carries_reasoning_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the bridge turns the key into a reasoning item; Chat Completions has no
    such field, so the same model without the flag gets none."""
    serve_http(monkeypatch, REPLY)
    first = _first_turn(_base())
    requests = serve_http(monkeypatch, chat_completion_reply("Sunny."))

    _second_turn(ChatLiteLLM(model=MODEL, api_key=KEY), first)

    assert "reasoning_items" not in requests[-1].content.decode()
    assert ENCRYPTED not in requests[-1].content.decode()


@pytest.mark.parametrize(
    "origin",
    [
        pytest.param(None, id="unmarked"),
        pytest.param("0000000000000000", id="foreign"),
    ],
)
def test_an_item_from_elsewhere_stays_behind(
    monkeypatch: pytest.MonkeyPatch, origin: str | None
) -> None:
    """History built by hand or by another app carries no origin this endpoint issued."""
    requests = serve_http(monkeypatch, REPLY)
    item = {**REASONING, _ORIGIN: origin} if origin else dict(REASONING)
    first = AIMessage(
        content="Checking.",
        additional_kwargs={"reasoning_items": [item]},
        tool_calls=[
            {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_Qx7wP2"}
        ],
    )

    _second_turn(_base(), first)

    assert _sent_reasoning(requests[-1]) == []


def test_a_turn_goes_back_whole_or_not_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Items of one turn followed one another; sending part of them edits the turn."""
    requests = serve_http(monkeypatch, REPLY)
    first = _first_turn(_base())
    kept = first.additional_kwargs["reasoning_items"]
    first.additional_kwargs["reasoning_items"] = [
        *kept,
        {**reasoning_item("rs_2", "Foreign.", "gAAAAB-other"), _ORIGIN: "0" * 16},
    ]

    _second_turn(_base(), first)

    assert _sent_reasoning(requests[-1]) == []


def test_a_call_with_fallbacks_keeps_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fallback may answer from another model, whose items no issuer accepts."""
    serve_http(monkeypatch, REPLY)

    message = _first_turn(_base(model_kwargs={"fallbacks": ["openai/gpt-5"]}))

    assert "reasoning_items" not in message.additional_kwargs


@pytest.mark.parametrize(
    "llm",
    [
        pytest.param(
            lambda: _router(
                {"model": DEPLOYMENT, "api_key": KEY},
                {"model": DEPLOYMENT, "api_key": OTHER_KEY},
            ),
            id="deployments-with-different-keys",
        ),
        pytest.param(
            lambda: _router(
                {"model": DEPLOYMENT, "api_key": KEY},
                {
                    "model": "azure/responses/gpt-5-mini",
                    "api_key": KEY,
                    "api_base": "https://east.openai.azure.com",
                },
            ),
            id="deployments-at-different-endpoints",
        ),
        pytest.param(
            lambda: _router(fallbacks=[{"gpt": ["gpt-backup"]}]),
            id="router-fallbacks",
        ),
    ],
)
def test_a_router_that_may_answer_elsewhere_keeps_nothing(
    monkeypatch: pytest.MonkeyPatch, llm: Any
) -> None:
    """The Router picks a deployment per call, so it keeps an item only when every
    deployment it may pick would accept it back."""
    serve_http(monkeypatch, REPLY)

    message = _first_turn(llm())

    assert "reasoning_items" not in message.additional_kwargs


def test_a_message_dict_never_carries_reasoning_items() -> None:
    """Only the checked replay path sends them, never the plain conversion."""
    message = AIMessage(
        content="Checking.",
        additional_kwargs={"reasoning_items": [{**REASONING, _ORIGIN: "a" * 16}]},
    )

    assert "reasoning_items" not in _convert_message_to_dict(message)


def test_a_split_reply_rejoins_with_its_reasoning_items() -> None:
    """The bridge puts the items on the text choice and the tool calls on another."""
    choices = [
        {"message": {"role": "assistant", "content": "A", "reasoning_items": [1]}},
        {"message": {"role": "assistant", "content": "B", "reasoning_items": [2]}},
        {"message": {"role": "assistant", "tool_calls": [{"id": "c"}]}},
    ]

    (rejoined,) = _rejoin_split_reply(choices, None)

    assert rejoined["message"]["reasoning_items"] == [1, 2]
    assert rejoined["message"]["content"] == "AB"


def test_a_split_reply_without_items_gains_no_key() -> None:
    choices = [
        {"message": {"role": "assistant", "content": "A"}},
        {"message": {"role": "assistant", "tool_calls": [{"id": "c"}]}},
    ]

    (rejoined,) = _rejoin_split_reply(choices, None)

    assert "reasoning_items" not in rejoined["message"]
