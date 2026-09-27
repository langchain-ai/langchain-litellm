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
    _attach_reasoning_items,
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
    """A fresh instance with the same endpoint and key replays the item ahead of
    the function call it led to."""
    requests = serve_http(monkeypatch, REPLY, EVENTS)
    first = _first_turn(_model(kind), mode)

    _second_turn(_model(kind), first, mode)

    assert _sent_reasoning(requests[-1]) == [("rs_1", ENCRYPTED)]
    types = [item.get("type") for item in json.loads(requests[-1].content)["input"]]
    assert types.index("reasoning") < types.index("function_call")


@pytest.mark.parametrize(
    "llm",
    [
        pytest.param(
            lambda: ChatLiteLLM(
                model="gpt-5-mini",
                custom_llm_provider="openai",
                api_key=KEY,
                use_responses_api=True,
            ),
            id="flag-with-provider",
        ),
        pytest.param(
            lambda: ChatLiteLLM(
                model=DEPLOYMENT, custom_llm_provider="openai", api_key=KEY
            ),
            id="named-with-provider",
        ),
        pytest.param(
            lambda: _router(
                {"model": DEPLOYMENT, "custom_llm_provider": "openai", "api_key": KEY}
            ),
            id="router-deployment-with-provider",
        ),
    ],
)
def test_a_named_provider_still_sends_the_item_back(
    monkeypatch: pytest.MonkeyPatch, llm: Any
) -> None:
    """litellm drops a model prefix that repeats the provider, so the origin must too."""
    requests = serve_http(monkeypatch, REPLY)
    first = _first_turn(llm())

    _second_turn(llm(), first)

    assert _sent_reasoning(requests[-1]) == [("rs_1", ENCRYPTED)]


def test_stateless_turns_carry_the_encrypted_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented setup: with store=False, only the encrypted item keeps reasoning."""
    requests = serve_http(monkeypatch, REPLY)
    stateless = {
        "extra_body": {"store": False, "include": ["reasoning.encrypted_content"]}
    }
    first = _first_turn(_base(model_kwargs=stateless))

    _second_turn(_base(model_kwargs=stateless), first)

    sent = json.loads(requests[-1].content)
    assert (sent["store"], sent["include"]) == (False, ["reasoning.encrypted_content"])
    assert _sent_reasoning(requests[-1]) == [("rs_1", ENCRYPTED)]


@pytest.mark.parametrize(
    "model_kwargs",
    [
        pytest.param({}, id="stored"),
        pytest.param({"extra_body": {"store": False}}, id="store-false"),
    ],
)
def test_an_item_without_encrypted_content_stays_behind(
    monkeypatch: pytest.MonkeyPatch, model_kwargs: dict[str, Any]
) -> None:
    """An id alone resolves only while the server keeps the response, which
    store=False and zero-retention accounts rule out, so it is never kept."""
    id_only = responses_api_reply(
        reasoning_item("rs_1", "Need the weather tool."),
        message_item("Checking."),
        CALL,
    )
    requests = serve_http(monkeypatch, id_only)
    first = _first_turn(_base(model_kwargs=model_kwargs))

    _second_turn(_base(model_kwargs=model_kwargs), first)

    assert "reasoning_items" not in first.additional_kwargs
    assert _sent_reasoning(requests[-1]) == []


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
    ("first", "second", "replayed"),
    [
        pytest.param(
            {"api_key": None, "env:OPENAI_API_KEY": KEY},
            {"api_key": None, "env:OPENAI_API_KEY": KEY},
            True,
            id="same-env-key",
        ),
        pytest.param(
            {"api_key": None, "env:OPENAI_API_KEY": KEY},
            {"api_key": None, "env:OPENAI_API_KEY": OTHER_KEY},
            False,
            id="another-env-key",
        ),
        pytest.param(
            {"api_key": None, "litellm:openai_key": KEY},
            {"api_key": None, "litellm:openai_key": OTHER_KEY},
            False,
            id="another-litellm-openai-key",
        ),
        pytest.param(
            {"api_key": None, "litellm:api_key": KEY},
            {"api_key": None, "litellm:api_key": OTHER_KEY},
            False,
            id="another-litellm-api-key",
        ),
        pytest.param(
            {"env:OPENAI_BASE_URL": "https://a.example/v1"},
            {"env:OPENAI_BASE_URL": "https://b.example/v1"},
            False,
            id="another-base-url-env",
        ),
        pytest.param(
            {"env:OPENAI_API_BASE": "https://a.example/v1"},
            {"env:OPENAI_API_BASE": "https://b.example/v1"},
            False,
            id="another-api-base-env",
        ),
        pytest.param(
            {
                "env:OPENAI_API_BASE": "https://a.example/v1",
                "env:OPENAI_BASE_URL": "https://a.example/v1",
            },
            {
                "env:OPENAI_API_BASE": "https://a.example/v1",
                "env:OPENAI_BASE_URL": "https://b.example/v1",
            },
            False,
            id="another-base-url-env-beside-api-base",
        ),
        pytest.param(
            {"api_base": "https://a:pw@gateway.example/v1"},
            {"api_base": "https://b:pw@gateway.example/v1"},
            False,
            id="another-user-in-the-base",
        ),
        pytest.param(
            {"extra_headers": {"OpenAI-Organization": "org-a"}},
            {"extra_headers": {"OpenAI-Organization": "org-b"}},
            False,
            id="another-organization-header",
        ),
        pytest.param(
            {"extra_headers": {"OpenAI-Project": "proj-a"}},
            {"extra_headers": {"OpenAI-Project": "proj-b"}},
            False,
            id="another-project-header",
        ),
    ],
)
def test_the_origin_follows_every_credential_litellm_may_read(
    monkeypatch: pytest.MonkeyPatch,
    first: dict[str, Any],
    second: dict[str, Any],
    replayed: bool,
) -> None:
    """litellm reads keys and bases from several places, in an order that differs by
    provider, so a change to any of them replays nothing."""
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    requests = serve_http(monkeypatch, REPLY)

    def model(setup: dict[str, Any]) -> ChatLiteLLM:
        kwargs: dict[str, Any] = {"model": MODEL, "use_responses_api": True}
        for name, value in setup.items():
            where, _, attr = name.partition(":")
            if where == "env":
                monkeypatch.setenv(attr, value)
            elif where == "litellm":
                monkeypatch.setattr(litellm, attr, value)
            else:
                kwargs[name] = value
        kwargs.setdefault("api_key", KEY)
        return ChatLiteLLM(**kwargs)

    reply = _first_turn(model(first))
    _second_turn(model(second), reply)

    assert _sent_reasoning(requests[-1]) == ([("rs_1", ENCRYPTED)] if replayed else [])


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


@pytest.mark.parametrize(
    ("mode", "kept", "sent"),
    [
        pytest.param("invoke", ["rs_2"], [("rs_2", "gAAAAB-2")], id="invoke"),
        pytest.param("stream", ["rs_1", "rs_2"], [], id="stream"),
    ],
)
def test_a_turn_with_several_items_goes_back_without_them(
    monkeypatch: pytest.MonkeyPatch, mode: str, kept: list[str], sent: list[Any]
) -> None:
    """litellm puts a turn's items ahead of its text and tool calls, which moves all
    but a lone item. Unstreamed, litellm itself keeps only the last one."""
    first_item = reasoning_item("rs_1", "Search first.", "gAAAAB-1")
    second_item = reasoning_item("rs_2", "Now the weather.", "gAAAAB-2")
    reply = responses_api_reply(first_item, second_item, message_item("On it."), CALL)
    events = responses_api_events(
        {
            "type": "response.created",
            "response": {**reply, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "output_index": 2,
            "item_id": "msg_1",
            "content_index": 0,
            "delta": "On it.",
            "logprobs": [],
        },
        {"type": "response.completed", "response": reply},
    )
    requests = serve_http(monkeypatch, reply, events)
    first = _first_turn(_base(), mode)

    _second_turn(_base(), first, mode)

    assert [item["id"] for item in first.additional_kwargs["reasoning_items"]] == kept
    assert _sent_reasoning(requests[-1]) == sent


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


def test_an_unmarked_item_never_reaches_a_request_without_an_issuer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no single issuer, an item without a mark must not count as matching."""
    requests = serve_http(monkeypatch, REPLY)
    first = AIMessage(
        content="Checking.",
        additional_kwargs={"reasoning_items": [dict(REASONING)]},
        tool_calls=[
            {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_Qx7wP2"}
        ],
    )
    split = _router(
        {"model": DEPLOYMENT, "api_key": KEY},
        {"model": DEPLOYMENT, "api_key": OTHER_KEY},
    )

    _second_turn(split, first)

    assert _sent_reasoning(requests[-1]) == []


def test_litellm_gets_each_item_as_it_returned_it() -> None:
    """The mark is the connector's own; litellm receives only the keys it gave."""
    origin = "a" * 16
    messages = [
        HumanMessage("weather?"),
        AIMessage(
            content="Checking.",
            additional_kwargs={"reasoning_items": [{**REASONING, _ORIGIN: origin}]},
        ),
    ]
    message_dicts = [_convert_message_to_dict(m) for m in messages]

    _attach_reasoning_items(messages, message_dicts, origin)

    assert message_dicts[1]["reasoning_items"] == [REASONING]


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
