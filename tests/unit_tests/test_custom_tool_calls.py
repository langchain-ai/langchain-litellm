"""A call to a custom (freeform) tool reaches the caller and goes back as one.

OpenAI's custom tools take one freeform string instead of JSON arguments, and
litellm returns a call to one as ``{"type": "custom", "custom": {"name", "input"}}``
with no ``function``. The connector hands it over as a tool call whose args are
``{"__arg1": input}``, the argument a single-input LangChain tool receives, and
sends it back in the custom shape. The wire tests patch only the transport.
"""

import json
from typing import Any

import litellm
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai.types.chat import ChatCompletion

from langchain_litellm.chat_models import ChatLiteLLM, ChatLiteLLMRouter
from tests.utils import responses_api_reply, serve_http, whole_reply

# Freeform input carries what JSON must escape: a quote, a backslash, a newline and
# non-ASCII. The stream splits it right after the backslash.
CODE = 'print("tab\\there")\n# ✓'
CODE_PARTS = ['print("tab\\', 'there")\n', "# ✓"]

FUNCTION_CALL = {
    "id": "call_function",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
}
CUSTOM_CALL = {
    "id": "call_custom",
    "type": "custom",
    "custom": {"name": "run_python", "input": CODE},
}

REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [FUNCTION_CALL, CUSTOM_CALL],
            },
            "finish_reason": "tool_calls",
        }
    ],
}
ChatCompletion.model_validate(REPLY)

_CHUNK = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "gpt-5",
}


def _event(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        **_CHUNK,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# The openai SDK types no custom delta, so unlike REPLY these go unvalidated. The
# custom call's first delta names it and carries no input yet.
EVENTS = [
    _event(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_function",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ""},
                }
            ],
        }
    ),
    _event(
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": "Paris"}'}}]}
    ),
    _event(
        {
            "tool_calls": [
                {
                    "index": 1,
                    "id": "call_custom",
                    "type": "custom",
                    "custom": {"name": "run_python"},
                }
            ]
        }
    ),
    *(
        _event({"tool_calls": [{"index": 1, "custom": {"input": part}}]})
        for part in CODE_PARTS
    ),
    _event({}, "tool_calls"),
]

MODES = ["invoke", "ainvoke", "stream", "astream"]


def _model(kind: str) -> ChatLiteLLM:
    if kind == "base":
        return ChatLiteLLM(model="openai/gpt-5", api_key="k")
    deployment = {"model": "openai/gpt-5", "api_key": "k"}
    router = litellm.Router(
        model_list=[{"model_name": "g", "litellm_params": deployment}]
    )
    return ChatLiteLLMRouter(router=router, model_name="g")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["base", "router"])
@pytest.mark.asyncio
async def test_a_custom_tool_call_reaches_tool_calls(
    monkeypatch: pytest.MonkeyPatch, kind: str, mode: str
) -> None:
    """Its input arrives whole as ``__arg1``, beside a function call in the reply."""
    serve_http(monkeypatch, REPLY, EVENTS)

    message = await whole_reply(_model(kind), mode)

    assert isinstance(message, AIMessage)
    assert [(c["name"], c["args"], c["id"]) for c in message.tool_calls] == [
        ("get_weather", {"city": "Paris"}, "call_function"),
        ("run_python", {"__arg1": CODE}, "call_custom"),
    ]
    assert message.invalid_tool_calls == []


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["base", "router"])
@pytest.mark.asyncio
async def test_a_custom_tool_call_goes_back_as_a_custom_call(
    monkeypatch: pytest.MonkeyPatch, kind: str, mode: str
) -> None:
    """The next turn must keep every call its tool messages answer, each in the
    shape of the tool it called."""
    requests = serve_http(monkeypatch, REPLY, EVENTS)
    llm = _model(kind)

    llm.invoke(
        [
            HumanMessage("hi"),
            await whole_reply(llm, mode),
            ToolMessage("Sunny", tool_call_id="call_function"),
            ToolMessage("tab\\there", tool_call_id="call_custom"),
        ]
    )

    sent = json.loads(requests[-1].content)["messages"][1]
    assert sent["tool_calls"] == [
        {
            "type": "function",
            "id": "call_function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        },
        {
            "type": "custom",
            "id": "call_custom",
            "custom": {"name": "run_python", "input": CODE},
        },
    ]


def test_a_bridged_custom_call_arrives_as_a_function_call_until_litellm_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """litellm's Responses API bridge returns a custom_tool_call as a function call
    with the raw input as its arguments; this fails once it keeps the custom shape."""
    serve_http(
        monkeypatch,
        responses_api_reply(
            {
                "type": "custom_tool_call",
                "call_id": "call_custom",
                "name": "run_python",
                "input": CODE,
            }
        ),
    )

    message = ChatLiteLLM(
        model="openai/gpt-5", api_key="k", use_responses_api=True
    ).invoke("Run it.")

    assert message.tool_calls == []
    assert [(c["name"], c["args"], c["id"]) for c in message.invalid_tool_calls] == [
        ("run_python", CODE, "call_custom")
    ]


@pytest.mark.parametrize(
    ("args", "keys"),
    [
        pytest.param({"code": CODE}, "['code']", id="input-moved"),
        pytest.param(
            {"__arg1": CODE, "timeout": 5}, "['__arg1', 'timeout']", id="key-added"
        ),
    ],
)
def test_a_custom_call_edited_off_its_one_input_is_refused(
    monkeypatch: pytest.MonkeyPatch, args: dict[str, Any], keys: str
) -> None:
    """A human-in-the-loop edit can change the args, but a custom tool takes one
    string: any other key would be dropped on the way back."""
    serve_http(monkeypatch, REPLY, EVENTS)
    llm = _model("base")
    reply = llm.invoke("hi")
    weather, run = reply.tool_calls
    reply.tool_calls = [weather, {**run, "args": args}]

    with pytest.raises(ValueError) as refused:
        llm.invoke([HumanMessage("hi"), reply])

    assert str(refused.value) == (
        "Tool call 'call_custom' calls the custom tool 'run_python', which takes one "
        f"string, so its args must be {{'__arg1': <input>}}; got the keys {keys}."
    )
