"""use_responses_api on ChatLiteLLMRouter checks the deployments a call can reach.

The Router picks a deployment per call, so the flag renames nothing: the call goes
out unchanged when litellm sends every deployment it can reach to a Responses API,
and raises before any request otherwise.
"""

import copy
import json
from typing import Any
from unittest.mock import patch

import httpx
import litellm
import pytest
from langchain_core.messages import BaseMessage

from langchain_litellm import ChatLiteLLMRouter
from tests.utils import (
    message_item,
    responses_api_events,
    responses_api_reply,
    serve_http,
)

REPLY = responses_api_reply(message_item("ok"))
EVENTS = responses_api_events(
    {"type": "response.created", "response": {**REPLY, "status": "in_progress"}},
    {
        "type": "response.output_text.delta",
        "output_index": 0,
        "item_id": "msg_1",
        "content_index": 0,
        "delta": "ok",
        "logprobs": [],
    },
    {"type": "response.completed", "response": REPLY},
)
AZURE = {"api_base": "https://x.openai.azure.com", "api_version": "2025-04-01-preview"}

# Where the flag can be set, as on ChatLiteLLM: (constructor kwargs, call kwargs).
FLAG_SOURCES = [
    pytest.param({"use_responses_api": True}, {}, id="constructor"),
    pytest.param({"model_kwargs": {"use_responses_api": True}}, {}, id="model_kwargs"),
    pytest.param({}, {"use_responses_api": True}, id="call"),
]
METHODS = ["invoke", "ainvoke", "stream", "astream"]


def _deployment(model: str | dict[str, Any], group: str = "g") -> dict[str, Any]:
    params = {"model": model} if isinstance(model, str) else model
    return {"model_name": group, "litellm_params": {"api_key": "k", **params}}


def _router(*models: str | dict[str, Any], **settings: Any) -> litellm.Router:
    return litellm.Router(model_list=[_deployment(m) for m in models], **settings)


async def _reply(llm: ChatLiteLLMRouter, method: str, **call: Any) -> BaseMessage:
    if method == "invoke":
        return llm.invoke("hi", **call)
    if method == "ainvoke":
        return await llm.ainvoke("hi", **call)
    if method == "stream":
        chunks = list(llm.stream("hi", **call))
    else:
        chunks = [chunk async for chunk in llm.astream("hi", **call)]
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    return merged


async def _refused(llm: ChatLiteLLMRouter, method: str, **call: Any) -> str:
    """The ValueError ``method`` raises, checked to come before any request."""
    router = llm.router
    with (
        patch.object(router, "completion") as completion,
        patch.object(router, "acompletion") as acompletion,
        pytest.raises(ValueError) as refusal,
    ):
        await _reply(llm, method, **call)
    completion.assert_not_called()
    acompletion.assert_not_called()
    return str(refusal.value)


def _urls(requests: list[httpx.Request]) -> list[str]:
    return [str(request.url) for request in requests]


@pytest.mark.parametrize(("config", "call"), FLAG_SOURCES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.asyncio
async def test_a_group_of_responses_deployments_goes_out_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    call: dict[str, Any],
    method: str,
) -> None:
    """Each entry point, wherever the flag is set, reaches the Responses API."""
    requests = serve_http(monkeypatch, REPLY, EVENTS)
    llm = ChatLiteLLMRouter(router=_router("openai/responses/gpt-4o-mini"), **config)

    message = await _reply(llm, method, **call)

    assert _urls(requests) == ["https://api.openai.com/v1/responses"]
    assert "use_responses_api" not in json.loads(requests[0].content)
    assert message.content == "ok"


def test_the_call_keeps_the_group_name() -> None:
    """The Router resolves the group itself, so nothing is renamed."""
    llm = ChatLiteLLMRouter(
        router=_router("openai/responses/gpt-4o-mini"), use_responses_api=True
    )

    with (
        patch.object(llm.router, "completion", side_effect=RuntimeError) as sent,
        pytest.raises(RuntimeError),
    ):
        llm.invoke("hi")

    assert sent.call_args.kwargs["model"] == "g"
    assert "use_responses_api" not in sent.call_args.kwargs


@pytest.mark.parametrize(
    "deployment",
    [
        pytest.param("openai/responses/gpt-4o-mini", id="openai"),
        pytest.param({"model": "azure/responses/my-dep", **AZURE}, id="azure"),
        pytest.param(
            {"model": "responses/gpt-4o-mini", "custom_llm_provider": "openai"},
            id="provider-set-on-the-deployment",
        ),
        pytest.param("openai/gpt-5-pro", id="bridged-by-litellm-on-its-own"),
    ],
)
def test_a_deployment_litellm_sends_to_a_responses_api_passes(
    monkeypatch: pytest.MonkeyPatch, deployment: str | dict[str, Any]
) -> None:
    requests = serve_http(monkeypatch, REPLY)
    llm = ChatLiteLLMRouter(router=_router(deployment), use_responses_api=True)

    llm.invoke("hi")

    assert [request.url.path for request in requests] in (
        ["/v1/responses"],
        ["/openai/responses"],
        ["/openai/v1/responses"],
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.asyncio
async def test_each_deployment_outside_a_responses_api_is_named(method: str) -> None:
    """One such deployment is enough: the Router may pick it on any call."""
    llm = ChatLiteLLMRouter(
        router=_router(
            "openai/responses/gpt-4o-mini",
            "openai/gpt-4o-mini",
            {"model": "azure/my-dep", **AZURE},
            "anthropic/responses/claude-sonnet-4-5",
        ),
        use_responses_api=True,
    )

    refusal = await _refused(llm, method)

    assert "'openai/gpt-4o-mini'" in refusal
    assert "'azure/my-dep'" in refusal
    assert "'anthropic/responses/claude-sonnet-4-5'" in refusal
    assert "'openai/responses/gpt-4o-mini'" not in refusal
    assert "<provider>/responses/<model>" in refusal


def test_only_the_group_the_call_reaches_is_checked() -> None:
    router = litellm.Router(
        model_list=[
            _deployment("openai/responses/gpt-4o-mini", group="g"),
            _deployment("openai/gpt-4o-mini", group="chat"),
        ]
    )
    llm = ChatLiteLLMRouter(router=router, model_name="g", use_responses_api=True)

    with patch.object(router, "completion", side_effect=RuntimeError) as sent:
        with pytest.raises(RuntimeError):
            llm.invoke("hi")
        with pytest.raises(ValueError, match="'openai/gpt-4o-mini'"):
            llm.invoke("hi", model="chat")

    assert [call.kwargs["model"] for call in sent.call_args_list] == ["g"]


@pytest.mark.asyncio
async def test_a_group_no_model_list_entry_names_is_refused() -> None:
    """Only a wildcard deployment serves it, and which one is not known here."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "k"},
            }
        ]
    )
    llm = ChatLiteLLMRouter(
        router=router,
        model_name="openai/responses/gpt-4o-mini",
        use_responses_api=True,
    )

    refusal = await _refused(llm, "invoke")

    assert "'openai/responses/gpt-4o-mini'" in refusal


TEAM = {"metadata": {"user_api_key_team_id": "t"}}
FALLBACK = {"fallbacks": [{"g": ["h"]}]}


@pytest.mark.parametrize(
    ("settings", "config", "call", "reason"),
    [
        pytest.param(FALLBACK, {}, {}, "fallbacks", id="fallbacks"),
        pytest.param(
            {"default_fallbacks": ["h"]}, {}, {}, "fallbacks", id="default_fallbacks"
        ),
        pytest.param(
            {"context_window_fallbacks": [{"g": ["h"]}]},
            {},
            {},
            "context_window_fallbacks",
            id="context_window_fallbacks",
        ),
        pytest.param(
            {"content_policy_fallbacks": [{"g": ["h"]}]},
            {},
            {},
            "content_policy_fallbacks",
            id="content_policy_fallbacks",
        ),
        pytest.param(
            {"default_litellm_params": FALLBACK},
            {},
            {},
            "fallbacks",
            id="fallbacks-in-router-defaults",
        ),
        pytest.param({}, {}, FALLBACK, "fallbacks", id="fallbacks-per-call"),
        pytest.param(
            {}, {}, {"fallbacks": ["h"]}, "fallbacks", id="fallback-groups-per-call"
        ),
        pytest.param(
            {},
            {},
            {"context_window_fallbacks": [{"g": ["h"]}]},
            "context_window_fallbacks",
            id="context_window_fallbacks-per-call",
        ),
        pytest.param(
            {},
            {"model_kwargs": FALLBACK},
            {},
            "fallbacks",
            id="fallbacks-in-model_kwargs",
        ),
        pytest.param(
            {"model_group_alias": {"g": "h"}},
            {},
            {},
            "model_group_alias",
            id="model_group_alias",
        ),
        pytest.param(
            {"model_group_alias": {"g": {"model": "h", "hidden": True}}},
            {},
            {},
            "model_group_alias",
            id="hidden-model_group_alias",
        ),
        pytest.param({}, {"model_kwargs": TEAM}, {}, "team", id="team-caller"),
        pytest.param(
            {"default_litellm_params": TEAM}, {}, {}, "team", id="team-default"
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_call_that_may_reach_another_group_is_refused(
    settings: dict[str, Any],
    config: dict[str, Any],
    call: dict[str, Any],
    reason: str,
) -> None:
    """Those deployments are not the ones checked, so nothing passes.

    Every deployment here would pass, so only the reach can refuse the call.
    """
    # The Router writes into the defaults it is given, so no case shares a dict.
    settings, config, call = copy.deepcopy((settings, config, call))
    router = litellm.Router(
        model_list=[
            _deployment("openai/responses/gpt-4o-mini", group="g"),
            _deployment("openai/responses/gpt-4o-mini", group="h"),
        ],
        **settings,
    )
    llm = ChatLiteLLMRouter(
        router=router, model_name="g", use_responses_api=True, **config
    )

    assert reason in await _refused(llm, "invoke", **call)


@pytest.mark.asyncio
async def test_litellm_s_own_fallbacks_refuse_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """litellm falls back to these inside the Router's call, past every group."""
    monkeypatch.setattr(litellm, "model_fallbacks", ["openai/gpt-4o-mini"])
    llm = ChatLiteLLMRouter(
        router=_router("openai/responses/gpt-4o-mini"), use_responses_api=True
    )

    assert "litellm.model_fallbacks" in await _refused(llm, "invoke")


def test_a_wildcard_beside_the_group_s_own_entry_is_never_picked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Router serves a group from its exact entries whenever it has one."""
    requests = serve_http(monkeypatch, REPLY)
    router = litellm.Router(
        model_list=[
            _deployment("openai/responses/gpt-4o-mini"),
            {
                "model_name": "*",
                "litellm_params": {"model": "openai/*", "api_key": "k"},
            },
        ]
    )
    llm = ChatLiteLLMRouter(router=router, model_name="g", use_responses_api=True)

    llm.invoke("hi")

    assert _urls(requests) == ["https://api.openai.com/v1/responses"]


def test_litellm_s_switch_for_every_openai_call_is_followed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """litellm is asked, not predicted, so its process-wide route counts too."""
    monkeypatch.setattr(litellm, "route_all_chat_openai_to_responses", True)
    requests = serve_http(monkeypatch, REPLY)
    llm = ChatLiteLLMRouter(
        router=_router("openai/gpt-4o-mini"), use_responses_api=True
    )

    llm.invoke("hi")

    assert _urls(requests) == ["https://api.openai.com/v1/responses"]


@pytest.mark.asyncio
async def test_a_deployment_that_asks_for_chat_completions_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """litellm sends it to Chat Completions even under that switch."""
    monkeypatch.setattr(litellm, "route_all_chat_openai_to_responses", True)
    llm = ChatLiteLLMRouter(
        router=_router("openai/chat_completions/gpt-5-pro"), use_responses_api=True
    )

    assert "'openai/chat_completions/gpt-5-pro'" in await _refused(llm, "invoke")


@pytest.mark.asyncio
async def test_a_deployment_placed_by_its_api_base_is_named() -> None:
    """litellm reads Groq off the endpoint, and Groq has no Responses API here."""
    deployment = {
        "model": "responses/llama-3.3-70b",
        "api_base": "https://api.groq.com/openai/v1",
    }
    try:
        router = _router(deployment)
    except litellm.BadRequestError:
        pytest.skip("this litellm's Router does not place a deployment by api_base")
    llm = ChatLiteLLMRouter(router=router, use_responses_api=True)

    assert "'responses/llama-3.3-70b'" in await _refused(llm, "invoke")
