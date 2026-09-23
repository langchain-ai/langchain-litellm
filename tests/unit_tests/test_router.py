"""Test router chat model integration."""

from typing import Any, Callable, Dict, List
from unittest.mock import patch

import litellm
import pytest
from langchain_core.messages import AIMessage

from langchain_litellm._version import __version__
from langchain_litellm.chat_models import ChatLiteLLMRouter
from langchain_litellm.chat_models.litellm_router import _deployment_metadata
from tests.utils import make_router


def _completion_double(seen: List[Dict[str, Any]]) -> Callable[..., Any]:
    """Record the outbound kwargs and answer in the shape `stream` asked for.

    A double that returns one canned value either way hands a streaming request a
    mapping, so a wrong `stream` flag reads as success instead of the crash it is.
    """

    def _completion(**kwargs: Any) -> Any:
        seen.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [
                    {
                        "choices": [{"delta": {"role": "assistant", "content": "ok"}}],
                        "usage": None,
                    }
                ]
            )
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    return _completion


def _acompletion_double(seen: List[Dict[str, Any]]) -> Callable[..., Any]:
    """Async twin of `_completion_double`."""

    async def _acompletion(**kwargs: Any) -> Any:
        sync = _completion_double(seen)(**kwargs)

        if not kwargs.get("stream"):
            return sync

        async def _aiter() -> Any:
            for chunk in sync:
                yield chunk

        return _aiter()

    return _acompletion


def _rate_limit_error() -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message="rate limited", llm_provider="openai", model="gpt-4"
    )


_ROUTER_OK = {
    "choices": [
        {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
}


def _router_usage() -> dict:
    from litellm.utils import Usage

    return {
        **_ROUTER_OK,
        "usage": Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    }


def test_router_forwards_explicitly_configured_connector_params() -> None:
    """A connector-level endpoint is the caller's choice and must reach litellm.

    #200/#203 added the `base_url` alias for exactly this, so dropping it would make
    a router built with `base_url=...` silently ignore it. Only an INFERRED
    credential is withheld, which `_resolve_api_key` handles.
    """
    llm = ChatLiteLLMRouter(
        router=make_router(),
        base_url="https://proxy.internal/v1",  # type: ignore[call-arg]
        organization="org-ACME",
        extra_headers={"X-Team": "platform"},
    )

    with patch.object(
        llm.router, "completion", return_value=_router_usage()
    ) as mock_completion:
        llm.invoke("hi")

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["api_base"] == "https://proxy.internal/v1"
    assert kwargs["organization"] == "org-ACME"
    assert kwargs["extra_headers"] == {"X-Team": "platform"}


def test_router_forwards_a_per_call_api_base() -> None:
    """A caller who wants one destination for this call still gets it."""
    llm = ChatLiteLLMRouter(router=make_router(), api_base="https://connector/v1")

    with patch.object(
        llm.router, "completion", return_value=_router_usage()
    ) as mock_completion:
        llm.invoke("hi", api_base="https://for-this-call/v1")

    assert mock_completion.call_args.kwargs["api_base"] == "https://for-this-call/v1"


def test_router_does_not_forward_an_ambient_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A router deployment's own credential must not be overridden.

    litellm treats a passed `api_key` as a clientside credential that wins over the
    one configured on the deployment, so a key the caller never supplied must not
    travel with the call.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient-openai")
    llm = ChatLiteLLMRouter(router=make_router())

    from litellm.utils import Usage

    mock_response = {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    }
    with patch.object(
        llm.router, "completion", return_value=mock_response
    ) as mock_completion:
        llm.invoke("hi")

    assert mock_completion.call_args.kwargs.get("api_key") is None


def test_router_keeps_a_model_kwargs_api_key_across_a_redirect() -> None:
    """A key in `model_kwargs` is explicit caller config, so a redirect keeps it.

    Withholding a key is only ever right for one inferred from the destination.
    """
    llm = ChatLiteLLMRouter(router=make_router(), model_kwargs={"api_key": "sk-caller"})

    with patch.object(
        llm.router, "completion", return_value=_router_usage()
    ) as mock_completion:
        llm.invoke("hi", model="gpt-3.5-turbo")

    assert mock_completion.call_args.kwargs["api_key"] == "sk-caller"


def test_router_provider_specific_fields_in_chat_result() -> None:
    """Test that Router preserves top-level provider_specific_fields."""
    router = make_router()
    llm = ChatLiteLLMRouter(router=router)

    mock_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "provider_specific_fields": {"citations": [{"source": "vertex"}]},
    }

    result = llm._create_chat_result(mock_response, metadata={})

    assert result.llm_output is not None
    assert "provider_specific_fields" in result.llm_output
    assert (
        result.llm_output["provider_specific_fields"]["citations"][0]["source"]
        == "vertex"
    )


def test_router_create_chat_result_sets_usage_metadata() -> None:
    """Router _create_chat_result should set usage_metadata on AIMessage."""
    router = make_router()
    llm = ChatLiteLLMRouter(router=router)

    mock_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
    }

    result = llm._create_chat_result(mock_response, metadata={})
    msg = result.generations[0].message
    assert isinstance(msg, AIMessage)
    assert msg.usage_metadata is not None
    assert msg.usage_metadata["input_tokens"] == 12
    assert msg.usage_metadata["output_tokens"] == 8
    assert msg.usage_metadata["total_tokens"] == 20


def test_router_stream_options_default_to_include_usage() -> None:
    """Providers other than OpenAI only report usage when it is asked for."""
    seen: List[Dict[str, Any]] = []
    llm = ChatLiteLLMRouter(router=make_router())

    with patch.object(llm.router, "completion", side_effect=_completion_double(seen)):
        list(llm.stream("hi"))

    assert seen[0]["stream_options"] == {"include_usage": True}


def test_router_stream_honours_a_per_call_stream_options() -> None:
    """`stream_options` is caller configuration, so a per-call value is not replaced."""
    seen: List[Dict[str, Any]] = []
    llm = ChatLiteLLMRouter(router=make_router())

    with patch.object(llm.router, "completion", side_effect=_completion_double(seen)):
        list(llm.stream("hi", stream_options={"include_usage": False}))

    assert seen[0]["stream_options"] == {"include_usage": False}


@pytest.mark.asyncio
async def test_router_astream_honours_a_per_call_stream_options() -> None:
    """The async path carries the same caller configuration as the sync one."""
    seen: List[Dict[str, Any]] = []
    llm = ChatLiteLLMRouter(router=make_router())

    with patch.object(llm.router, "acompletion", side_effect=_acompletion_double(seen)):
        async for _ in llm.astream("hi", stream_options={"include_usage": False}):
            pass

    assert seen[0]["stream_options"] == {"include_usage": False}


@pytest.mark.parametrize("method", ["stream", "astream"])
@pytest.mark.asyncio
async def test_router_per_call_stream_options_none_matches_the_base_class(
    method: str,
) -> None:
    """The two classes must not disagree about what an explicit None means."""
    seen: List[Dict[str, Any]] = []
    llm = ChatLiteLLMRouter(router=make_router())

    if method == "stream":
        with patch.object(
            llm.router, "completion", side_effect=_completion_double(seen)
        ):
            list(llm.stream("hi", stream_options=None))
    else:
        with patch.object(
            llm.router, "acompletion", side_effect=_acompletion_double(seen)
        ):
            async for _ in llm.astream("hi", stream_options=None):
                pass

    assert "stream_options" in seen[0]
    assert seen[0]["stream_options"] is None


def test_router_is_claude_model_reads_the_matching_deployment() -> None:
    """A model group can fan across providers, so the first entry is not the answer."""
    import litellm

    router = litellm.Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {"model": "azure/gpt-4", "api_key": "k"},
            },
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-20241022",
                    "api_key": "k",
                },
            },
        ]
    )

    assert ChatLiteLLMRouter(router=router, model_name="mixed")._is_claude_model()


def test_router_generate_does_not_inherit_a_streaming_default() -> None:
    """This branch parses a mapping, so a caller's `stream=False` must reach litellm.

    A `streaming=True` instance otherwise sends `stream=True` and is handed an
    iterator where it expects a mapping.
    """
    seen: List[Dict[str, Any]] = []
    llm = ChatLiteLLMRouter(router=make_router(), streaming=True)

    with patch.object(llm.router, "completion", side_effect=_completion_double(seen)):
        message = llm.invoke("hi", stream=False)

    assert seen[0]["stream"] is False
    assert message.content == "ok"


@pytest.mark.asyncio
async def test_router_agenerate_does_not_inherit_a_streaming_default() -> None:
    """The async twin of the same branch."""
    seen: List[Dict[str, Any]] = []
    llm = ChatLiteLLMRouter(router=make_router(), streaming=True)

    with patch.object(llm.router, "acompletion", side_effect=_acompletion_double(seen)):
        message = await llm.ainvoke("hi", stream=False)

    assert seen[0]["stream"] is False
    assert message.content == "ok"


def test_router_metadata_versions() -> None:
    """Test that router metadata reports the correct version info."""
    router = make_router()
    llm = ChatLiteLLMRouter(router=router)
    assert llm.metadata is not None
    assert llm.metadata["lc_versions"]["langchain-litellm"] == __version__


def test_router_create_chat_result_sets_model_provider() -> None:
    """Router non-streaming path must set model_provider. Fixes #152."""
    router = make_router()
    llm = ChatLiteLLMRouter(router=router)
    mock_response = {
        "choices": [
            {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    result = llm._create_chat_result(mock_response, metadata={})
    msg = result.generations[0].message
    assert isinstance(msg, AIMessage)
    assert msg.response_metadata.get("model_provider") == "litellm"


def test_router_stream_sets_model_provider_in_response_metadata() -> None:
    """Router first streaming chunk must carry model_provider. Fixes #152."""
    from unittest.mock import patch

    router = make_router()
    llm = ChatLiteLLMRouter(router=router)
    fake_chunks = [
        {
            "choices": [{"delta": {"role": "assistant", "content": "hel"}}],
            "usage": None,
        },
        {"choices": [{"delta": {"content": "lo"}}], "usage": None},
        {
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    ]

    with patch.object(llm.router, "completion", return_value=iter(fake_chunks)):
        chunks = list(llm._stream([]))

    assert chunks[0].message.response_metadata.get("model_provider") == "litellm"
    assert chunks[1].message.response_metadata == {}


def _router_chunks_with_cost() -> List[Dict[str, Any]]:
    """The shape the router streams back: every chunk names the deployment."""
    deployment = {"model_id": "deployment-A"}
    return [
        {
            "choices": [{"delta": {"role": "assistant", "content": "hel"}}],
            "usage": None,
            "_hidden_params": deployment,
        },
        {
            "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}],
            "usage": None,
            "_hidden_params": deployment,
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "cost": 2.4e-06,
            },
            "_hidden_params": deployment,
        },
    ]


def _merge(chunks: List[Any]) -> Any:
    """Merge a stream the way a caller consuming it does."""
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk
    return merged


def test_router_streamed_cost_reaches_response_metadata() -> None:
    """The router reports a streamed cost on a trailing chunk with no content.

    That chunk arrives with an empty `choices`, so a cost read only off content
    chunks never surfaces at all.
    """
    llm = ChatLiteLLMRouter(router=make_router())

    with patch.object(
        llm.router, "completion", return_value=iter(_router_chunks_with_cost())
    ):
        chunks = [chunk.message for chunk in llm._stream([])]

    assert _merge(chunks).response_metadata["response_cost"] == 2.4e-06


@pytest.mark.asyncio
async def test_router_astreamed_cost_reaches_response_metadata() -> None:
    """The async path must surface the cost the sync path surfaces."""
    llm = ChatLiteLLMRouter(router=make_router())

    async def _acompletion(**kwargs: Any) -> Any:
        async def _aiter() -> Any:
            for chunk in _router_chunks_with_cost():
                yield chunk

        return _aiter()

    with patch.object(llm.router, "acompletion", side_effect=_acompletion):
        chunks = [chunk.message async for chunk in llm._astream([])]

    assert _merge(chunks).response_metadata["response_cost"] == 2.4e-06


def test_router_names_the_deployment_once_across_a_stream() -> None:
    """The deployment holds for the whole response, and langchain joins strings.

    Named on every chunk, the merged message reads `deployment-Adeployment-A`.
    """
    llm = ChatLiteLLMRouter(router=make_router())

    with patch.object(
        llm.router, "completion", return_value=iter(_router_chunks_with_cost())
    ):
        chunks = [chunk.message for chunk in llm._stream([])]

    assert _merge(chunks).response_metadata["model_id"] == "deployment-A"


def test_deployment_is_read_from_either_shape_litellm_hands_over() -> None:
    """The router's loops hold the response model, and its result builder a mapping.

    An unset deployment must add no key: a `model_id` of None reads downstream as a
    deployment that was named.
    """
    from types import SimpleNamespace

    named = {"model_id": "deployment-A"}
    assert _deployment_metadata({"_hidden_params": named}) == named
    assert _deployment_metadata(SimpleNamespace(_hidden_params=named)) == named
    assert _deployment_metadata({"_hidden_params": {"model_id": None}}) == {}
    assert _deployment_metadata({}) == {}


def test_router_create_chat_result_names_the_cost_and_deployment() -> None:
    """A complete response carries both in `_hidden_params`, alongside much else.

    The assertion is on the exact set this block builds, so copying `_hidden_params`
    wholesale fails here. What a caller finally receives is wider: `get_llm_output`
    copies litellm's router bookkeeping into `llm_output`, and core merges that in
    afterwards. Narrowing that copy is its own change.
    """
    llm = ChatLiteLLMRouter(router=make_router())
    mock_response = {
        "choices": [
            {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "_hidden_params": {
            "response_cost": 1.35e-05,
            "model_id": "deployment-A",
            "api_base": "https://faketesturl/",
            "optional_params": {"temperature": 0.1},
        },
    }

    result = llm._create_chat_result(mock_response, metadata={})
    metadata = result.generations[0].message.response_metadata

    assert metadata == {
        "model_name": llm.model_name or llm.model,
        "model_provider": "litellm",
        "response_cost": 1.35e-05,
        "model_id": "deployment-A",
    }


def _completion_filling_router_metadata() -> Callable[..., Any]:
    """Answer the way litellm does, by filling the caller's `metadata` dict in place.

    The Router treats that dict as an out-parameter for its own bookkeeping, so a
    double that ignores it cannot reproduce what a caller finally receives.
    """

    def _completion(**kwargs: Any) -> Any:
        kwargs["metadata"].update(
            {
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/fake-deployment-name-2",
                "api_base": "https://faketesturl/",
                "model_info": {"id": "deployment-A"},
                "hidden_params": {
                    "api_base": "https://faketesturl/",
                    "optional_params": {"temperature": 0.1},
                    "response_cost": 1.35e-05,
                },
            }
        )
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "_hidden_params": {"response_cost": 1.35e-05, "model_id": "deployment-A"},
        }

    return _completion


_DELIVERED_KEYS = {
    "model_name",
    "model_provider",
    "finish_reason",
    "token_usage",
    "response_cost",
    "model_id",
}


def test_router_delivers_only_the_keys_it_names() -> None:
    """`llm_output` reaches the message only after `_create_chat_result` returns.

    Reading the generation directly is blind to that merge, so this goes through
    `invoke()`, which is what a caller receives.
    """
    llm = ChatLiteLLMRouter(router=make_router())

    with patch.object(
        llm.router, "completion", side_effect=_completion_filling_router_metadata()
    ):
        metadata = llm.invoke("hi").response_metadata

    assert set(metadata) == _DELIVERED_KEYS


@pytest.mark.asyncio
async def test_router_delivers_the_same_keys_on_both_entry_points() -> None:
    """litellm fills `hidden_params` on the sync path and not on the async one.

    Copying that dict makes the delivered key set depend on which method was
    called, so a caller reading a key cannot know whether to expect it.
    """
    llm = ChatLiteLLMRouter(router=make_router())
    sync_double = _completion_filling_router_metadata()

    async def _acompletion(**kwargs: Any) -> Any:
        # litellm does not reach the logging path that writes `hidden_params` here.
        result = sync_double(**kwargs)
        kwargs["metadata"].pop("hidden_params", None)
        return result

    with patch.object(llm.router, "completion", side_effect=sync_double):
        sync_metadata = llm.invoke("hi").response_metadata
    with patch.object(llm.router, "acompletion", side_effect=_acompletion):
        async_metadata = (await llm.ainvoke("hi")).response_metadata

    assert set(sync_metadata) == set(async_metadata) == _DELIVERED_KEYS


def test_router_base_url_alias_reaches_completion() -> None:
    """Test that base_url flows through inheritance and survives router param stripping."""
    from litellm import Router

    from langchain_litellm import ChatLiteLLMRouter

    # Setup a basic underlying litellm router
    litellm_router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ]
    )

    # Initialize the LangChain router wrapper with a base_url override
    chat_router = ChatLiteLLMRouter(
        router=litellm_router, base_url="https://proxy.example/v1"
    )

    # Assert 1: The inherited validator normalized base_url into api_base
    assert chat_router.api_base == "https://proxy.example/v1"

    # Assert 2: it survives all the way to the outbound call, not just to
    # _prepare_params_for_router. Asserting on that helper alone left this green
    # while an override upstream was discarding the value.
    with patch.object(
        chat_router.router, "completion", return_value=_router_usage()
    ) as mock_completion:
        chat_router.invoke("hi")

    assert mock_completion.call_args.kwargs["api_base"] == "https://proxy.example/v1"


def test_router_generate_honours_max_retries() -> None:
    """ChatLiteLLMRouter._generate must retry via completion_with_retry.

    Regression test: previously `_generate` called `self.router.completion`
    directly, bypassing the tenacity retry decorator entirely, so
    `max_retries` had no effect for `ChatLiteLLMRouter`.
    """
    router = make_router()
    llm = ChatLiteLLMRouter(router=router, max_retries=4)

    with patch.object(
        llm.router, "completion", side_effect=_rate_limit_error()
    ) as mock_completion:
        with patch("time.sleep", return_value=None):  # skip tenacity backoff
            with pytest.raises(litellm.RateLimitError):
                llm.invoke("hi")

    assert mock_completion.call_count == 4


def test_router_stream_honours_max_retries() -> None:
    """ChatLiteLLMRouter._stream must retry via completion_with_retry."""
    router = make_router()
    llm = ChatLiteLLMRouter(router=router, max_retries=4, streaming=True)

    with patch.object(
        llm.router, "completion", side_effect=_rate_limit_error()
    ) as mock_completion:
        with patch("time.sleep", return_value=None):
            with pytest.raises(litellm.RateLimitError):
                list(llm.stream("hi"))

    assert mock_completion.call_count == 4


@pytest.mark.asyncio
async def test_router_agenerate_honours_max_retries() -> None:
    """ChatLiteLLMRouter._agenerate must retry via acompletion_with_retry."""
    router = make_router()
    llm = ChatLiteLLMRouter(router=router, max_retries=4)

    async def _raise(**kwargs: object) -> None:
        raise _rate_limit_error()

    with patch.object(
        llm.router, "acompletion", side_effect=_raise
    ) as mock_acompletion:
        with patch("asyncio.sleep", return_value=None):  # skip tenacity backoff
            with pytest.raises(litellm.RateLimitError):
                await llm.ainvoke("hi")

    assert mock_acompletion.call_count == 4


@pytest.mark.asyncio
async def test_router_astream_honours_max_retries() -> None:
    """ChatLiteLLMRouter._astream must retry via acompletion_with_retry.

    This is the only call site written as ``async for chunk in await ...``, so
    neither the ``_agenerate`` test (which awaits a value) nor the ``_stream``
    test (which iterates without awaiting) covers it.
    """
    router = make_router()
    llm = ChatLiteLLMRouter(router=router, max_retries=4, streaming=True)

    async def _raise(**kwargs: object) -> None:
        raise _rate_limit_error()

    with patch.object(
        llm.router, "acompletion", side_effect=_raise
    ) as mock_acompletion:
        with patch("asyncio.sleep", return_value=None):  # skip tenacity backoff
            with pytest.raises(litellm.RateLimitError):
                async for _ in llm.astream("hi"):
                    pass

    assert mock_acompletion.call_count == 4


def test_router_generate_no_retry_on_success() -> None:
    """A successful router call must not be retried unnecessarily."""
    from litellm.utils import Usage

    router = make_router()
    llm = ChatLiteLLMRouter(router=router, max_retries=4)

    mock_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    }

    with patch.object(
        llm.router, "completion", return_value=mock_response
    ) as mock_completion:
        result = llm.invoke("hi")

    assert mock_completion.call_count == 1
    assert result.content == "hello"


def _usage_response() -> dict:
    from litellm.utils import Usage

    return {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    }


def test_router_set_default_model_changes_the_model_sent() -> None:
    """`_default_params` prefers `model_name`, so setting only `model` had no effect."""
    llm = ChatLiteLLMRouter(router=make_router(), model_name="gpt-4")

    with patch.object(
        llm.router, "completion", return_value=_usage_response()
    ) as first:
        llm.invoke("hi")
    llm.set_default_model("gpt-3.5-turbo")
    with patch.object(
        llm.router, "completion", return_value=_usage_response()
    ) as second:
        llm.invoke("hi")

    assert first.call_args.kwargs["model"] == "gpt-4"
    assert second.call_args.kwargs["model"] == "gpt-3.5-turbo"


def test_router_is_claude_model_reads_the_deployment() -> None:
    """The Router alias need not contain the provider's model name at all."""
    import litellm

    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-alias",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-20241022",
                    "api_key": "sk-x",
                },
            }
        ]
    )
    llm = ChatLiteLLMRouter(router=router)
    assert llm._is_claude_model() is True

    assert ChatLiteLLMRouter(router=make_router())._is_claude_model() is False


def test_router_combine_llm_outputs_accepts_a_plain_dict_usage() -> None:
    """`_create_chat_result` passes `response["usage"]` through unchanged."""
    llm = ChatLiteLLMRouter(router=make_router())
    combined = llm._combine_llm_outputs(
        [{"token_usage": {"total_tokens": 3}, "model": "gpt-4"}]
    )
    assert combined["token_usage"]["total_tokens"] == 3


@pytest.mark.parametrize("method", ["stream", "astream"])
@pytest.mark.asyncio
async def test_router_surfaces_finish_reason_in_response_metadata(method: str) -> None:
    """The Router overrides both streaming paths, so it needs the same handling.

    A caller streaming through a Router sees the same completed response the base
    class reports, so it carries the same `finish_reason`.
    """
    chunks = [
        {
            "choices": [{"delta": {"role": "assistant", "content": "hel"}}],
            "usage": None,
        },
        {
            "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}],
            "usage": None,
        },
    ]
    llm = ChatLiteLLMRouter(router=make_router())

    async def _acompletion(**kwargs: Any) -> Any:
        async def _aiter() -> Any:
            for chunk in chunks:
                yield chunk

        return _aiter()

    merged = None
    if method == "stream":
        with patch.object(llm.router, "completion", return_value=iter(chunks)):
            for chunk in llm.stream("hi"):
                merged = chunk if merged is None else merged + chunk
    else:
        with patch.object(llm.router, "acompletion", side_effect=_acompletion):
            async for chunk in llm.astream("hi"):
                merged = chunk if merged is None else merged + chunk

    assert merged is not None
    assert merged.response_metadata["finish_reason"] == "stop"
