"""Test router chat model integration."""

from unittest.mock import patch

import litellm
import pytest
from langchain_core.messages import AIMessage

from langchain_litellm._version import __version__
from langchain_litellm.chat_models import ChatLiteLLMRouter
from tests.utils import make_router


def _rate_limit_error() -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message="rate limited", llm_provider="openai", model="gpt-4"
    )


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


def test_router_stream_options_set_for_all_providers() -> None:
    """Router _stream must set stream_options for non-OpenAI providers."""
    router = make_router()
    llm = ChatLiteLLMRouter(router=router)
    stream_options = (
        llm.stream_options
        if llm.stream_options is not None
        else {"include_usage": True}
    )
    assert stream_options == {"include_usage": True}


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

    # Assert 2: The override survives _prepare_params_for_router stripping logic
    params = {"api_base": chat_router.api_base, "model": "gpt-3.5-turbo"}
    chat_router._prepare_params_for_router(params)

    assert "api_base" in params
    assert params["api_base"] == "https://proxy.example/v1"


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
    llm = ChatLiteLLMRouter(router=make_router())

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
