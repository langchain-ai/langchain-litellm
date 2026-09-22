"""Test chat model integration."""

# stdlib
import logging
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Union
from unittest.mock import patch

# third-party
import litellm
import pytest
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableLambda
from litellm.types.utils import ChatCompletionDeltaToolCall, Delta, Function
from pydantic import BaseModel, ValidationError

# first-party
from langchain_litellm._version import __version__
from langchain_litellm.chat_models import ChatLiteLLM
from langchain_litellm.chat_models.litellm import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
    _create_usage_metadata,
    _provider_api_key_field,
)


def _dummy_tool(x: str) -> str:
    """A dummy tool for testing."""
    return x


class _StructuredResponse(BaseModel):
    value: str


# ── delta / message conversion ────────────────────────────────────────────────


def test_litellm_delta_to_langchain_message_chunk() -> None:
    """Test the litellm._convert_delta_to_message_chunk method, to ensure compatibility when converting a LiteLLM delta to a LangChain message chunk."""
    mock_content = "This is a test content"
    mock_tool_call_id = "call_test"
    mock_tool_call_name = "test_tool_call"
    mock_tool_call_arguments = ""
    mock_tool_call_index = 3
    mock_delta = Delta(
        content=mock_content,
        role="assistant",
        tool_calls=[
            ChatCompletionDeltaToolCall(
                id=mock_tool_call_id,
                function=Function(
                    arguments=mock_tool_call_arguments, name=mock_tool_call_name
                ),
                type="function",
                index=mock_tool_call_index,
            )
        ],
    )
    message_chunk = _convert_delta_to_message_chunk(mock_delta, AIMessageChunk)
    assert isinstance(message_chunk, AIMessageChunk)
    assert message_chunk.content == mock_content
    tool_call_chunk = message_chunk.tool_call_chunks[0]
    assert tool_call_chunk["id"] == mock_tool_call_id
    assert tool_call_chunk["name"] == mock_tool_call_name
    assert tool_call_chunk["args"] == mock_tool_call_arguments
    assert tool_call_chunk["index"] == mock_tool_call_index


def test_convert_dict_to_tool_message() -> None:
    """Ensure tool role dicts convert to ToolMessage."""
    mock_dict = {"role": "tool", "content": "result", "tool_call_id": "123"}
    message = _convert_dict_to_message(mock_dict)
    assert isinstance(message, ToolMessage)
    assert message.content == "result"
    assert message.tool_call_id == "123"


def test_malformed_tool_call_arguments_are_reported_as_invalid() -> None:
    """Unparsable arguments must not become a tool call with empty args.

    A truncated `arguments` string cannot be recovered. Returning it as a valid
    tool call with `args={}` makes an agent invoke the tool with no input and
    leaves it no way to detect the failure, so the call belongs in
    `invalid_tool_calls` with the raw string kept for inspection.
    """
    raw_arguments = '{"city": "Par'
    message = _convert_dict_to_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": raw_arguments},
                }
            ],
        }
    )

    assert isinstance(message, AIMessage)
    assert message.tool_calls == []
    assert len(message.invalid_tool_calls) == 1

    invalid = message.invalid_tool_calls[0]
    assert invalid["name"] == "get_weather"
    assert invalid["args"] == raw_arguments
    assert invalid["id"] == "call_1"


def test_unparsable_tool_calls_are_not_echoed_back() -> None:
    """A call that never parsed was never dispatched, so nothing can answer it.

    Sending the raw arguments back makes the provider fail the same parse, which
    breaks the recovery turn that reporting the call as invalid exists to enable.
    """
    message = _convert_dict_to_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "bad",
                    "type": "function",
                    "function": {"name": "broken", "arguments": '{"city": "Par'},
                }
            ],
        }
    )

    assert "tool_calls" not in _convert_message_to_dict(message)


def test_valid_tool_calls_survive_an_invalid_sibling_in_the_request() -> None:
    """Dropping the unparsable call must not drop the ones that did parse."""
    message = _convert_dict_to_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "ok",
                    "type": "function",
                    "function": {"name": "with_args", "arguments": '{"x": 1}'},
                },
                {
                    "id": "bad",
                    "type": "function",
                    "function": {"name": "broken", "arguments": "{oops"},
                },
            ],
        }
    )

    sent = _convert_message_to_dict(message)["tool_calls"]
    assert [tc["function"]["name"] for tc in sent] == ["with_args"]


def test_tool_calls_partition_valid_and_invalid_arguments() -> None:
    """Valid calls in the same response are unaffected by an invalid sibling."""
    message = _convert_dict_to_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "ok",
                    "type": "function",
                    "function": {"name": "with_args", "arguments": '{"x": 1}'},
                },
                {
                    "id": "bad",
                    "type": "function",
                    "function": {"name": "broken", "arguments": "{oops"},
                },
                {
                    "id": "empty",
                    "type": "function",
                    "function": {"name": "no_args", "arguments": "{}"},
                },
            ],
        }
    )

    assert isinstance(message, AIMessage)
    assert [tc["name"] for tc in message.tool_calls] == ["with_args", "no_args"]
    assert message.tool_calls[0]["args"] == {"x": 1}
    assert message.tool_calls[1]["args"] == {}
    assert [tc["name"] for tc in message.invalid_tool_calls] == ["broken"]


@pytest.fixture
def _no_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep key resolution independent of the developer's own environment."""
    for var in (
        "OPENAI_API_KEY",
        "AZURE_API_KEY",
        "ANTHROPIC_API_KEY",
        "REPLICATE_API_KEY",
        "OPENROUTER_API_KEY",
        "COHERE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_provider_specific_api_key_is_forwarded(_no_provider_env: None) -> None:
    """A key passed as `openai_api_key` must reach litellm.

    Only the generic `api_key` was forwarded, so a provider-specific key was
    accepted, stored, and then dropped before the request.
    """
    llm = ChatLiteLLM(model="gpt-4o", openai_api_key="sk-openai")
    assert llm._client_params["api_key"] == "sk-openai"


def test_explicit_api_key_takes_precedence(_no_provider_env: None) -> None:
    """`api_key` still wins when both are supplied."""
    llm = ChatLiteLLM(model="gpt-4o", api_key="sk-explicit", openai_api_key="sk-openai")
    assert llm._client_params["api_key"] == "sk-explicit"


def test_provider_specific_api_key_not_used_for_other_provider(
    _no_provider_env: None,
) -> None:
    """A key is only forwarded to the provider it belongs to."""
    llm = ChatLiteLLM(
        model="openrouter/meta-llama/llama-3-8b-instruct",
        openai_api_key="sk-openai",
    )
    assert llm._client_params.get("api_key") is None


def test_custom_llm_provider_selects_the_api_key(_no_provider_env: None) -> None:
    """An explicit `custom_llm_provider` decides which field is used."""
    llm = ChatLiteLLM(
        model="my-proxy-deployment",
        custom_llm_provider="anthropic",
        anthropic_api_key="sk-anthropic",
    )
    assert llm._client_params["api_key"] == "sk-anthropic"


_MOCK_OK = {
    "choices": [
        {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def test_per_call_model_override_does_not_reuse_the_other_providers_key(
    _no_provider_env: None,
) -> None:
    """A per-call `model` override must re-resolve the key it travels with.

    `_client_params` resolves the key from the constructor's model, but the four
    entry points merge per-call kwargs over it afterwards. Without re-resolving,
    the OpenAI key would be sent as the credential for an Anthropic model.

    Asserted at the litellm boundary rather than on `_client_params`, which is
    evaluated before the override and so cannot observe this.
    """
    llm = ChatLiteLLM(model="gpt-4o", openai_api_key="sk-openai")

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi", model="anthropic/claude-3-5-sonnet-20241022")

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["model"] == "anthropic/claude-3-5-sonnet-20241022"
    assert kwargs["api_key"] != "sk-openai"
    assert kwargs["api_key"] is None


def test_per_call_model_override_selects_that_providers_key(
    _no_provider_env: None,
) -> None:
    """The override picks the field belonging to the model actually being called."""
    llm = ChatLiteLLM(
        model="gpt-4o", openai_api_key="sk-openai", anthropic_api_key="sk-anthropic"
    )

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi", model="anthropic/claude-3-5-sonnet-20241022")

    assert mock_completion.call_args.kwargs["api_key"] == "sk-anthropic"


@pytest.mark.parametrize(
    ("provider", "field"),
    [("huggingface", "huggingface_api_key"), ("together_ai", "together_ai_api_key")],
)
def test_late_declared_provider_keys_reach_litellm(
    _no_provider_env: None, provider: str, field: str
) -> None:
    """These two had no field, so pydantic discarded whatever the caller passed."""
    kwargs: Dict[str, Any] = {"model": f"{provider}/some-model", field: "sk-late"}
    llm = ChatLiteLLM(**kwargs)  # type: ignore[arg-type]

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi")

    assert mock_completion.call_args.kwargs["api_key"] == "sk-late"


def test_an_ambient_env_key_is_not_sent_as_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """litellm resolves the environment itself, and an explicit key overrides it.

    Passing back a value read from the environment beats `litellm.api_key` and the
    other module globals, so configuring litellm programmatically stops working.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient")
    llm = ChatLiteLLM(model="gpt-4o")

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi")

    assert mock_completion.call_args.kwargs.get("api_key") is None


def test_an_ambient_env_key_does_not_follow_a_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pinned gateway holds the credential steady, so it must hold a real one.

    Freezing a key the caller never supplied sends an OpenAI environment key to an
    arbitrary endpoint as the credential for an Anthropic model.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient")
    llm = ChatLiteLLM(model="gpt-4o", api_base="https://gateway.internal/v1")

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi", model="anthropic/claude-3-5-sonnet-20241022")

    kwargs = mock_completion.call_args.kwargs
    assert kwargs.get("api_key") is None
    assert kwargs["api_base"] == "https://gateway.internal/v1"


def test_no_provider_key_configured_skips_the_registry_lookup(
    _no_provider_env: None,
) -> None:
    """litellm prints a banner to stdout for every model it cannot attribute.

    With no provider-scoped key set there is nothing to attribute, so a proxy
    deployment name must not cost a lookup and that banner on every request.
    """
    llm = ChatLiteLLM(model="my-deployment")

    with (
        patch.object(litellm, "get_llm_provider") as mock_lookup,
        patch.object(llm.client, "completion", return_value=_MOCK_OK),
    ):
        llm.invoke("hi", custom_llm_provider="openai")

    mock_lookup.assert_not_called()


def test_a_subclass_provider_key_field_is_forwarded(_no_provider_env: None) -> None:
    """Resolution reads the runtime class, so a subclass can add a provider.

    Pinning the base class drops a field the subclass declares and accepts, which
    is the silent discard this resolution exists to stop.
    """

    class _DeepSeekChat(ChatLiteLLM):
        deepseek_api_key: Optional[str] = None

    llm = _DeepSeekChat(model="deepseek/deepseek-chat", deepseek_api_key="sk-deepseek")

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi")

    assert mock_completion.call_args.kwargs["api_key"] == "sk-deepseek"


def test_an_unexpected_provider_lookup_error_surfaces(_no_provider_env: None) -> None:
    """Only litellm's own "cannot attribute this model" is a reason to give up.

    Swallowing anything else silently stops forwarding the key, reproducing the bug
    this resolution fixes with no signal that it happened.
    """
    llm = ChatLiteLLM(model="gpt-4o", openai_api_key="sk-openai")

    with patch.object(
        litellm, "get_llm_provider", side_effect=TypeError("signature changed")
    ):
        with pytest.raises(TypeError):
            llm._client_params


def test_model_kwargs_decides_the_timeout(_no_provider_env: None) -> None:
    """`model_kwargs` is merged last, so it decides every parameter it names.

    `timeout` used to be the one exception, overwritten by `request_timeout` after
    the merge.
    """
    llm = ChatLiteLLM(model="gpt-4o", request_timeout=7, model_kwargs={"timeout": 42})

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi")

    assert mock_completion.call_args.kwargs["timeout"] == 42


def test_redirect_re_resolves_only_the_inferred_credential(
    _no_provider_env: None,
) -> None:
    """Only the provider-scoped key is inferred from the destination.

    `api_base`, `organization` and `extra_headers` exist because a caller set them,
    so a redirect must not discard that intent. Without an `api_base` pinned, the
    key is the one thing derived from where the request goes, so it re-resolves.
    """
    llm = ChatLiteLLM(
        model="gpt-4o",
        openai_api_key="sk-openai",
        anthropic_api_key="sk-anthropic",
        organization="org-openai",
        extra_headers={"X-Team": "platform"},
    )

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi", model="anthropic/claude-3-5-sonnet-20241022")

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["api_key"] == "sk-anthropic"
    assert kwargs["organization"] == "org-openai"
    assert kwargs["extra_headers"] == {"X-Team": "platform"}


def test_base_model_survives_a_redirect(_no_provider_env: None) -> None:
    """`base_model` is caller configuration, not something inferred from the model.

    It drives cost attribution for deployments litellm's cost map does not know, so
    dropping it on a redirect would silently misattribute spend.
    """
    llm = ChatLiteLLM(model="gpt-4o", openai_api_key="sk-openai", base_model="gpt-4o")

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi", model="anthropic/claude-3-5-sonnet-20241022")

    assert mock_completion.call_args.kwargs["base_model"] == "gpt-4o"


def test_a_pinned_api_base_keeps_its_credential_across_a_redirect(
    _no_provider_env: None,
) -> None:
    """A pinned endpoint is one gateway serving many models on one credential.

    Swapping in a provider-scoped key would send the wrong credential to that
    gateway, and dropping the endpoint would ignore the caller's explicit choice.
    """
    llm = ChatLiteLLM(
        model="gpt-4o",
        openai_api_key="sk-openai",
        anthropic_api_key="sk-anthropic",
        api_base="https://gateway.internal/v1",
    )

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi", model="anthropic/claude-3-5-sonnet-20241022")

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["api_base"] == "https://gateway.internal/v1"
    assert kwargs["api_key"] == "sk-openai"


def test_caller_supplied_destination_params_survive_a_redirect(
    _no_provider_env: None,
) -> None:
    """A caller who wants an endpoint at the new destination passes it per call."""
    llm = ChatLiteLLM(
        model="gpt-4o", openai_api_key="sk-openai", api_base="https://old/v1"
    )

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke(
            "hi",
            model="anthropic/claude-3-5-sonnet-20241022",
            api_base="https://new/v1",
        )

    assert mock_completion.call_args.kwargs["api_base"] == "https://new/v1"


def test_no_redirect_keeps_the_configured_destination(_no_provider_env: None) -> None:
    """Without a redirect nothing is dropped."""
    llm = ChatLiteLLM(
        model="gpt-4o", openai_api_key="sk-openai", api_base="https://proxy/v1"
    )

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi")

    assert mock_completion.call_args.kwargs["api_base"] == "https://proxy/v1"


def test_none_override_is_treated_as_omitted(_no_provider_env: None) -> None:
    """`model=None` must fall back rather than sending a null destination."""
    llm = ChatLiteLLM(model="gpt-4o", openai_api_key="sk-openai")

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi", model=None, custom_llm_provider=None)

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["api_key"] == "sk-openai"


def test_model_kwargs_destination_decides_the_key(_no_provider_env: None) -> None:
    """`model_kwargs` is merged last, so an entry there is the real destination."""
    llm = ChatLiteLLM(
        model="gpt-4o",
        openai_api_key="sk-openai",
        anthropic_api_key="sk-anthropic",
        model_kwargs={"model": "anthropic/claude-3-5-sonnet-20241022"},
    )
    assert llm._client_params["api_key"] == "sk-anthropic"


def test_model_kwargs_api_key_survives_a_redirect(_no_provider_env: None) -> None:
    """A generic key is provider-agnostic wherever it was supplied.

    `model_kwargs["api_key"]` reaches litellm like the field does, so a redirect must
    not replace it with a provider-scoped resolution.
    """
    llm = ChatLiteLLM(
        model="gpt-4o",
        anthropic_api_key="sk-anthropic",
        model_kwargs={"api_key": "sk-generic"},
    )

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi", model="anthropic/claude-3-5-sonnet-20241022")

    assert mock_completion.call_args.kwargs["api_key"] == "sk-generic"


def test_model_kwargs_credentials_are_not_clobbered(_no_provider_env: None) -> None:
    """An unset field must not overwrite the same key supplied via model_kwargs."""
    llm = ChatLiteLLM(
        model="gpt-4o",
        openai_api_key="sk-openai",
        model_kwargs={"api_base": "https://from-model-kwargs/v1"},
    )
    assert llm._client_params["api_base"] == "https://from-model-kwargs/v1"


def test_provider_api_key_field_is_derived_from_declared_fields() -> None:
    """The mapping follows the `<provider>_api_key` convention, with aliases named."""
    assert _provider_api_key_field(ChatLiteLLM, "anthropic") == "anthropic_api_key"
    assert _provider_api_key_field(ChatLiteLLM, "cohere_chat") == "cohere_api_key"
    assert (
        _provider_api_key_field(ChatLiteLLM, "text-completion-openai")
        == "openai_api_key"
    )
    # No declared field, so no key is guessed.
    assert _provider_api_key_field(ChatLiteLLM, "bedrock") is None
    assert _provider_api_key_field(ChatLiteLLM, None) is None


def test_per_call_custom_llm_provider_does_not_reuse_the_other_providers_key(
    _no_provider_env: None,
) -> None:
    """`custom_llm_provider` is the other per-call route to the destination.

    It decides where litellm sends the request just as `model` does, so a key
    resolved from the constructor's provider must not travel with it.
    """
    llm = ChatLiteLLM(model="gpt-4o", openai_api_key="sk-openai")

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi", custom_llm_provider="anthropic")

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["custom_llm_provider"] == "anthropic"
    assert kwargs["api_key"] != "sk-openai"
    assert kwargs["api_key"] is None


def test_per_call_custom_llm_provider_selects_that_providers_key(
    _no_provider_env: None,
) -> None:
    """The per-call provider picks its own field, even for a model litellm cannot attribute."""
    llm = ChatLiteLLM(
        model="totally-made-up-model",
        openai_api_key="sk-openai",
        anthropic_api_key="sk-anthropic",
    )

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi", custom_llm_provider="anthropic")

    assert mock_completion.call_args.kwargs["api_key"] == "sk-anthropic"


def test_explicit_per_call_api_key_survives_a_model_override(
    _no_provider_env: None,
) -> None:
    """An `api_key` passed per call still wins over any resolution."""
    llm = ChatLiteLLM(model="gpt-4o", openai_api_key="sk-openai")

    with patch.object(
        llm.client, "completion", return_value=_MOCK_OK
    ) as mock_completion:
        llm.invoke(
            "hi", model="anthropic/claude-3-5-sonnet-20241022", api_key="sk-explicit"
        )

    assert mock_completion.call_args.kwargs["api_key"] == "sk-explicit"


def test_unattributable_model_leaves_api_key_unset(_no_provider_env: None) -> None:
    """Models litellm cannot attribute fall back to the previous behaviour."""
    llm = ChatLiteLLM(model="totally-made-up-model", openai_api_key="sk-openai")
    assert llm._client_params.get("api_key") is None


def test_provider_specific_fields_in_delta() -> None:
    """Test that provider_specific_fields are preserved when converting deltas."""
    mock_delta = {
        "role": "assistant",
        "content": "Paris is the capital of France",
        "provider_specific_fields": {
            "citations": [
                {"source": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Paris"}
            ]
        },
    }

    chunk = _convert_delta_to_message_chunk(mock_delta, AIMessageChunk)

    assert isinstance(chunk, AIMessageChunk)
    assert "provider_specific_fields" in chunk.additional_kwargs
    assert (
        chunk.additional_kwargs["provider_specific_fields"]["citations"][0]["source"]
        == "Wikipedia"
    )


def test_provider_specific_fields_in_message() -> None:
    """Test that provider_specific_fields are preserved when converting message dicts."""
    mock_message_dict = {
        "role": "assistant",
        "content": "The Earth orbits the Sun",
        "provider_specific_fields": {
            "grounding_metadata": {
                "search_queries": ["Earth orbit"],
                "grounding_supports": [{"segment": "The Earth orbits"}],
            }
        },
    }

    message = _convert_dict_to_message(mock_message_dict)

    assert isinstance(message, AIMessage)
    assert "provider_specific_fields" in message.additional_kwargs
    assert "grounding_metadata" in message.additional_kwargs["provider_specific_fields"]


def test_provider_specific_fields_in_chat_result() -> None:
    """Test that top-level provider_specific_fields appear in llm_output."""
    llm = ChatLiteLLM(model="gpt-3.5-turbo", api_key="fake")

    mock_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "provider_specific_fields": {"citations": [{"source": "test"}]},
    }

    result = llm._create_chat_result(mock_response)

    assert result.llm_output is not None
    assert "provider_specific_fields" in result.llm_output
    assert (
        result.llm_output["provider_specific_fields"]["citations"][0]["source"]
        == "test"
    )


# ── usage metadata ─────────────────────────────────────────────────────────────


def test_create_usage_metadata_reads_pydantic_prompt_details() -> None:
    """Cache token details should be extracted from Pydantic prompt_tokens_details."""
    from litellm.types.utils import PromptTokensDetailsWrapper, Usage

    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=30,
            cache_creation_tokens=10,
        ),
    )
    meta = _create_usage_metadata(usage)
    assert meta["input_tokens"] == 100
    assert meta["input_token_details"]["cache_read"] == 30
    assert meta["input_token_details"]["cache_creation"] == 10


def test_create_usage_metadata_reads_dict_prompt_details() -> None:
    """Cache token details should also work from plain dict prompt_tokens_details."""
    usage = {
        "prompt_tokens": 50,
        "completion_tokens": 25,
        "total_tokens": 75,
        "prompt_tokens_details": {
            "cached_tokens": 15,
            "cache_creation_tokens": 5,
        },
    }
    meta = _create_usage_metadata(usage)
    assert meta["input_tokens"] == 50
    assert meta["input_token_details"]["cache_read"] == 15
    assert meta["input_token_details"]["cache_creation"] == 5


def test_create_usage_metadata_uses_total_tokens_from_response() -> None:
    """total_tokens should be read from the response, not recomputed."""
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 20,  # deliberately != 10 + 5
    }
    meta = _create_usage_metadata(usage)
    assert meta["total_tokens"] == 20


def test_create_usage_metadata_extracts_reasoning_tokens() -> None:
    """Reasoning tokens from completion_tokens_details should populate
    output_token_details."""
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 50,
        "total_tokens": 60,
        "completion_tokens_details": {"reasoning_tokens": 30},
    }
    meta = _create_usage_metadata(usage)
    assert meta["output_token_details"]["reasoning"] == 30


def test_create_usage_metadata_extracts_reasoning_tokens_pydantic() -> None:
    """Reasoning tokens should be extracted from Pydantic Usage models too."""
    from litellm.types.utils import CompletionTokensDetailsWrapper, Usage

    usage = Usage(
        prompt_tokens=10,
        completion_tokens=50,
        total_tokens=60,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=30,
        ),
    )
    meta = _create_usage_metadata(usage)
    assert meta["output_token_details"]["reasoning"] == 30


def test_create_usage_metadata_handles_none_values() -> None:
    """Explicit None values for token counts should be treated as 0."""
    usage = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    meta = _create_usage_metadata(usage)
    assert meta["input_tokens"] == 0
    assert meta["output_tokens"] == 0
    assert meta["total_tokens"] == 0


# ── reasoning content stays out of message content ─────────────────────────────


def test_reasoning_content_does_not_alter_content_for_dict() -> None:
    """`.content` must stay a plain str; reasoning_content lives in additional_kwargs."""
    mock_dict = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "hidden chain",
    }

    message = _convert_dict_to_message(mock_dict)

    assert message.content == "answer"
    assert message.additional_kwargs["reasoning_content"] == "hidden chain"


def test_reasoning_content_does_not_alter_empty_content_for_dict() -> None:
    mock_dict = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "hidden chain",
    }

    message = _convert_dict_to_message(mock_dict)

    assert message.content == ""
    assert message.additional_kwargs["reasoning_content"] == "hidden chain"


def test_reasoning_content_does_not_alter_content_for_delta() -> None:
    mock_delta = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "hidden chain",
    }

    chunk = _convert_delta_to_message_chunk(mock_delta, AIMessageChunk)

    assert chunk.content == "answer"
    assert chunk.additional_kwargs["reasoning_content"] == "hidden chain"


def test_reasoning_content_surfaces_as_standard_content_block() -> None:
    """Consumers reading the standard `content_blocks` API (e.g. LangGraph) still
    see reasoning content even though it is no longer duplicated into `.content`."""
    message = AIMessage(
        content="answer", additional_kwargs={"reasoning_content": "hidden chain"}
    )

    assert {"type": "reasoning", "reasoning": "hidden chain"} in message.content_blocks


# ── credential forwarding ─────────────────────────────────────────────────────


def test_client_params_forwards_api_key() -> None:
    """api_key must be forwarded as an explicit kwarg so providers can prefer it
    instead of using the environment variables."""
    llm = ChatLiteLLM(
        model="openrouter/anthropic/claude-sonnet-4-5",
        api_base="https://openrouter.ai/api/v1",
        api_key="my-explicit-token",
    )
    params = llm._client_params
    assert params.get("api_key") == "my-explicit-token"


# ── stream_options ─────────────────────────────────────────────────────────────


def test_stream_options_set_for_non_openai_model() -> None:
    """stream_options must be set for non-OpenAI providers too."""
    llm = ChatLiteLLM(model="anthropic/claude-3-5-sonnet-20241022", api_key="fake")
    _, params = llm._create_message_dicts([], None)
    # Simulate what _stream does
    params = {**params, "stream": True}
    if llm.stream_options is not None:
        params["stream_options"] = llm.stream_options
    else:
        params["stream_options"] = {"include_usage": True}
    assert params.get("stream_options") == {"include_usage": True}


def test_stream_options_respected_when_set_explicitly() -> None:
    """User-provided stream_options must not be overwritten."""
    custom = {"include_usage": False}
    llm = ChatLiteLLM(
        model="anthropic/claude-3-5-sonnet-20241022",
        api_key="fake",
        stream_options=custom,
    )
    params = {}
    if llm.stream_options is not None:
        params["stream_options"] = llm.stream_options
    else:
        params["stream_options"] = {"include_usage": True}
    assert params["stream_options"] == custom


# ── tool_choice mapping with thinking enabled ──────────────────────────────────

_THINKING_KWARGS = {"thinking": {"type": "enabled", "budget_tokens": 5000}}


def test_bind_tools_any_becomes_required_without_thinking() -> None:
    """`tool_choice='any'` should map to `'required'`."""
    llm = ChatLiteLLM(model="anthropic/claude-sonnet-4-20250514", api_key="fake")
    bound = llm.bind_tools([_dummy_tool], tool_choice="any")
    assert bound.kwargs["tool_choice"] == "required"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "tool_choice",
    [
        "any",
        "required",
        True,
        {"type": "function", "function": {"name": "_dummy_tool"}},
    ],
    ids=["any", "required", "True", "dict"],
)
def test_bind_tools_downgraded_with_thinking(
    tool_choice: Union[str, bool, Dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Forced tool_choice values should be downgraded to 'auto' when thinking
    is enabled, so the model can produce CoT text before tool calls.
    """
    llm = ChatLiteLLM(
        model="anthropic/claude-sonnet-4-20250514",
        api_key="fake",
        model_kwargs=_THINKING_KWARGS,
    )
    with caplog.at_level(
        logging.WARNING, logger="langchain_litellm.chat_models.litellm"
    ):
        bound = llm.bind_tools([_dummy_tool], tool_choice=tool_choice)
    assert bound.kwargs["tool_choice"] == "auto"  # type: ignore[attr-defined]
    assert "incompatible with thinking" in caplog.text


@pytest.mark.parametrize(
    "tool_choice",
    [
        "any",
        "required",
        True,
        {"type": "function", "function": {"name": "_dummy_tool"}},
    ],
    ids=["any", "required", "True", "dict"],
)
def test_bind_tools_not_downgraded_with_thinking_on_non_claude_models(
    tool_choice: Union[str, bool, Dict[str, Any]],
) -> None:
    """Forced tool choices should be preserved for non-Claude models."""
    llm = ChatLiteLLM(
        model="gpt-4o-mini",
        api_key="fake",
        model_kwargs=_THINKING_KWARGS,
    )
    bound = llm.bind_tools([_dummy_tool], tool_choice=tool_choice)
    expected_tool_choice = "required" if tool_choice in ("any", True) else tool_choice
    assert bound.kwargs["tool_choice"] == expected_tool_choice  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "tool_choice",
    ["auto", "none", None, False],
    ids=["auto", "none", "None", "False"],
)
def test_bind_tools_non_forced_unchanged_with_thinking(
    tool_choice: Optional[Union[str, bool]],
) -> None:
    """Non-forced tool_choice values should pass through untouched."""
    llm = ChatLiteLLM(
        model="anthropic/claude-sonnet-4-20250514",
        api_key="fake",
        model_kwargs=_THINKING_KWARGS,
    )
    bound = llm.bind_tools([_dummy_tool], tool_choice=tool_choice)
    assert bound.kwargs["tool_choice"] == tool_choice  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "thinking_config",
    [None, {}, {"type": "disabled"}],
    ids=["None", "empty", "disabled"],
)
def test_bind_tools_no_downgrade_without_thinking_enabled(
    thinking_config: Optional[Dict[str, Any]],
) -> None:
    """tool_choice='any' should stay 'required' when thinking is not enabled."""
    kwargs: dict = {}
    if thinking_config is not None:
        kwargs["thinking"] = thinking_config
    llm = ChatLiteLLM(
        model="anthropic/claude-sonnet-4-20250514",
        api_key="fake",
        model_kwargs=kwargs,
    )
    bound = llm.bind_tools([_dummy_tool], tool_choice="any")
    assert bound.kwargs["tool_choice"] == "required"  # type: ignore[attr-defined]


def test_bind_tools_dict_validation_with_thinking() -> None:
    """Invalid dict tool_choice should raise ValueError even with thinking."""
    llm = ChatLiteLLM(
        model="anthropic/claude-sonnet-4-20250514",
        api_key="fake",
        model_kwargs=_THINKING_KWARGS,
    )
    with pytest.raises(ValueError, match="nonexistent_tool"):
        llm.bind_tools(
            [_dummy_tool],
            tool_choice={"type": "function", "function": {"name": "nonexistent_tool"}},
        )


def test_with_structured_output_function_calling_warns_and_raises_for_claude_thinking() -> (
    None
):
    """Claude thinking should not silently fall back to plain-text structured output."""
    bind_kwargs: dict[str, Any] = {}

    class _FakeChatLiteLLM(ChatLiteLLM):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            bind_kwargs.update(kwargs)
            return RunnableLambda(lambda _: AIMessage(content="plain text"))

    llm = _FakeChatLiteLLM(
        model="anthropic/claude-sonnet-4-20250514",
        api_key="fake",
        model_kwargs=_THINKING_KWARGS,
    )

    with pytest.warns(UserWarning, match="Structured output via function calling"):
        structured = llm.with_structured_output(
            _StructuredResponse, method="function_calling"
        )

    assert "tool_choice" not in bind_kwargs
    with pytest.raises(OutputParserException, match="no tool call is returned"):
        structured.invoke("Return structured output.")


def test_with_structured_output_include_raw_preserves_raw_for_claude_thinking() -> None:
    """`include_raw` should surface the parsing error without dropping the raw message."""

    class _FakeChatLiteLLM(ChatLiteLLM):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            return RunnableLambda(lambda _: AIMessage(content="plain text"))

    llm = _FakeChatLiteLLM(
        model="anthropic/claude-sonnet-4-20250514",
        api_key="fake",
        model_kwargs=_THINKING_KWARGS,
    )

    with pytest.warns(UserWarning, match="Structured output via function calling"):
        structured = llm.with_structured_output(
            _StructuredResponse,
            method="function_calling",
            include_raw=True,
        )

    result = structured.invoke("Return structured output.")

    assert isinstance(result, dict)
    assert isinstance(result["raw"], AIMessage)
    assert result["raw"].content == "plain text"
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], OutputParserException)


def test_create_chat_result_sets_model_provider() -> None:
    """Non-streaming path must set model_provider. Fixes #152."""
    llm = ChatLiteLLM(model="gpt-4", api_key="fake")
    mock_response = {
        "choices": [
            {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    result = llm._create_chat_result(mock_response)
    msg = result.generations[0].message
    assert isinstance(msg, AIMessage)
    assert msg.response_metadata.get("model_provider") == "litellm"


def test_stream_sets_model_provider_in_response_metadata() -> None:
    """First streaming chunk must carry model_provider. Fixes #152."""

    llm = ChatLiteLLM(model="gpt-4", api_key="fake")
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

    with patch.object(
        ChatLiteLLM, "completion_with_retry", return_value=iter(fake_chunks)
    ):
        chunks = list(llm._stream([]))

    assert chunks[0].message.response_metadata.get("model_provider") == "litellm"
    assert chunks[1].message.response_metadata == {}


def test_get_ls_params_sets_ls_provider() -> None:
    """ls_provider must match model_provider so SummarizationMiddleware's equality check passes."""
    llm = ChatLiteLLM(model="gpt-4", api_key="fake")
    params = llm._get_ls_params()
    assert params["ls_provider"] == "litellm"
    assert params["ls_model_name"] == "gpt-4"

    # model_name takes precedence over model when set
    llm_with_name = ChatLiteLLM(
        model="gpt-4", model_name="my-deployment", api_key="fake"
    )
    params = llm_with_name._get_ls_params()
    assert params["ls_model_name"] == "my-deployment"


def test_get_ls_params_honors_per_call_model_override() -> None:
    """A per-call `model` kwarg must be reflected in traces.

    `_generate` merges per-call kwargs over the default params, so a `model`
    passed via `bind` or `invoke` is the model actually requested. Reporting
    the constructor default instead misattributes the run.
    """
    llm = ChatLiteLLM(model="gpt-4", api_key="fake")
    assert llm._get_ls_params(model="gpt-4o-mini")["ls_model_name"] == "gpt-4o-mini"

    # The override also wins over an explicitly configured model_name.
    llm_with_name = ChatLiteLLM(
        model="gpt-4", model_name="my-deployment", api_key="fake"
    )
    params = llm_with_name._get_ls_params(model="gpt-4o-mini")
    assert params["ls_model_name"] == "gpt-4o-mini"

    # Without an override the configured model_name is still used.
    assert llm_with_name._get_ls_params()["ls_model_name"] == "my-deployment"


def test_metadata_versions() -> None:
    """Test that metadata reports the correct version info."""
    llm = ChatLiteLLM(model="gpt-4", api_key="fake")
    assert llm.metadata is not None
    assert llm.metadata["lc_versions"]["langchain-litellm"] == __version__


def test_metadata_versions_preserves_user_versions() -> None:
    """Test that user-provided version metadata is preserved."""
    llm = ChatLiteLLM(
        model="gpt-4",
        api_key="fake",
        metadata={"lc_versions": {"my-app": "2.0"}},
    )
    assert llm.metadata is not None
    assert llm.metadata["lc_versions"]["my-app"] == "2.0"
    assert llm.metadata["lc_versions"]["langchain-litellm"] == __version__


def test_metadata_versions_replaces_non_dict_versions() -> None:
    """Test that invalid version metadata is replaced with a warning."""
    with pytest.warns(UserWarning, match="expected a dict"):
        llm = ChatLiteLLM(
            model="gpt-4",
            api_key="fake",
            metadata={"lc_versions": "garbage"},
        )
    assert llm.metadata is not None
    assert llm.metadata["lc_versions"]["langchain-litellm"] == __version__


def test_convert_message_to_dict_strips_thinking_blocks() -> None:
    """thinking/redacted_thinking blocks must not reach non-Anthropic providers."""

    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": "internal reasoning"},
            {"type": "redacted_thinking", "data": "encrypted"},
            {"type": "text", "text": "hello"},
        ],
        additional_kwargs={"reasoning_content": "internal reasoning"},
    )
    d = _convert_message_to_dict(msg)

    types = [block.get("type") for block in d["content"]]
    assert "thinking" not in types
    assert "redacted_thinking" not in types
    assert {"type": "text", "text": "hello"} in d["content"]
    assert d["reasoning_content"] == "internal reasoning"


def test_client_params_does_not_mutate_litellm_globals() -> None:
    """_client_params must not write instance config to litellm module globals. Fixes #132."""
    before = {
        "api_base": litellm.api_base,
        "api_key": litellm.api_key,
        "organization": getattr(litellm, "organization", None),
    }

    llm = ChatLiteLLM(
        model="azure/gpt-4o",
        api_base="https://my-azure.openai.azure.com",
        api_key="azure-key",
        organization="my-org",
        extra_headers={"X-Custom": "value"},
    )
    params = llm._client_params

    # globals must be untouched
    assert litellm.api_base == before["api_base"]
    assert litellm.api_key == before["api_key"]
    assert getattr(litellm, "organization", None) == before["organization"]
    assert getattr(litellm, "extra_headers", None) != {"X-Custom": "value"}

    # values must be present in the returned per-call params instead
    assert params["api_base"] == "https://my-azure.openai.azure.com"
    assert params["api_key"] == "azure-key"
    assert params["organization"] == "my-org"
    assert params["extra_headers"] == {"X-Custom": "value"}


def test_top_p_and_top_k_in_default_params() -> None:
    """Test that top_p and top_k are included in _default_params and _client_params."""
    llm = ChatLiteLLM(
        model="gpt-4",
        api_key="fake",
        top_p=0.8,
        top_k=40,
    )
    params = llm._default_params
    assert params["top_p"] == 0.8
    assert params["top_k"] == 40

    client_params = llm._client_params
    assert client_params["top_p"] == 0.8
    assert client_params["top_k"] == 40


def test_top_p_and_top_k_default_to_none() -> None:
    """When unset, top_p/top_k should be present but None (litellm drops them)."""
    llm = ChatLiteLLM(model="gpt-4o-mini")
    assert llm._default_params["top_p"] is None
    assert llm._default_params["top_k"] is None


# ── base_url / api_base alias ──────────────────────────────────────────────────


def test_base_url_alias_sets_api_base() -> None:
    """`base_url=` must populate `api_base`, matching the rest of the ecosystem.

    Regression for #189: previously `base_url` was silently dropped by Pydantic's
    `extra="ignore"`, so the endpoint override was never applied.
    """
    # `base_url` is a runtime alias normalized in `validate_environment`, not a
    # declared field, hence the `call-arg` ignore.
    llm = ChatLiteLLM(
        model="gpt-4o-mini",
        api_key="fake",
        base_url="https://proxy.example/v1",  # type: ignore[call-arg]
    )
    assert llm.api_base == "https://proxy.example/v1"


def test_api_base_still_supported() -> None:
    """`api_base=` must keep working for existing callers (non-breaking)."""
    llm = ChatLiteLLM(
        model="gpt-4o-mini", api_key="fake", api_base="https://legacy.example/v1"
    )
    assert llm.api_base == "https://legacy.example/v1"


def test_api_base_takes_precedence_over_base_url() -> None:
    """When both are supplied, the explicit `api_base` wins.

    Covers the precedence branch in `validate_environment` (#189): `base_url` is
    only applied when `api_base` is unset, so the canonical field always wins.
    """
    llm = ChatLiteLLM(
        model="gpt-4o-mini",
        api_key="fake",
        api_base="https://explicit.example/v1",
        base_url="https://alias.example/v1",  # type: ignore[call-arg]
    )
    assert llm.api_base == "https://explicit.example/v1"


def test_base_url_reaches_completion_call_once() -> None:
    """The configured endpoint must reach the underlying completion call once.

    Regression for #189: `base_url` is normalized to `api_base` and must be
    forwarded to `litellm.completion` as `api_base` on a single call, with the
    value unchanged (no duplication such as ``/v1/v1``).
    """
    llm = ChatLiteLLM(
        model="gpt-4o-mini",
        api_key="fake",
        base_url="https://proxy.example/v1",  # type: ignore[call-arg]
    )
    mock_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    # Patch at the litellm boundary, not at `completion_with_retry`, which is the
    # method that calls it -- otherwise the retry path is never exercised and the
    # endpoint is never seen at the point it is actually sent.
    with patch.object(
        llm.client, "completion", return_value=mock_response
    ) as mock_completion:
        llm.invoke("hi")

    mock_completion.assert_called_once()
    assert mock_completion.call_args.kwargs["api_base"] == "https://proxy.example/v1"


def test_init_chat_model_forwards_base_url() -> None:
    """The generic factory path must forward `base_url` to LiteLLM.

    `init_chat_model(model_provider="litellm", base_url=...)` is the exact path
    from #189, since its docstring lists `base_url` as the common endpoint kwarg.

    Skips (rather than fails) if a future `langchain` changes how the "litellm"
    provider resolves, so this test stays a signal about *this* package's code
    and not about the external provider registry.
    """
    # Resolve the provider first, without the kwarg under test. A failure here is
    # about the external registry, so it skips; anything raised once base_url is
    # added is this package's and must fail.
    try:
        init_chat_model("gpt-4o-mini", model_provider="litellm", api_key="fake")
    except (ImportError, ValueError) as exc:
        pytest.skip(f"init_chat_model could not resolve the litellm provider: {exc}")

    llm = init_chat_model(
        "gpt-4o-mini",
        model_provider="litellm",
        api_key="fake",
        base_url="https://proxy.example/v1",
    )

    assert isinstance(llm, ChatLiteLLM)
    assert llm.api_base == "https://proxy.example/v1"


_STREAM_MOCK_OK = {
    "choices": [
        {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def test_stream_actually_streams_without_opting_in() -> None:
    """`.stream()` must reach `_stream` for a caller who never mentioned streaming.

    `@pre_init` hands pydantic a fully-populated dict, so every field read as
    explicitly set. langchain-core treats an explicitly-set `streaming=False` as a
    hard opt-out that overrides even the `stream=True` its own `.stream()` passes,
    so `.stream()` silently fell back to `invoke()`.
    """
    llm = ChatLiteLLM(model="gpt-4o", api_key="k")
    assert "streaming" not in llm.model_fields_set
    assert llm._should_stream(async_api=False, stream=True) is True


def test_explicit_streaming_false_still_opts_out() -> None:
    """A caller who deliberately says `streaming=False` keeps the hard opt-out."""
    llm = ChatLiteLLM(model="gpt-4o", api_key="k", streaming=False)
    assert "streaming" in llm.model_fields_set
    assert llm._should_stream(async_api=False, stream=True) is False


def test_fields_set_distinguishes_a_chosen_streaming_flag_from_the_default() -> None:
    """`streaming` is the one field langchain-core reads out of `model_fields_set`.

    It treats a set `streaming=False` as a hard opt-out overriding a per-call
    `stream=True`, so the default must not look chosen.
    """
    assert "streaming" not in ChatLiteLLM(model="gpt-4o", api_key="k").model_fields_set
    assert "streaming" in ChatLiteLLM(model="gpt-4o", streaming=False).model_fields_set


@pytest.mark.parametrize(
    "field",
    ["model", "streaming", "max_retries", "model_kwargs", "disable_streaming"],
)
def test_an_explicit_none_falls_back_to_the_default(field: str) -> None:
    """A config built from JSON or os.getenv carries nulls for unset values.

    These fields are not Optional, so a None that reaches pydantic is rejected.
    Dropping it leaves the default in place and, unlike backfilling the default,
    keeps the field out of `model_fields_set` where langchain-core reads it.
    """
    kwargs: Dict[str, Any] = {"model": "anthropic/claude-3-5-sonnet-20241022"}
    kwargs[field] = None
    llm = ChatLiteLLM(**kwargs)  # type: ignore[arg-type]

    assert getattr(llm, field) == ChatLiteLLM.model_fields[field].get_default(
        call_default_factory=True
    )
    assert field not in llm.model_fields_set


def test_a_null_streaming_still_streams() -> None:
    """`streaming=None` means unset, so it must not read as a chosen opt-out.

    A type checker rejects this, which is why only config-driven callers hit it.
    """
    llm = ChatLiteLLM(model="gpt-4o", api_key="k", streaming=None)  # type: ignore[arg-type]

    assert "streaming" not in llm.model_fields_set


def test_a_validator_assigned_field_counts_as_set() -> None:
    """`base_url` reaches `api_base` through a validator, so a round-trip keeps it.

    `model_dump(exclude_unset=True)` is how a configuration is carried between
    processes; dropping a field no kwarg named loses the endpoint the caller chose.
    """
    llm = ChatLiteLLM(
        model="gpt-4",
        api_key="k",
        base_url="https://proxy.example/v1",  # type: ignore[call-arg]
    )

    config = llm.model_dump(exclude_unset=True)

    assert config["api_base"] == "https://proxy.example/v1"
    assert ChatLiteLLM(**config).api_base == "https://proxy.example/v1"


def test_stream_false_is_not_overridden_by_a_streaming_instance() -> None:
    """The non-streaming branch parses a mapping, so it must send stream=False."""
    llm = ChatLiteLLM(model="gpt-4o", api_key="k", streaming=True)

    with patch.object(
        llm.client, "completion", return_value=_STREAM_MOCK_OK
    ) as mock_completion:
        llm.invoke("hi", stream=False)

    assert mock_completion.call_args.kwargs["stream"] is False


def test_per_call_stream_options_are_not_discarded() -> None:
    """`_stream` overwrote a per-call value with the instance one or the default."""
    llm = ChatLiteLLM(model="gpt-4o", api_key="k", streaming=True)

    def _chunks(**kwargs: Any) -> Any:
        yield {
            "choices": [
                {"delta": {"role": "assistant", "content": "x"}, "finish_reason": None}
            ]
        }

    with patch.object(llm.client, "completion", side_effect=_chunks) as mock_completion:
        list(llm.stream("hi", stream_options={"include_usage": False}))

    assert mock_completion.call_args.kwargs["stream_options"] == {
        "include_usage": False
    }


@pytest.mark.asyncio
async def test_astream_false_is_not_overridden_by_a_streaming_instance() -> None:
    """The async twin of the branch that parses a mapping."""
    llm = ChatLiteLLM(model="gpt-4o", api_key="k", streaming=True)

    async def _response(**kwargs: Any) -> Any:
        return _STREAM_MOCK_OK

    with patch.object(
        llm.client, "acompletion", side_effect=_response
    ) as mock_completion:
        await llm.ainvoke("hi", stream=False)

    assert mock_completion.call_args.kwargs["stream"] is False


@pytest.mark.asyncio
async def test_per_call_stream_options_are_not_discarded_on_the_async_path() -> None:
    """`_astream` carries the same caller configuration as `_stream`."""
    llm = ChatLiteLLM(model="gpt-4o", api_key="k", streaming=True)

    async def _chunks(**kwargs: Any) -> Any:
        async def _aiter() -> Any:
            yield {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "x"},
                        "finish_reason": None,
                    }
                ]
            }

        return _aiter()

    with patch.object(
        llm.client, "acompletion", side_effect=_chunks
    ) as mock_completion:
        async for _ in llm.astream("hi", stream_options={"include_usage": False}):
            pass

    assert mock_completion.call_args.kwargs["stream_options"] == {
        "include_usage": False
    }


def test_credentials_are_not_shown_in_repr() -> None:
    """A key in repr() reaches logs and tracebacks."""
    llm = ChatLiteLLM(
        model="gpt-4o",
        api_key="sk-generic",
        openai_api_key="sk-openai",
        azure_api_key="sk-azure",
        anthropic_api_key="sk-anthropic",
        replicate_api_key="sk-replicate",
        cohere_api_key="sk-cohere",
        openrouter_api_key="sk-openrouter",
    )
    assert "sk-" not in repr(llm)


def test_every_credential_field_is_kept_out_of_repr() -> None:
    """A provider added later must not arrive without the same protection."""
    for name, field in ChatLiteLLM.model_fields.items():
        if name in ("api_key", "extra_headers") or name.endswith("_api_key"):
            assert field.repr is False, name


def test_a_token_in_extra_headers_is_not_shown_in_repr() -> None:
    """`extra_headers` is how a caller reaches a gateway, so it carries a token."""
    llm = ChatLiteLLM(
        model="gpt-4o",
        extra_headers={"Authorization": "Bearer sk-should-not-appear"},
    )
    assert "sk-should-not-appear" not in repr(llm)


def test_copying_model_kwargs_preserves_the_container_type() -> None:
    """A caller's defaultdict must not come back as a plain dict.

    `model_kwargs` is the escape hatch for provider payloads, so the type the
    caller chose is part of the value.
    """
    llm = ChatLiteLLM(
        model="gpt-4o",
        api_key="k",
        model_kwargs={"cfg": defaultdict(list), "ord": OrderedDict(b=2)},
    )

    params = llm._default_params

    assert isinstance(params["cfg"], defaultdict)
    assert params["cfg"]["missing"] == []
    assert isinstance(params["ord"], OrderedDict)


def test_a_self_referential_model_kwarg_does_not_recurse_forever() -> None:
    """Copying has to terminate on a value that contains itself."""
    cyclic: Dict[str, Any] = {}
    cyclic["self"] = cyclic
    llm = ChatLiteLLM(model="gpt-4o", api_key="k", model_kwargs={"c": cyclic})

    copied = llm._default_params["c"]

    assert copied is not cyclic
    assert copied["self"] is copied


def test_a_non_mapping_input_raises_a_validation_error() -> None:
    """The validator's guard must hand pydantic the bad input, not crash inside it."""
    with pytest.raises(ValidationError):
        ChatLiteLLM.model_validate([1, 2])


def test_client_params_does_not_alias_model_kwargs() -> None:
    """A caller mutating the returned params must not reach back into the model."""
    llm = ChatLiteLLM(
        model="gpt-4o",
        api_key="k",
        model_kwargs={"top": {"nested": {"a": 1}}, "items": [{"b": 2}]},
    )
    params = llm._client_params
    params["top"]["nested"]["a"] = 999
    params["items"][0]["b"] = 999

    # Copying only the first level would leave both of these aliased.
    assert llm.model_kwargs["top"]["nested"]["a"] == 1
    assert llm.model_kwargs["items"][0]["b"] == 2


def test_constructor_signature_is_not_erased(tmp_path: Path) -> None:
    """Nothing may replace pydantic's synthesized `__init__`.

    An override taking `**kwargs` silently stops type checkers flagging an unknown
    or mistyped field, and only a type checker can see it: the two are identical at
    runtime.
    """
    pytest.importorskip("mypy")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from langchain_litellm import ChatLiteLLM\n"
        "ChatLiteLLM(not_a_real_field=1)\n"
        "ChatLiteLLM(temperature='warm')\n"
    )

    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-incremental", str(probe)],
        capture_output=True,
        text=True,
    )

    assert "call-arg" in result.stdout, result.stdout
    assert "arg-type" in result.stdout, result.stdout
