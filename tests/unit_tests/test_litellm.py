"""Test chat model integration."""

# stdlib
import logging
from typing import Any, Dict, Optional, Union
from unittest.mock import patch

# third-party
import litellm
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableLambda
from litellm.types.utils import ChatCompletionDeltaToolCall, Delta, Function
from pydantic import BaseModel

# first-party
from langchain_litellm._version import __version__
from langchain_litellm.chat_models import ChatLiteLLM
from langchain_litellm.chat_models.litellm import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
    _create_usage_metadata,
    _inject_reasoning_content_into_content,
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


# ── reasoning content injection ────────────────────────────────────────────────


def test_inject_reasoning_content_into_string_content() -> None:
    result = _inject_reasoning_content_into_content("answer", "hidden chain")

    assert result == [
        {"type": "thinking", "thinking": "hidden chain"},
        {"type": "text", "text": "answer"},
    ]


def test_inject_reasoning_content_into_empty_content() -> None:
    result = _inject_reasoning_content_into_content("", "hidden chain")

    assert result == [{"type": "thinking", "thinking": "hidden chain"}]


def test_inject_reasoning_content_prepends_for_list_without_thinking() -> None:
    content = [{"type": "text", "text": "answer"}]

    result = _inject_reasoning_content_into_content(content, "hidden chain")

    assert result == [
        {"type": "thinking", "thinking": "hidden chain"},
        {"type": "text", "text": "answer"},
    ]


def test_inject_reasoning_content_does_not_duplicate_existing_thinking() -> None:
    content = [
        {"type": "thinking", "thinking": "already there"},
        {"type": "text", "text": "answer"},
    ]

    result = _inject_reasoning_content_into_content(content, "hidden chain")

    assert result == content


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
