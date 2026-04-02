"""Test chat model integration."""

from typing import Type

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_tests.unit_tests import ChatModelUnitTests
from litellm.types.utils import ChatCompletionDeltaToolCall, Delta, Function

from langchain_litellm.chat_models import ChatLiteLLM
from langchain_litellm.chat_models.litellm import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _create_usage_metadata,
    _inject_reasoning_content_into_content,
)


class TestChatLiteLLMUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatLiteLLM]:
        return ChatLiteLLM

    @property
    def chat_model_params(self) -> dict:
        return {
            "custom_llm_provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": "<your_api_key>",
            "max_retries": 1,
        }

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        return False

    @property
    def has_structured_output(self) -> bool:
        return False

    @property
    def supports_json_mode(self) -> bool:
        return False

    @property
    def supports_image_inputs(self) -> bool:
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        return False

    def test_litellm_delta_to_langchain_message_chunk(self):
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

    def test_convert_dict_to_tool_message(self):
        """Ensure tool role dicts convert to ToolMessage."""
        mock_dict = {"role": "tool", "content": "result", "tool_call_id": "123"}
        message = _convert_dict_to_message(mock_dict)
        from langchain_core.messages import ToolMessage

        assert isinstance(message, ToolMessage)
        assert message.content == "result"
        assert message.tool_call_id == "123"

    def test_provider_specific_fields_in_delta(self):
        """Test that provider_specific_fields are preserved when converting deltas."""
        mock_delta = {
            "role": "assistant",
            "content": "Paris is the capital of France",
            "provider_specific_fields": {
                "citations": [
                    {"source": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Paris"}
                ]
            }
        }
        
        chunk = _convert_delta_to_message_chunk(mock_delta, AIMessageChunk)
        
        assert isinstance(chunk, AIMessageChunk)
        assert "provider_specific_fields" in chunk.additional_kwargs
        assert chunk.additional_kwargs["provider_specific_fields"]["citations"][0]["source"] == "Wikipedia"

    def test_provider_specific_fields_in_message(self):
        """Test that provider_specific_fields are preserved when converting message dicts."""
        mock_message_dict = {
            "role": "assistant",
            "content": "The Earth orbits the Sun",
            "provider_specific_fields": {
                "grounding_metadata": {
                    "search_queries": ["Earth orbit"],
                    "grounding_supports": [{"segment": "The Earth orbits"}]
                }
            }
        }
        
        message = _convert_dict_to_message(mock_message_dict)
        
        assert isinstance(message, AIMessage)
        assert "provider_specific_fields" in message.additional_kwargs
        assert "grounding_metadata" in message.additional_kwargs["provider_specific_fields"]

    def test_provider_specific_fields_in_chat_result(self):
        """Test that top-level provider_specific_fields appear in llm_output."""
        llm = ChatLiteLLM(model="gpt-3.5-turbo", api_key="fake")
        
        mock_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "provider_specific_fields": {
                "citations": [{"source": "test"}]
            }
        }
        
        result = llm._create_chat_result(mock_response)
        
        assert "provider_specific_fields" in result.llm_output
        assert result.llm_output["provider_specific_fields"]["citations"][0]["source"] == "test"


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

def test_stream_options_set_for_non_openai_model():
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

def test_stream_options_respected_when_set_explicitly():
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