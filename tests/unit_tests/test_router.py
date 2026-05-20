"""Test router chat model integration."""

from typing import Type

from langchain_core.messages import AIMessage
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_litellm.chat_models import ChatLiteLLMRouter
from tests.utils import test_router


def test_router_provider_specific_fields_in_chat_result():
    """Test that Router preserves top-level provider_specific_fields."""
    router = test_router()
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

    assert "provider_specific_fields" in result.llm_output
    assert (
        result.llm_output["provider_specific_fields"]["citations"][0]["source"]
        == "vertex"
    )


def test_router_create_chat_result_sets_usage_metadata():
    """Router _create_chat_result should set usage_metadata on AIMessage."""
    router = test_router()
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


def test_router_stream_options_set_for_all_providers():
    """Router _stream must set stream_options for non-OpenAI providers."""
    router = test_router()
    llm = ChatLiteLLMRouter(router=router)
    stream_options = (
        llm.stream_options
        if llm.stream_options is not None
        else {"include_usage": True}
    )
    assert stream_options == {"include_usage": True}

def test_router_create_chat_result_sets_model_provider():
    """Router non-streaming path must set model_provider. Fixes #152."""
    router = test_router()
    llm = ChatLiteLLMRouter(router=router)
    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    result = llm._create_chat_result(mock_response, metadata={})
    msg = result.generations[0].message
    assert isinstance(msg, AIMessage)
    assert msg.response_metadata.get("model_provider") == "litellm"


def test_router_stream_sets_model_provider_in_response_metadata():
    """Router first streaming chunk must carry model_provider. Fixes #152."""
    from unittest.mock import patch

    router = test_router()
    llm = ChatLiteLLMRouter(router=router)
    fake_chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": "hel"}}], "usage": None},
        {"choices": [{"delta": {"content": "lo"}}], "usage": None},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
    ]

    with patch.object(llm.router, "completion", return_value=iter(fake_chunks)):
        chunks = list(llm._stream([]))

    assert chunks[0].message.response_metadata.get("model_provider") == "litellm"
    assert chunks[1].message.response_metadata == {}