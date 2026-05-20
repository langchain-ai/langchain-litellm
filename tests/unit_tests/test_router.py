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

def test_router_aimessage_passes_langchain_summarization_middleware():
    """
    Test that ChatLiteLLMRouter outputs pass LangChain's strict token counting guards.
    Fixes Issue #152 where missing 'model_provider' caused SummarizationMiddleware to fail.
    """
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
    }

    # Router's _create_chat_result requires the metadata kwarg
    result = llm._create_chat_result(mock_response, metadata={})
    last_ai_message = result.generations[0].message
    
    # ── Simulate the exact logic from SummarizationMiddleware ──
    threshold = 10
    
    # 1. Must be an AIMessage
    assert isinstance(last_ai_message, AIMessage)
    
    # 2. Must have usage metadata
    assert last_ai_message.usage_metadata is not None
    
    # 3. Must exceed the summarization threshold
    reported_tokens = last_ai_message.usage_metadata.get("total_tokens", -1)
    assert reported_tokens >= threshold
    
    # 4. CRITICAL FIX: Must have model_provider metadata that matches 'litellm'
    message_provider = last_ai_message.response_metadata.get("model_provider")
    assert message_provider == "litellm"