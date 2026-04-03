"""Test chat model integration."""

from typing import Type

from langchain_core.messages import AIMessage
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_litellm.chat_models import ChatLiteLLMRouter
from tests.utils import test_router


class TestChatLiteLLMRouterUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatLiteLLMRouter]:
        return ChatLiteLLMRouter

    @property
    def chat_model_params(self) -> dict:
        return {
            "router": test_router(),
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

    def test_router_provider_specific_fields_in_chat_result(self):
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

    def test_router_create_chat_result_sets_usage_metadata(self):
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

    def test_router_stream_options_set_for_all_providers(self):
        """Router _stream must set stream_options for non-OpenAI providers."""
        router = test_router()
        llm = ChatLiteLLMRouter(router=router)
        stream_options = (
            llm.stream_options
            if llm.stream_options is not None
            else {"include_usage": True}
        )
        assert stream_options == {"include_usage": True}
