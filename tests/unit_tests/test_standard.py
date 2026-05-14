"""Chat model standard unit tests.

From LangChain's standard test suite.
"""

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
