"""Test ChatLiteLLMRouter chat model."""

from typing import Type

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_litellm.chat_models import ChatLiteLLMRouter
from tests.utils import test_router


class TestChatLiteLLMRouterIntegration(ChatModelIntegrationTests):
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
