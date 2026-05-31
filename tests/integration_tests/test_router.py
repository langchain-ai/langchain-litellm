"""Test ChatLiteLLMRouter chat model."""

from typing import Type

from langchain_tests.integration_tests import ChatModelIntegrationTests
from litellm import Router

from langchain_litellm.chat_models import ChatLiteLLMRouter


class TestChatLiteLLMRouterIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatLiteLLMRouter]:
        return ChatLiteLLMRouter

    @property
    def chat_model_params(self) -> dict:
        return {
            "router": Router(
                model_list=[
                    {
                        "model_name": "openai/gpt-4o-mini",
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                        },
                    },
                ]
            ),
        }

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        return False

    @property
    def has_structured_output(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def returns_usage_metadata(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        return False
