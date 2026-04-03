"""Test ChatLiteLLM chat model."""

from typing import Type

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_litellm.chat_models import ChatLiteLLM


class TestChatLiteLLMIntegration(ChatModelIntegrationTests):
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
