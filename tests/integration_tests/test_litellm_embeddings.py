"""Integration tests for LiteLLMEmbeddings."""

from typing import Type

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_litellm.embeddings import LiteLLMEmbeddings


class TestLiteLLMEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[LiteLLMEmbeddings]:
        return LiteLLMEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "model": "openai/text-embedding-3-small",
            "api_key": "<your_api_key>",
        }
