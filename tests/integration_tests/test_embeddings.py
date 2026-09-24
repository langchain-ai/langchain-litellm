"""Integration tests for LiteLLMEmbeddings."""

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_litellm.embeddings import LiteLLMEmbeddings


class TestLiteLLMEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[LiteLLMEmbeddings]:
        return LiteLLMEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "model": "openai/text-embedding-3-small",
        }
