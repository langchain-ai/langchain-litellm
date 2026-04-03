"""Integration tests for LiteLLMEmbeddingsRouter."""

from typing import Type

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_litellm.embeddings import LiteLLMEmbeddingsRouter
from tests.utils import test_embedding_router


class TestLiteLLMEmbeddingsRouterIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[LiteLLMEmbeddingsRouter]:
        return LiteLLMEmbeddingsRouter

    @property
    def embedding_model_params(self) -> dict:
        return {
            "router": test_embedding_router(),
        }
