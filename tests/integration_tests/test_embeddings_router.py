"""Integration tests for LiteLLMEmbeddingsRouter."""

from langchain_tests.integration_tests import EmbeddingsIntegrationTests
from litellm import Router

from langchain_litellm.embeddings import LiteLLMEmbeddingsRouter


class TestLiteLLMEmbeddingsRouterIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[LiteLLMEmbeddingsRouter]:
        return LiteLLMEmbeddingsRouter

    @property
    def embedding_model_params(self) -> dict:
        return {
            "router": Router(
                model_list=[
                    {
                        "model_name": "openai/text-embedding-3-small",
                        "litellm_params": {
                            "model": "openai/text-embedding-3-small",
                        },
                    },
                ]
            ),
        }
