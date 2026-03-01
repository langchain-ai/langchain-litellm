"""Unit tests for LiteLLMEmbeddingsRouter."""

from typing import Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_litellm.embeddings import LiteLLMEmbeddingsRouter
from tests.utils import test_embedding_router, mock_embedding_response


class TestLiteLLMEmbeddingsRouterUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[LiteLLMEmbeddingsRouter]:
        return LiteLLMEmbeddingsRouter

    @property
    def embedding_model_params(self) -> dict:
        return {
            "router": test_embedding_router(),
        }


class TestLiteLLMEmbeddingsRouterParams:
    def test_router_stored(self):
        """Test that the router instance is stored."""
        router = test_embedding_router()
        embeddings = LiteLLMEmbeddingsRouter(router=router)
        assert embeddings.router is router

    def test_router_params_exclude_none(self):
        """Test that None-valued params are excluded from router calls."""
        router = test_embedding_router()
        embeddings = LiteLLMEmbeddingsRouter(router=router)
        params = embeddings._get_router_params()
        assert "timeout" not in params
        assert "dimensions" not in params
        assert "model" in params

    def test_embed_documents_uses_router(self):
        """Test that embed_documents delegates to router.embedding()."""
        router = MagicMock()
        router.embedding.return_value = mock_embedding_response(["hello", "world"])

        embeddings = LiteLLMEmbeddingsRouter(router=router, model="text-embedding-3-small")
        result = embeddings.embed_documents(["hello", "world"])

        router.embedding.assert_called_once()
        call_kwargs = router.embedding.call_args[1]
        assert call_kwargs["input"] == ["hello", "world"]
        assert call_kwargs["model"] == "text-embedding-3-small"
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    def test_embed_query_uses_router(self):
        """Test that embed_query delegates to router.embedding()."""
        router = MagicMock()
        router.embedding.return_value = mock_embedding_response(["hello"])

        embeddings = LiteLLMEmbeddingsRouter(router=router, model="text-embedding-3-small")
        result = embeddings.embed_query("hello")

        router.embedding.assert_called_once()
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_aembed_documents_uses_router(self):
        """Test that aembed_documents delegates to router.aembedding()."""
        router = MagicMock()
        router.aembedding = AsyncMock(
            return_value=mock_embedding_response(["hello", "world"])
        )

        embeddings = LiteLLMEmbeddingsRouter(router=router, model="text-embedding-3-small")
        result = await embeddings.aembed_documents(["hello", "world"])

        router.aembedding.assert_called_once()
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_aembed_query_uses_router(self):
        """Test that aembed_query delegates to router.aembedding()."""
        router = MagicMock()
        router.aembedding = AsyncMock(
            return_value=mock_embedding_response(["hello"])
        )

        embeddings = LiteLLMEmbeddingsRouter(router=router, model="text-embedding-3-small")
        result = await embeddings.aembed_query("hello")

        router.aembedding.assert_called_once()
        assert result == [0.1, 0.2, 0.3]
