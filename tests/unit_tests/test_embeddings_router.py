"""Unit tests for LiteLLMEmbeddingsRouter."""

from typing import Any, Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_litellm.embeddings import LiteLLMEmbeddingsRouter
from tests.utils import make_embedding_router, mock_embedding_response


class TestLiteLLMEmbeddingsRouterUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[LiteLLMEmbeddingsRouter]:
        return LiteLLMEmbeddingsRouter

    @property
    def embedding_model_params(self) -> dict:
        return {
            "router": make_embedding_router(),
        }


class TestLiteLLMEmbeddingsRouterParams:
    def test_router_stored(self):
        """Test that the router instance is stored."""
        router = make_embedding_router()
        embeddings = LiteLLMEmbeddingsRouter(router=router)
        assert embeddings.router is router

    def test_router_params_exclude_none(self):
        """Test that None-valued params are excluded from router calls."""
        router = make_embedding_router()
        embeddings = LiteLLMEmbeddingsRouter(router=router)
        params = embeddings._get_router_params()
        assert "timeout" not in params
        assert "dimensions" not in params
        assert "model" in params

    def test_base_url_alias_sets_api_base(self):
        """Test that router embeddings accept base_url like LiteLLMEmbeddings."""
        router = make_embedding_router()
        embeddings = LiteLLMEmbeddingsRouter(
            router=router,
            base_url="https://proxy.example/v1",  # type: ignore[call-arg]
        )
        assert embeddings.api_base == "https://proxy.example/v1"

    def test_api_base_takes_precedence_over_base_url(self):
        """Test that api_base wins when both endpoint names are supplied."""
        router = make_embedding_router()
        embeddings = LiteLLMEmbeddingsRouter(
            router=router,
            api_base="https://explicit.example/v1",
            base_url="https://alias.example/v1",  # type: ignore[call-arg]
        )
        assert embeddings.api_base == "https://explicit.example/v1"

    def test_embed_documents_uses_router(self):
        """Test that embed_documents delegates to router.embedding()."""
        router = MagicMock()
        router.embedding.return_value = mock_embedding_response(["hello", "world"])

        embeddings = LiteLLMEmbeddingsRouter(
            router=router, model="text-embedding-3-small"
        )
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

        embeddings = LiteLLMEmbeddingsRouter(
            router=router, model="text-embedding-3-small"
        )
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

        embeddings = LiteLLMEmbeddingsRouter(
            router=router, model="text-embedding-3-small"
        )
        result = await embeddings.aembed_documents(["hello", "world"])

        router.aembedding.assert_called_once()
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_aembed_query_uses_router(self):
        """Test that aembed_query delegates to router.aembedding()."""
        router = MagicMock()
        router.aembedding = AsyncMock(return_value=mock_embedding_response(["hello"]))

        embeddings = LiteLLMEmbeddingsRouter(
            router=router, model="text-embedding-3-small"
        )
        result = await embeddings.aembed_query("hello")

        router.aembedding.assert_called_once()
        assert result == [0.1, 0.2, 0.3]


def _one_deployment_router() -> Any:
    import litellm

    return litellm.Router(
        model_list=[
            {
                "model_name": "emb-small",
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_key": "sk-deployment",
                },
            }
        ]
    )


def test_embeddings_router_honours_max_retries() -> None:
    """The embed methods called router.embedding directly, bypassing the decorator.

    Same defect ChatLiteLLMRouter had: the inherited `max_retries` had no effect.
    """
    import litellm

    embeddings = LiteLLMEmbeddingsRouter(router=_one_deployment_router(), max_retries=4)

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise litellm.RateLimitError(
            message="rate limited", llm_provider="openai", model="x"
        )

    with patch.object(
        embeddings.router, "embedding", side_effect=_raise
    ) as mock_embedding:
        with patch("time.sleep", return_value=None):
            with pytest.raises(litellm.RateLimitError):
                embeddings.embed_query("hi")

    assert mock_embedding.call_count == 4


def test_embeddings_router_defaults_its_model_from_the_router() -> None:
    """ChatLiteLLMRouter does this; the embeddings router left `model` unset."""
    embeddings = LiteLLMEmbeddingsRouter(router=_one_deployment_router())
    assert embeddings.model == "emb-small"


def test_embeddings_router_forwards_only_an_explicit_api_key() -> None:
    """Each deployment owns its endpoint, so the connector's must not override it."""

    class _Response:
        data = [{"embedding": [0.1]}]

    captured: dict = {}

    def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _Response()

    embeddings = LiteLLMEmbeddingsRouter(
        router=_one_deployment_router(), api_key="sk-explicit"
    )
    with patch.object(embeddings.router, "embedding", side_effect=_capture):
        embeddings.embed_query("hi")

    assert captured["api_key"] == "sk-explicit"
    assert captured.get("api_base") is None
