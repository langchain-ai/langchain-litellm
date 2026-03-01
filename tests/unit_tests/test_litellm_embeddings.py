"""Unit tests for LiteLLMEmbeddings."""

from typing import Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_tests.unit_tests import EmbeddingsUnitTests
from pydantic import ValidationError

from langchain_litellm.embeddings import LiteLLMEmbeddings
from tests.utils import mock_embedding_response


class TestLiteLLMEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[LiteLLMEmbeddings]:
        return LiteLLMEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "model": "openai/text-embedding-3-small",
            "api_key": "fake-key",
        }


class TestLiteLLMEmbeddingsParams:
    def test_default_params(self):
        """Test default parameter values."""
        embeddings = LiteLLMEmbeddings(api_key="fake")
        assert embeddings.model == "openai/text-embedding-3-small"
        assert embeddings.max_retries == 1
        assert embeddings.api_base is None

    def test_custom_params(self):
        """Test custom parameter passthrough."""
        embeddings = LiteLLMEmbeddings(
            model="cohere/embed-english-v3.0",
            api_key="fake-key",
            api_base="https://custom.endpoint.com",
            dimensions=256,
            request_timeout=30.0,
        )
        params = embeddings._get_litellm_params()
        assert params["model"] == "cohere/embed-english-v3.0"
        assert params["api_key"] == "fake-key"
        assert params["api_base"] == "https://custom.endpoint.com"
        assert params["dimensions"] == 256
        assert params["timeout"] == 30.0

    def test_none_params_excluded(self):
        """Test that None-valued params are excluded from the litellm call."""
        embeddings = LiteLLMEmbeddings(model="openai/text-embedding-3-small", api_key="fake")
        params = embeddings._get_litellm_params()
        assert "api_base" not in params
        assert "api_version" not in params
        assert "dimensions" not in params

    def test_model_kwargs_merged(self):
        """Test that model_kwargs are merged into params."""
        embeddings = LiteLLMEmbeddings(
            api_key="fake",
            model_kwargs={"user": "test-user"},
        )
        params = embeddings._get_litellm_params()
        assert params["user"] == "test-user"

    def test_explicit_params_override_model_kwargs(self):
        """Test that explicit params take precedence over model_kwargs."""
        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="explicit-key",
            model_kwargs={"api_key": "kwargs-key"},
        )
        params = embeddings._get_litellm_params()
        assert params["api_key"] == "explicit-key"

    def test_encoding_format_rejects_base64(self):
        """Test that encoding_format='base64' is rejected by validation."""
        with pytest.raises(ValidationError):
            LiteLLMEmbeddings(api_key="fake", encoding_format="base64")

    def test_encoding_format_accepts_float(self):
        """Test that encoding_format='float' is accepted."""
        embeddings = LiteLLMEmbeddings(api_key="fake", encoding_format="float")
        assert embeddings.encoding_format == "float"

    @patch("litellm.embedding")
    def test_embed_documents(self, mock_embedding):
        """Test embed_documents calls litellm.embedding correctly."""
        mock_embedding.return_value = mock_embedding_response(["hello", "world"])

        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="fake-key",
        )
        result = embeddings.embed_documents(["hello", "world"])

        mock_embedding.assert_called_once()
        call_kwargs = mock_embedding.call_args[1]
        assert call_kwargs["input"] == ["hello", "world"]
        assert call_kwargs["model"] == "openai/text-embedding-3-small"
        assert call_kwargs["api_key"] == "fake-key"
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @patch("litellm.embedding")
    def test_embed_query(self, mock_embedding):
        """Test embed_query calls litellm.embedding with a single-item list."""
        mock_embedding.return_value = mock_embedding_response(["hello"])

        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="fake-key",
        )
        result = embeddings.embed_query("hello")

        mock_embedding.assert_called_once()
        call_kwargs = mock_embedding.call_args[1]
        assert call_kwargs["input"] == ["hello"]
        assert result == [0.1, 0.2, 0.3]

    @patch("litellm.aembedding", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_aembed_documents(self, mock_aembedding):
        """Test aembed_documents calls litellm.aembedding correctly."""
        mock_aembedding.return_value = mock_embedding_response(["hello", "world"])

        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="fake-key",
        )
        result = await embeddings.aembed_documents(["hello", "world"])

        mock_aembedding.assert_called_once()
        call_kwargs = mock_aembedding.call_args[1]
        assert call_kwargs["input"] == ["hello", "world"]
        assert call_kwargs["model"] == "openai/text-embedding-3-small"
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @patch("litellm.aembedding", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_aembed_query(self, mock_aembedding):
        """Test aembed_query calls litellm.aembedding with a single-item list."""
        mock_aembedding.return_value = mock_embedding_response(["hello"])

        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="fake-key",
        )
        result = await embeddings.aembed_query("hello")

        mock_aembedding.assert_called_once()
        call_kwargs = mock_aembedding.call_args[1]
        assert call_kwargs["input"] == ["hello"]
        assert result == [0.1, 0.2, 0.3]

    def test_embed_documents_empty_list(self):
        """Test that embed_documents returns [] for empty input without API call."""
        embeddings = LiteLLMEmbeddings(api_key="fake")
        result = embeddings.embed_documents([])
        assert result == []

    @pytest.mark.asyncio
    async def test_aembed_documents_empty_list(self):
        """Test that aembed_documents returns [] for empty input without API call."""
        embeddings = LiteLLMEmbeddings(api_key="fake")
        result = await embeddings.aembed_documents([])
        assert result == []

    @patch("litellm.embedding")
    def test_embed_documents_with_input_type(self, mock_embedding):
        """Test that document_input_type is passed for embed_documents."""
        mock_embedding.return_value = mock_embedding_response(["hello"])

        embeddings = LiteLLMEmbeddings(
            model="cohere/embed-english-v3.0",
            api_key="fake-key",
            document_input_type="search_document",
            query_input_type="search_query",
        )
        embeddings.embed_documents(["hello"])

        call_kwargs = mock_embedding.call_args[1]
        assert call_kwargs["input_type"] == "search_document"

    @patch("litellm.embedding")
    def test_embed_query_with_input_type(self, mock_embedding):
        """Test that query_input_type is passed for embed_query."""
        mock_embedding.return_value = mock_embedding_response(["hello"])

        embeddings = LiteLLMEmbeddings(
            model="cohere/embed-english-v3.0",
            api_key="fake-key",
            document_input_type="search_document",
            query_input_type="search_query",
        )
        embeddings.embed_query("hello")

        call_kwargs = mock_embedding.call_args[1]
        assert call_kwargs["input_type"] == "search_query"

    @patch("litellm.embedding")
    def test_no_input_type_when_unset(self, mock_embedding):
        """Test that input_type is not passed when not configured."""
        mock_embedding.return_value = mock_embedding_response(["hello"])

        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="fake-key",
        )
        embeddings.embed_documents(["hello"])

        call_kwargs = mock_embedding.call_args[1]
        assert "input_type" not in call_kwargs
