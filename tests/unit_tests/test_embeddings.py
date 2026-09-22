"""Unit tests for LiteLLMEmbeddings."""

from typing import Type
from unittest.mock import AsyncMock, patch

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

    def test_base_url_alias_sets_api_base(self):
        """Test that base_url is accepted as an alias for api_base."""
        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="fake-key",
            base_url="https://proxy.example/v1",  # type: ignore[call-arg]
        )
        assert embeddings.api_base == "https://proxy.example/v1"

    def test_api_base_takes_precedence_over_base_url(self):
        """Test that api_base wins when both endpoint names are supplied."""
        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="fake-key",
            api_base="https://explicit.example/v1",
            base_url="https://alias.example/v1",  # type: ignore[call-arg]
        )
        assert embeddings.api_base == "https://explicit.example/v1"

    def test_base_url_forwarded_to_litellm_params_once(self):
        """Test that the alias reaches litellm.embedding as api_base."""
        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small",
            api_key="fake-key",
            base_url="https://proxy.example/v1",  # type: ignore[call-arg]
        )
        params = embeddings._get_litellm_params()
        assert params["api_base"] == "https://proxy.example/v1"
        assert params["api_base"].count("/v1") == 1

    def test_none_params_excluded(self):
        """Test that None-valued params are excluded from the litellm call."""
        embeddings = LiteLLMEmbeddings(
            model="openai/text-embedding-3-small", api_key="fake"
        )
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


def test_unknown_constructor_kwargs_are_rejected() -> None:
    """A credential the caller believes is set must never vanish silently.

    `LiteLLMEmbeddings` has no provider-scoped `*_api_key` fields, so a name like
    `openai_api_key` was accepted by pydantic and then dropped. Provider-specific
    values belong in `model_kwargs`.
    """
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        LiteLLMEmbeddings(
            model="text-embedding-3-small",
            openai_api_key="sk-openai",  # type: ignore[call-arg]
        )

    # A declared field is of course still accepted.
    assert LiteLLMEmbeddings(model="text-embedding-3-small", api_key="sk-x").api_key


def test_rejecting_an_unknown_kwarg_does_not_echo_its_value() -> None:
    """The rejection must not print the credential it was protecting.

    Pydantic's own `extra_forbidden` error carries `input_value`, so refusing a
    misspelled credential would put it in the traceback verbatim.
    """
    with pytest.raises(ValueError) as caught:
        LiteLLMEmbeddings(
            model="text-embedding-3-small",
            openai_api_key="sk-should-not-appear",  # type: ignore[call-arg]
        )

    assert "openai_api_key" in str(caught.value)
    assert "sk-should-not-appear" not in str(caught.value)


def test_embeddings_credentials_are_not_shown_in_repr() -> None:
    """The repr protection must cover this class too; the router inherits it."""
    assert "sk-should-not-appear" not in repr(
        LiteLLMEmbeddings(
            model="text-embedding-3-small", api_key="sk-should-not-appear"
        )
    )


def test_every_embeddings_credential_field_is_kept_out_of_repr() -> None:
    """A credential added later must not arrive without the same protection."""
    for name, field in LiteLLMEmbeddings.model_fields.items():
        if name in ("api_key", "extra_headers") or name.endswith("_api_key"):
            assert field.repr is False, name


def test_embeddings_token_in_extra_headers_is_not_shown_in_repr() -> None:
    """`extra_headers` is how a caller reaches a gateway, so it carries a token."""
    assert "sk-should-not-appear" not in repr(
        LiteLLMEmbeddings(
            model="text-embedding-3-small",
            extra_headers={"Authorization": "Bearer sk-should-not-appear"},
        )
    )
