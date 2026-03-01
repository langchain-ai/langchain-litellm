"""Wrapper around LiteLLM's embedding API."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    """Return a tenacity retry decorator for transient litellm errors."""
    import litellm
    from langchain_core.language_models.llms import create_base_retry_decorator

    errors = [
        litellm.Timeout,
        litellm.APIError,
        litellm.APIConnectionError,
        litellm.RateLimitError,
    ]
    return create_base_retry_decorator(error_types=errors, max_retries=max_retries)


class LiteLLMEmbeddings(BaseModel, Embeddings):
    """LiteLLM embedding model.

    Uses `litellm.embedding()` to support 100+ providers through a unified
    interface. All provider configuration (api_key, api_base, etc.) can be
    passed explicitly—no environment variables required.

    Example:
        .. code-block:: python

            from langchain_litellm import LiteLLMEmbeddings

            embeddings = LiteLLMEmbeddings(
                model="openai/text-embedding-3-small",
                api_key="sk-...",
            )
            vectors = embeddings.embed_documents(["hello", "world"])

    For providers that distinguish document vs query embeddings (Cohere,
    Voyage, Vertex AI, etc.), set ``document_input_type`` and
    ``query_input_type``:

        .. code-block:: python

            embeddings = LiteLLMEmbeddings(
                model="cohere/embed-english-v3.0",
                api_key="...",
                document_input_type="search_document",
                query_input_type="search_query",
            )
    """

    model: str = "openai/text-embedding-3-small"
    """Model name in litellm format (e.g. 'openai/text-embedding-3-small',
    'cohere/embed-english-v3.0', 'bedrock/amazon.titan-embed-text-v1')."""

    api_key: Optional[str] = None
    """API key for the provider."""

    api_base: Optional[str] = None
    """Base URL for the API endpoint."""

    api_version: Optional[str] = None
    """API version (e.g. for Azure)."""

    custom_llm_provider: Optional[str] = None
    """Override the litellm provider routing."""

    organization: Optional[str] = None
    """Organization ID (e.g. for OpenAI)."""

    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for API requests."""

    max_retries: int = 1
    """Maximum number of retries on transient errors (Timeout, APIError,
    APIConnectionError, RateLimitError)."""

    extra_headers: Optional[Dict[str, str]] = None
    """Extra headers to include in the request."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Additional model parameters passed to litellm.embedding()."""

    dimensions: Optional[int] = None
    """Output embedding dimensions (if supported by the model)."""

    encoding_format: Optional[Literal["float"]] = None
    """Encoding format for the embeddings. Only 'float' is supported;
    'base64' is not supported as it would return strings instead of floats."""

    document_input_type: Optional[str] = None
    """Input type to send when embedding documents (e.g. 'search_document'
    for Cohere, 'RETRIEVAL_DOCUMENT' for Vertex AI). When set,
    ``embed_documents`` passes this as ``input_type``."""

    query_input_type: Optional[str] = None
    """Input type to send when embedding queries (e.g. 'search_query'
    for Cohere, 'RETRIEVAL_QUERY' for Vertex AI). When set,
    ``embed_query`` passes this as ``input_type``."""

    def _get_litellm_params(
        self, *, input_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build parameter dict for litellm.embedding(), excluding None values."""
        params: Dict[str, Any] = {
            **self.model_kwargs,
            "model": self.model,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "custom_llm_provider": self.custom_llm_provider,
            "organization": self.organization,
            "timeout": self.request_timeout,
            "extra_headers": self.extra_headers,
            "dimensions": self.dimensions,
            "encoding_format": self.encoding_format,
            "input_type": input_type,
        }
        return {k: v for k, v in params.items() if v is not None}

    def _embedding_with_retry(self, **kwargs: Any) -> Any:
        """Call litellm.embedding with retry on transient errors."""
        import litellm

        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        def _embed() -> Any:
            return litellm.embedding(**kwargs)

        return _embed()

    async def _aembedding_with_retry(self, **kwargs: Any) -> Any:
        """Call litellm.aembedding with retry on transient errors."""
        import litellm

        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        async def _aembed() -> Any:
            return await litellm.aembedding(**kwargs)

        return await _aembed()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            return []

        params = self._get_litellm_params(input_type=self.document_input_type)
        response = self._embedding_with_retry(input=texts, **params)
        return [item["embedding"] for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        params = self._get_litellm_params(input_type=self.query_input_type)
        response = self._embedding_with_retry(input=[text], **params)
        return response.data[0]["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            return []

        params = self._get_litellm_params(input_type=self.document_input_type)
        response = await self._aembedding_with_retry(input=texts, **params)
        return [item["embedding"] for item in response.data]

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        params = self._get_litellm_params(input_type=self.query_input_type)
        response = await self._aembedding_with_retry(input=[text], **params)
        return response.data[0]["embedding"]
