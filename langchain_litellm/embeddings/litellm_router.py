"""LiteLLM Router as LangChain Embeddings model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_litellm.embeddings.litellm import (
    LiteLLMEmbeddings,
    _create_retry_decorator,
)


class LiteLLMEmbeddingsRouter(LiteLLMEmbeddings):
    """LiteLLM Router-backed embedding model.

    Wraps a ``litellm.Router`` instance to provide load-balanced embedding
    calls across multiple deployments of the same model.

    Example:
        .. code-block:: python

            from litellm import Router
            from langchain_litellm import LiteLLMEmbeddingsRouter

            router = Router(
                model_list=[
                    {
                        "model_name": "text-embedding-3-small",
                        "litellm_params": {
                            "model": "openai/text-embedding-3-small",
                            "api_key": "sk-key1",
                        },
                    },
                    {
                        "model_name": "text-embedding-3-small",
                        "litellm_params": {
                            "model": "openai/text-embedding-3-small",
                            "api_key": "sk-key2",
                        },
                    },
                ]
            )
            embeddings = LiteLLMEmbeddingsRouter(router=router)
    """

    router: Any
    """A litellm.Router instance."""

    def __init__(self, *, router: Any, **kwargs: Any) -> None:
        """Construct LiteLLMEmbeddingsRouter.

        Args:
            router: A litellm.Router instance.
            **kwargs: Additional parameters passed to LiteLLMEmbeddings.
        """
        if "model" not in kwargs:
            first = next(iter(getattr(router, "model_list", None) or []), None)
            alias = first.get("model_name") if isinstance(first, dict) else None
            if alias:
                kwargs["model"] = alias
        super().__init__(**{**kwargs, "router": router})  # type: ignore[call-arg]
        self.router = router

    def _get_router_params(self, *, input_type: Optional[str] = None) -> Dict[str, Any]:
        """Build parameter dict for router.embedding(), excluding None values.

        ``api_base``, ``organization`` and the rest are deliberately absent: the
        Router selects a deployment per call and each carries its own in
        ``litellm_params``, so forwarding this object's would override them. An
        explicitly configured ``api_key`` is passed through, matching
        ``ChatLiteLLMRouter``.
        """
        # An unset field must not clobber the same key supplied through
        # model_kwargs, which is where this class sends provider-specific values.
        params: Dict[str, Any] = {**self.model_kwargs}
        params.update(
            (key, value)
            for key, value in (
                ("model", self.model),
                ("api_key", self.api_key),
                ("timeout", self.request_timeout),
                ("dimensions", self.dimensions),
                ("encoding_format", self.encoding_format),
                ("input_type", input_type),
            )
            if value is not None
        )
        return {k: v for k, v in params.items() if v is not None}

    def _embedding_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the router embedding call.

        Note: `max_retries` here is independent of any retry/fallback
        configuration (e.g. `num_retries`, `fallbacks`) set on the underlying
        `litellm.Router` instance. If both are configured, retries will stack.
        """
        """Call router.embedding with retry, so max_retries is honoured."""
        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        def _embed() -> Any:
            return self.router.embedding(**kwargs)

        return _embed()

    async def _aembedding_with_retry(self, **kwargs: Any) -> Any:
        """Async twin of `_embedding_with_retry`; the same stacking note applies."""
        """Call router.aembedding with retry, so max_retries is honoured."""
        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        async def _aembed() -> Any:
            return await self.router.aembedding(**kwargs)

        return await _aembed()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts via the router.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            return []

        params = self._get_router_params(input_type=self.document_input_type)
        response = self._embedding_with_retry(input=texts, **params)
        return [item["embedding"] for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text via the router.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        params = self._get_router_params(input_type=self.query_input_type)
        response = self._embedding_with_retry(input=[text], **params)
        return response.data[0]["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of document texts via the router.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            return []

        params = self._get_router_params(input_type=self.document_input_type)
        response = await self._aembedding_with_retry(input=texts, **params)
        return [item["embedding"] for item in response.data]

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query text via the router.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        params = self._get_router_params(input_type=self.query_input_type)
        response = await self._aembedding_with_retry(input=[text], **params)
        return response.data[0]["embedding"]
