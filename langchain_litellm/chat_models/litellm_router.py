"""LiteLLM Router chat model integration for LangChain."""

from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_litellm.chat_models.litellm import (
    ChatLiteLLM,
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _cost_metadata,
    _create_retry_decorator,
    _create_usage_metadata,
    _get_field,
)

token_usage_key_name = "token_usage"  # nosec # incorrectly flagged as password
model_extra_key_name = "model_extra"  # nosec # incorrectly flagged as password


def _deployment_metadata(response: Any) -> Dict[str, Any]:
    """Name which deployment the router picked, never the rest of `_hidden_params`.

    `_hidden_params` also carries `api_base` and the resolved request params, and
    this metadata reaches every trace and log. Only the router routes, and only its
    loops still hold the response model: the base class dumps each chunk to a dict,
    which drops the private attribute this reads.
    """
    model_id = _get_field(_get_field(response, "_hidden_params"), "model_id")
    return {"model_id": model_id} if model_id is not None else {}


class ChatLiteLLMRouter(ChatLiteLLM):
    """LiteLLM Router-backed chat model."""

    router: Any

    def __init__(self, *, router: Any, **kwargs: Any) -> None:
        """Construct Chat LiteLLM Router."""
        if "model" not in kwargs and router.model_list:
            kwargs["model"] = router.model_list[0]["model_name"]
        super().__init__(router=router, **kwargs)  # type: ignore[call-arg]
        self.router = router

    @property
    def _llm_type(self) -> str:
        return "LiteLLMRouter"

    def _prepare_params_for_router(self, params: Any) -> None:
        """Add the metadata slot the Router fills in.

        A ``None`` ``api_base`` is already stripped by the caller's None filter, so
        the Router picks its deployment's own; an explicitly configured one is the
        caller's choice and is left alone.
        """
        params.setdefault("metadata", {})

    def set_default_model(self, model_name: str) -> None:
        """Set the default model to use for completion calls.

        Sets `self.model` to `model_name` if it is in the litellm router's
        (`self.router`) model list. This provides the default model to use
        for completion calls if no `model` kwarg is provided.
        """
        model_list = self.router.model_list
        if not model_list:
            raise ValueError("model_list is None or empty.")
        for entry in model_list:
            if entry["model_name"] == model_name:
                self.model = model_name
                # _default_params prefers model_name, so setting only `model`
                # would leave the previous default in force.
                self.model_name = model_name
                return
        raise ValueError(f"Model {model_name} not found in model_list.")

    def _is_claude_model(self) -> bool:
        """Answer for the deployment, not the Router alias.

        ``model``/``model_name`` here is the Router's alias, which need not contain
        the provider's model name at all, so the base implementation would miss a
        Claude deployment routed under an unrelated alias.
        """
        alias = self.model_name or self.model
        matched = [
            entry
            for entry in self.router.model_list or []
            if entry.get("model_name") == alias
        ]
        if not matched:
            return super()._is_claude_model()
        # A model group can fan across providers, so any Claude deployment counts.
        return any(
            "claude" in str(entry.get("litellm_params", {}).get("model", "")).lower()
            for entry in matched
        )

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the router completion call.

        Note: `max_retries` here is independent of any retry/fallback
        configuration (e.g. `num_retries`, `fallbacks`) set on the
        underlying `litellm.Router` instance. If both are configured,
        retries will stack.
        """
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.router.completion(**kwargs)

        return _completion_with_retry(**kwargs)

    async def acompletion_with_retry(
        self,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use tenacity to retry the async router completion call.

        Note: `max_retries` here is independent of any retry/fallback
        configuration (e.g. `num_retries`, `fallbacks`) set on the
        underlying `litellm.Router` instance. If both are configured,
        retries will stack.
        """
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            return await self.router.acompletion(**kwargs)

        return await _completion_with_retry(**kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = self._merge_call_params(params, kwargs)
        # This branch parses a mapping, so it must not inherit stream=True from a
        # streaming=True instance that the caller overrode with stream=False.
        params["stream"] = False
        params = {k: v for k, v in params.items() if v is not None}
        self._prepare_params_for_router(params)

        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response, **params)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        default_chunk_class = AIMessageChunk
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**self._merge_call_params(params, kwargs), "stream": True}
        if "stream_options" not in kwargs:
            params["stream_options"] = (
                self.stream_options
                if self.stream_options is not None
                else {"include_usage": True}
            )
        # After the default, so a caller's explicit None survives the way it does on
        # the base class rather than being filtered out here.
        params = {
            key: value
            for key, value in params.items()
            if value is not None or key == "stream_options"
        }
        self._prepare_params_for_router(params)
        first_chunk_yielded = False

        for chunk in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            usage_metadata = None
            if "usage" in chunk and chunk["usage"]:
                usage_metadata = _create_usage_metadata(chunk["usage"])

            # Read while `chunk` is still the raw response: both the usage-only
            # branch below and the content path need these.
            cost_metadata = _cost_metadata(chunk)
            deployment_metadata = _deployment_metadata(chunk)

            if len(chunk["choices"]) == 0:
                # If the chunk has usage metadata but no content (typical for final stream chunk),
                # yield it so the usage is not lost.
                if usage_metadata:
                    chunk_obj = default_chunk_class(
                        content="", usage_metadata=usage_metadata
                    )
                    # A stream reports its cost here, on a chunk with no content.
                    chunk_obj.response_metadata.update(cost_metadata)
                    cg_chunk = ChatGenerationChunk(message=chunk_obj)
                    if run_manager:
                        run_manager.on_llm_new_token("", chunk=cg_chunk, **params)
                    yield cg_chunk
                continue

            # Process standard content chunks
            delta = chunk["choices"][0]["delta"]
            # Read before `chunk` is rebound from the raw mapping to the message.
            finish_reason = chunk["choices"][0].get("finish_reason")
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)

            # Attach usage if it exists on a content chunk
            if usage_metadata and isinstance(chunk, AIMessageChunk):
                chunk.usage_metadata = usage_metadata

            # Set response_metadata on the first chunk only
            if not first_chunk_yielded and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata = {
                    "model_name": self.model_name or self.model,
                    "model_provider": "litellm",
                    # Named once: it holds for the whole response, and langchain
                    # concatenates a string that two merged chunks both carry.
                    **deployment_metadata,
                }
                first_chunk_yielded = True

            if finish_reason is not None and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata["finish_reason"] = finish_reason

            # Some providers attach the usage, and so the cost, to a content chunk.
            if cost_metadata and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata.update(cost_metadata)

            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk, **params)
            yield cg_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        default_chunk_class = AIMessageChunk
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**self._merge_call_params(params, kwargs), "stream": True}
        if "stream_options" not in kwargs:
            params["stream_options"] = (
                self.stream_options
                if self.stream_options is not None
                else {"include_usage": True}
            )
        # After the default, so a caller's explicit None survives the way it does on
        # the base class rather than being filtered out here.
        params = {
            key: value
            for key, value in params.items()
            if value is not None or key == "stream_options"
        }
        self._prepare_params_for_router(params)
        first_chunk_yielded = False

        async for chunk in await self.acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            # Parse usage metadata first
            usage_metadata = None
            if "usage" in chunk and chunk["usage"]:
                usage_metadata = _create_usage_metadata(chunk["usage"])

            # Read while `chunk` is still the raw response: both the usage-only
            # branch below and the content path need these.
            cost_metadata = _cost_metadata(chunk)
            deployment_metadata = _deployment_metadata(chunk)

            # Check for empty choices
            if len(chunk["choices"]) == 0:
                # Yield pure usage chunk if present
                if usage_metadata:
                    chunk_obj = default_chunk_class(
                        content="", usage_metadata=usage_metadata
                    )
                    # A stream reports its cost here, on a chunk with no content.
                    chunk_obj.response_metadata.update(cost_metadata)
                    cg_chunk = ChatGenerationChunk(message=chunk_obj)
                    if run_manager:
                        await run_manager.on_llm_new_token("", chunk=cg_chunk, **params)
                    yield cg_chunk
                continue

            delta = chunk["choices"][0]["delta"]
            # Read before `chunk` is rebound from the raw mapping to the message.
            finish_reason = chunk["choices"][0].get("finish_reason")
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)

            if usage_metadata and isinstance(chunk, AIMessageChunk):
                chunk.usage_metadata = usage_metadata

            # Set response_metadata on the first chunk only
            if not first_chunk_yielded and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata = {
                    "model_name": self.model_name or self.model,
                    "model_provider": "litellm",
                    # Named once: it holds for the whole response, and langchain
                    # concatenates a string that two merged chunks both carry.
                    **deployment_metadata,
                }
                first_chunk_yielded = True

            if finish_reason is not None and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata["finish_reason"] = finish_reason

            # Some providers attach the usage, and so the cost, to a content chunk.
            if cost_metadata and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata.update(cost_metadata)

            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.content, chunk=cg_chunk, **params
                )
            yield cg_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = self._merge_call_params(params, kwargs)
        # This branch parses a mapping, so it must not inherit stream=True from a
        # streaming=True instance that the caller overrode with stream=False.
        params["stream"] = False
        params = {k: v for k, v in params.items() if v is not None}
        self._prepare_params_for_router(params)

        response = await self.acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response, **params)

    # from
    # https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/openai.py
    # but modified to handle LiteLLM Usage class
    def _combine_llm_outputs(
        self, llm_outputs: List[Optional[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        overall_token_usage: Dict[str, Any] = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                # May be a litellm Usage model or the plain dict a caller mocked.
                usage_items = (
                    token_usage.model_dump()
                    if hasattr(token_usage, "model_dump")
                    else dict(token_usage)
                )
                for k, v in usage_items.items():
                    if k in overall_token_usage and overall_token_usage[k] is not None:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": self.model}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined

    def _create_chat_result(
        self, response: Mapping[str, Any], **params: Any
    ) -> ChatResult:
        from litellm.utils import Usage

        generations = []
        token_usage = response.get("usage", Usage(prompt_tokens=0, total_tokens=0))
        usage_metadata = _create_usage_metadata(token_usage)
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            if isinstance(message, AIMessage):
                message.response_metadata = {
                    "model_name": self.model_name or self.model,
                    "model_provider": "litellm",
                    **_deployment_metadata(response),
                    **_cost_metadata(response),
                }
                message.usage_metadata = usage_metadata
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        # The Router fills `params["metadata"]` in place with its own routing and
        # rate-limit bookkeeping. Core merges whatever is here into the message, so
        # nothing enters it that this class did not choose to name.
        llm_output: Dict[str, Any] = {token_usage_key_name: token_usage}

        # Check standard field first, then fallback to Vertex specific field
        provider_specific_fields = response.get("provider_specific_fields")
        if not provider_specific_fields:
            provider_specific_fields = response.get("vertex_ai_grounding_metadata")

        # Add top-level provider_specific_fields if present in response
        if provider_specific_fields:
            llm_output["provider_specific_fields"] = provider_specific_fields

        return ChatResult(generations=generations, llm_output=llm_output)
