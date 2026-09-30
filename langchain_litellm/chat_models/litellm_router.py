"""LiteLLM Router chat model integration for LangChain."""

from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any

import litellm
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
    _REPLAY_SETTINGS,
    ChatLiteLLM,
    _aliased,
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _cost_metadata,
    _create_retry_decorator,
    _create_usage_metadata,
    _extract_root_provider_specific_fields,
    _get_field,
    _keep_reasoning_items,
    _keep_thinking_blocks,
    _rejoin_split_reply,
    _responses_api_gap,
    _sends_manual_thinking,
    _sends_to_responses_api,
    _ThinkingBlockAssembler,
)

# Router settings that re-send a request to another group.
_FALLBACK_SETTINGS = (
    "fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
)


def _without_none(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


def _deployment_gap(deployment: Mapping[str, Any]) -> str | None:
    """Why litellm does not send a deployment to a Responses API, or None when it
    does, with the name that fixes it where renaming can.

    The deployment is resolved in the order ``litellm.completion`` resolves it
    before it picks the route: the named credential fills what is missing,
    ``base_url`` replaces ``api_base``, the model alias applies, and the ``azure``
    flag or a ``deployment_id`` forces Azure.
    """
    resolved = dict(deployment)
    litellm.utils.load_credentials_from_list(resolved)
    model: str = resolved["model"]
    provider = resolved.get("custom_llm_provider")
    # Where litellm replaces the model, renaming the deployment cannot help.
    replaced = []
    aliased = _aliased(model) or model
    if aliased != model:
        replaced.append(f"litellm.model_alias_map renames it {aliased!r}")
        model = aliased
    if resolved.get("azure") is True:
        replaced.append("azure=True makes it an Azure deployment")
        provider = "azure"
    if resolved.get("deployment_id") is not None:
        replaced.append(
            f"its deployment_id {resolved['deployment_id']!r} replaces the model name"
        )
        model, provider = resolved["deployment_id"], "azure"
    base_url = resolved.get("base_url")
    try:
        named, provider, _, _ = litellm.get_llm_provider(
            model=model,
            custom_llm_provider=provider,
            api_base=base_url if base_url is not None else resolved.get("api_base"),
        )
    except litellm.BadRequestError:
        # A prompt-management deployment names its model only once the prompt loads.
        return "litellm cannot tell which provider serves it"
    gap = _responses_api_gap(named, provider)
    if gap is None:
        return None
    if replaced:
        return f"{gap}, since {' and '.join(replaced)}"
    bare = named.removeprefix("chat_completions/").removeprefix("responses/")
    if _sends_to_responses_api(f"responses/{bare}", provider):
        return f"{gap}; name it '{provider}/responses/{bare}'"
    return gap


token_usage_key_name = "token_usage"  # nosec # incorrectly flagged as password
model_extra_key_name = "model_extra"  # nosec # incorrectly flagged as password


def _deployment_metadata(response: Any) -> dict[str, Any]:
    """Name which deployment the router picked, never the rest of `_hidden_params`.

    `_hidden_params` also carries `api_base` and the resolved request params, and
    this metadata reaches every trace and log. Only the router routes, and only its
    loops still hold the response model: the base class dumps each chunk to a dict,
    which drops the private attribute this reads.
    """
    model_id = _get_field(_get_field(response, "_hidden_params"), "model_id")
    return {"model_id": model_id} if model_id is not None else {}


class ChatLiteLLMRouter(ChatLiteLLM):
    """LiteLLM Router-backed chat model.

    The deployment the Router picks decides which API a call reaches. OpenAI's
    built-in tools, such as ``{"type": "web_search"}``, need its Responses API, so
    name the deployment's model ``<provider>/responses/<model>``. A reply's
    reasoning item goes back on later turns only when the group's deployments share
    one model, endpoint and set of credentials and no fallback is set, since an item
    decrypts only where it was issued.

    ``use_responses_api=True`` renames nothing here, since the Router picks the
    deployment on each call. It checks instead: the call goes out unchanged when
    litellm sends every deployment of the called group to a Responses API, and
    raises ``ValueError`` before any request otherwise, naming each deployment that
    is not. It also raises when the call may reach deployments outside the group,
    through a fallback, a ``model_group_alias``, a deployment id or a specific
    deployment of the group's name, a team's own deployments or a deployment's
    ``silent_model``, and when no ``model_list`` entry is named after the group, as
    for one only a wildcard deployment serves. A deployment litellm sends there on
    its own, such as
    ``openai/gpt-5-pro``, passes, but its reasoning items do not go back; name it
    ``openai/responses/gpt-5-pro`` for that.

    Example:
        .. code-block:: python

            from litellm import Router
            from langchain_litellm import ChatLiteLLMRouter

            router = Router(
                model_list=[
                    {
                        "model_name": "gpt-4o-mini",
                        "litellm_params": {"model": "openai/responses/gpt-4o-mini"},
                    }
                ]
            )
            llm = ChatLiteLLMRouter(router=router)
            llm.bind_tools([{"type": "web_search"}]).invoke("Today's top headline?")
    """

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
        matched = self._group_entries()
        if not matched:
            return super()._is_claude_model()
        # A model group can fan across providers, so any Claude deployment counts.
        return any(
            "claude" in str(entry.get("litellm_params", {}).get("model", "")).lower()
            for entry in matched
        )

    def _thinking_endpoint(self, params: dict[str, Any]) -> str | None:
        return self._group_endpoint(params, super()._thinking_endpoint)

    def _reasoning_endpoint(self, params: dict[str, Any]) -> str | None:
        return self._group_endpoint(params, super()._reasoning_endpoint)

    def _group_endpoint(
        self,
        params: dict[str, Any],
        resolve: Callable[[dict[str, Any]], str | None],
    ) -> str | None:
        """The one endpoint every deployment this request can reach shares.

        Each deployment resolves as a direct call would, layered as the Router
        layers it: the call's keys over the router's defaults over the deployment's
        own.
        """
        if self._unknown_reach(params) is not None:
            return None
        deployments = self._deployments(params)
        endpoints = set()
        for deployment in deployments:
            endpoints.add(resolve(deployment))
        views = [{k: d.get(k) for k in _REPLAY_SETTINGS} for d in deployments]
        if len(endpoints) != 1 or any(view != views[0] for view in views):
            return None
        return endpoints.pop()

    def _unknown_reach(self, params: Mapping[str, Any]) -> str | None:
        """Why this call may reach deployments other than its group's own, or None.

        The Router re-sends the same messages to any fallback: its own, the call's,
        and those its defaults or a deployment hand to litellm, which a call cannot
        clear. An alias or a deployment id of the group's name points it elsewhere,
        as does a specific deployment, and a team caller gets the team's own.
        """
        group = params.get("model")
        router = self.router
        defaults = getattr(router, "default_litellm_params", None) or {}
        for key in _FALLBACK_SETTINGS:
            if getattr(router, key, None):
                return f"the Router sets {key}"
            if defaults.get(key):
                return f"the Router's default_litellm_params set {key}"
            if params.get(key):
                return f"the call sets {key}"
        if litellm.model_fallbacks:
            return "litellm.model_fallbacks is set"
        if params.get("specific_deployment"):
            return "the call sets specific_deployment"
        if group in (getattr(router, "model_group_alias", None) or {}):
            return f"the Router's model_group_alias sends {group!r} to another group"
        for entry in getattr(router, "model_list", None) or []:
            if (entry.get("model_info") or {}).get("id") == group:
                return (
                    f"a deployment's model_info id is also {group!r}, and the Router "
                    "matches ids before group names"
                )
            own = entry.get("litellm_params") or {}
            for key in _FALLBACK_SETTINGS:
                if entry.get("model_name") == group and own.get(key):
                    return f"a deployment of {group!r} sets {key}"
        # The Router reads a caller's team from either bucket before it picks.
        buckets = (
            params.get("metadata"),
            params.get("litellm_metadata"),
            defaults.get("metadata"),
        )
        if any(
            isinstance(metadata, Mapping) and metadata.get("user_api_key_team_id")
            for metadata in buckets
        ):
            return (
                "the caller is a team, and the Router serves a team its own deployments"
            )
        return None

    def _route_to_responses_api(self, params: Mapping[str, Any]) -> str:
        """Check that litellm sends every deployment this call can reach to a
        Responses API, and send the call unchanged.

        The Router picks the deployment on each call, so there is no one model name
        to route: each deployment has to be one litellm sends there already.
        """
        group = params["model"]
        sent = _without_none(params)
        unknown = self._unknown_reach(sent)
        if unknown is not None:
            raise ValueError(
                f"use_responses_api=True, but {unknown}, so a call to model group "
                f"{group!r} may reach deployments ChatLiteLLMRouter cannot check. "
                "The flag only checks: without it, a deployment named "
                "'<provider>/responses/<model>' still reaches the Responses API."
            )
        deployments = self._deployments(sent)
        if not deployments:
            raise ValueError(
                "use_responses_api=True, but no model_list entry has model_name "
                f"{group!r}, so ChatLiteLLMRouter cannot tell which deployments serve "
                "it, such as a wildcard one."
            )
        # Replay ignores this one: the mirrored reply never reaches the caller.
        mirrors = [d["silent_model"] for d in deployments if d.get("silent_model")]
        if mirrors:
            raise ValueError(
                f"use_responses_api=True, but a deployment of model group {group!r} "
                "sets silent_model, so the Router also sends each call to "
                f"{', '.join(repr(m) for m in dict.fromkeys(mirrors))}, which "
                "ChatLiteLLMRouter does not check."
            )
        gaps: dict[str, str] = {}
        for deployment in deployments:
            gap = _deployment_gap(deployment)
            if gap is not None:
                gaps.setdefault(deployment["model"], gap)
        if gaps:
            lines = "".join(f"\n- {model!r}: {gap}" for model, gap in gaps.items())
            raise ValueError(
                "use_responses_api=True, but litellm would not send every deployment "
                f"of model group {group!r} to a Responses API:{lines}"
            )
        return group

    def _replay_params(self, params: dict[str, Any]) -> Mapping[str, Any]:
        """The group's deployment as sent. With an endpoint there is at least one,
        and every one has the same replay settings."""
        return self._deployments(params)[0]

    def _deployments(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Each deployment of the called group, layered as the Router sends it."""
        router = self.router
        defaults = getattr(router, "default_litellm_params", None) or {}
        call = {key: value for key, value in params.items() if key != "model"}
        layered = {**{k: v for k, v in defaults.items() if v is not None}, **call}
        deployments = []
        for entry in getattr(router, "model_list", None) or []:
            if entry.get("model_name") != params.get("model"):
                continue
            litellm_params = entry.get("litellm_params") or {}
            deployment = {**litellm_params, **layered}
            # The Router sends a deployment's own tools ahead of the call's.
            tools = [
                *(litellm_params.get("tools") or []),
                *(layered.get("tools") or []),
            ]
            if tools:
                deployment["tools"] = tools
            if not deployment.get("base_model"):
                deployment["base_model"] = (entry.get("model_info") or {}).get(
                    "base_model"
                )
            deployments.append(deployment)
        return deployments

    def _litellm_sends_manual_thinking(self, overrides: Mapping[str, Any]) -> bool:
        """Answer for the Claude deployments the Router may pick.

        A deployment or the Router's defaults can set thinking alone. The Router applies
        the caller's params to each deployment, by litellm's own rule where it has one,
        and only then fills gaps from its defaults.
        """
        matched = self._group_entries()
        if not matched:
            return super()._litellm_sends_manual_thinking(overrides)
        caller = _without_none(
            {"max_tokens": self.max_tokens, **self.model_kwargs, **overrides}
        )
        defaults = _without_none(self.router.default_litellm_params)
        replace = getattr(
            litellm.Router, "_deployment_params_with_request_reasoning_override", None
        )
        return any(
            "claude" in str(params.get("model", "")).lower()
            and _sends_manual_thinking(
                params["model"],
                params.get("custom_llm_provider"),
                params.get("api_base"),
                {
                    **_without_none(replace(params, caller) if replace else params),
                    **defaults,
                    **caller,
                },
            )
            for params in (entry.get("litellm_params", {}) for entry in matched)
        )

    def _group_entries(self) -> list[dict[str, Any]]:
        """The ``model_list`` entries this model's alias routes to."""
        alias = self.model_name or self.model
        return [
            entry
            for entry in self.router.model_list or []
            if entry.get("model_name") == alias
        ]

    def completion_with_retry(
        self, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any
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
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
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
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
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
        reasoning = self._bind_reasoning(messages, message_dicts, params)
        binding = self._bind_thinking(messages, message_dicts, params)

        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return _keep_reasoning_items(
            _keep_thinking_blocks(
                self._create_chat_result(response, **params), response, binding
            ),
            reasoning,
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
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
        reasoning = self._bind_reasoning(messages, message_dicts, params)
        binding = self._bind_thinking(messages, message_dicts, params)
        thinking = _ThinkingBlockAssembler(*binding) if binding else None
        first_chunk_yielded = False
        cost_named = False

        for chunk in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            usage_metadata = None
            if chunk.get("usage"):
                usage_metadata = _create_usage_metadata(chunk["usage"])

            # Read while `chunk` is still the raw response: both the usage-only
            # branch below and the content path need these. A cost named on two
            # chunks cannot be merged, since langchain raises on two floats.
            cost_metadata = {} if cost_named else _cost_metadata(chunk)
            deployment_metadata = _deployment_metadata(chunk)

            if len(chunk["choices"]) == 0:
                # If the chunk has usage metadata but no content (typical for final stream chunk),
                # yield it so the usage is not lost.
                if usage_metadata:
                    chunk_obj = default_chunk_class(
                        content="", usage_metadata=usage_metadata
                    )
                    # A stream reports its cost here, on a chunk with no content.
                    if cost_metadata:
                        chunk_obj.response_metadata.update(cost_metadata)
                        cost_named = True
                    cg_chunk = ChatGenerationChunk(message=chunk_obj)
                    if run_manager:
                        run_manager.on_llm_new_token("", chunk=cg_chunk, **params)
                    yield cg_chunk
                continue

            # Process standard content chunks
            delta = chunk["choices"][0]["delta"]
            # Read before `chunk` is rebound from the raw mapping to the message.
            finish_reason = chunk["choices"][0].get("finish_reason")
            root_metadata = _extract_root_provider_specific_fields(chunk)
            chunk = _convert_delta_to_message_chunk(
                delta, default_chunk_class, thinking, reasoning
            )

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

            # Response-level, so here as on invoke, where llm_output lands them.
            if root_metadata and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata["provider_specific_fields"] = root_metadata

            # Some providers attach the usage, and so the cost, to a content chunk.
            if cost_metadata and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata.update(cost_metadata)
                cost_named = True

            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk, **params)
            yield cg_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
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
        reasoning = self._bind_reasoning(messages, message_dicts, params)
        binding = self._bind_thinking(messages, message_dicts, params)
        thinking = _ThinkingBlockAssembler(*binding) if binding else None
        first_chunk_yielded = False
        cost_named = False

        async for chunk in await self.acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            # Parse usage metadata first
            usage_metadata = None
            if chunk.get("usage"):
                usage_metadata = _create_usage_metadata(chunk["usage"])

            # Read while `chunk` is still the raw response: both the usage-only
            # branch below and the content path need these. A cost named on two
            # chunks cannot be merged, since langchain raises on two floats.
            cost_metadata = {} if cost_named else _cost_metadata(chunk)
            deployment_metadata = _deployment_metadata(chunk)

            # Check for empty choices
            if len(chunk["choices"]) == 0:
                # Yield pure usage chunk if present
                if usage_metadata:
                    chunk_obj = default_chunk_class(
                        content="", usage_metadata=usage_metadata
                    )
                    # A stream reports its cost here, on a chunk with no content.
                    if cost_metadata:
                        chunk_obj.response_metadata.update(cost_metadata)
                        cost_named = True
                    cg_chunk = ChatGenerationChunk(message=chunk_obj)
                    if run_manager:
                        await run_manager.on_llm_new_token("", chunk=cg_chunk, **params)
                    yield cg_chunk
                continue

            delta = chunk["choices"][0]["delta"]
            # Read before `chunk` is rebound from the raw mapping to the message.
            finish_reason = chunk["choices"][0].get("finish_reason")
            root_metadata = _extract_root_provider_specific_fields(chunk)
            chunk = _convert_delta_to_message_chunk(
                delta, default_chunk_class, thinking, reasoning
            )

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

            # Response-level, so here as on invoke, where llm_output lands them.
            if root_metadata and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata["provider_specific_fields"] = root_metadata

            # Some providers attach the usage, and so the cost, to a content chunk.
            if cost_metadata and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata.update(cost_metadata)
                cost_named = True

            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.content, chunk=cg_chunk, **params
                )
            yield cg_chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
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
        reasoning = self._bind_reasoning(messages, message_dicts, params)
        binding = self._bind_thinking(messages, message_dicts, params)

        response = await self.acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return _keep_reasoning_items(
            _keep_thinking_blocks(
                self._create_chat_result(response, **params), response, binding
            ),
            reasoning,
        )

    # from
    # https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/openai.py
    # but modified to handle LiteLLM Usage class
    def _combine_llm_outputs(
        self, llm_outputs: list[dict[str, Any] | None]
    ) -> dict[str, Any]:
        overall_token_usage: dict[str, Any] = {}
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
        for res in _rejoin_split_reply(response["choices"], params.get("n")):
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
                generation_info={"finish_reason": res.get("finish_reason")},
            )
            generations.append(gen)
        # The Router fills `params["metadata"]` in place with its own routing and
        # rate-limit bookkeeping. Core merges whatever is here into the message, so
        # nothing enters it that this class did not choose to name.
        llm_output: dict[str, Any] = {token_usage_key_name: token_usage}

        # Check standard field first, then fallback to Vertex specific field
        provider_specific_fields = response.get("provider_specific_fields")
        if not provider_specific_fields:
            provider_specific_fields = response.get("vertex_ai_grounding_metadata")

        # Add top-level provider_specific_fields if present in response
        if provider_specific_fields:
            llm_output["provider_specific_fields"] = provider_specific_fields

        return ChatResult(generations=generations, llm_output=llm_output)
