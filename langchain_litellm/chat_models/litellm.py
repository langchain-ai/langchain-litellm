"""LiteLLM chat model integration for LangChain."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import (
    Any,
    Literal,
    Self,
    cast,
    get_args,
)
from urllib.parse import urlsplit

import litellm
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.base import LangSmithParams
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
)
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.messages.tool import invalid_tool_call
from langchain_core.messages.utils import (
    convert_to_openai_data_block,
    is_data_content_block,
)
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from litellm.types.utils import Delta
from pydantic import BaseModel, Field, model_validator
from typing_extensions import is_typeddict

from langchain_litellm._version import __version__

logger = logging.getLogger(__name__)

# Per-call kwargs that decide WHERE litellm sends the request.
_DESTINATION_KEYS = ("model", "custom_llm_provider")

# A provider's key lives in the field named `<provider>_api_key`, so the mapping is
# derived from the declared fields rather than restated. Only litellm provider ids
# that do NOT follow that convention need an entry here.
_PROVIDER_FIELD_ALIASES = {
    "cohere_chat": "cohere",
    "text-completion-openai": "openai",
    "azure_text": "azure",
}


def _provider_api_key_field(cls: type[BaseModel], provider: str | None) -> str | None:
    """Name the field holding this provider's key on ``cls``, or None if it has none.

    Reading ``model_fields`` off the runtime class keeps this honest and lets a
    subclass contribute a provider the base class does not declare.
    """
    if not provider:
        return None
    field = f"{_PROVIDER_FIELD_ALIASES.get(provider, provider)}_api_key"
    return field if field in cls.model_fields else None


def _get_field(source: Any, name: str) -> Any:
    """Retrieve *name* by dict lookup or attribute access.

    litellm hands a response over as a Pydantic model on one path and as the dict
    `model_dump()` produced on another, so neither access alone reaches both.
    """
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


# Hosts that serve Claude through litellm's Anthropic message format. An
# `anthropic/` route always does, including Anthropic-compatible endpoints like Kimi's.
_CLAUDE_HOSTS = frozenset({"bedrock", "vertex_ai", "azure_ai"})
# Where a stored thinking block records the endpoint that signed it.
_ORIGIN = "origin"


# A bare Bedrock model id, which litellm routes to Bedrock, optionally region-prefixed.
_BEDROCK_CLAUDE_ID = re.compile(r"^([a-z]+\.)?anthropic\.claude")


def _endpoint_name(
    model: str | None,
    custom_llm_provider: str | None,
    api_base: str | None,
    base_model: str | None = None,
) -> str | None:
    """Name where a request's replayed thinking signatures would be checked, if at all.

    Only an Anthropic-format route rebuilds a thinking block from a replayed
    ``thinking_blocks`` entry and checks its signature, and a signature holds only
    where it was issued: Kimi's are not Anthropic's, and one gateway URL can serve
    several models. So the name joins the provider, the base litellm sends to and
    the model. Anything this cannot place is None, which leaves the request as it
    was. The provider comes from the model prefix, not ``litellm.get_llm_provider``,
    which authenticates some providers while it resolves them.
    """
    provider = custom_llm_provider or ""
    # litellm routes a bare model to Anthropic whatever the case of "claude".
    name = (model or "").lower()
    if not provider and "/" in name:
        provider, name = name.split("/", 1)
    if not provider and name.startswith("claude"):
        provider = "anthropic"
    elif not provider and _BEDROCK_CLAUDE_ID.match(name):
        provider = "bedrock"
    claude = "claude" in name or "claude" in (base_model or "")
    if provider != "anthropic" and not (provider in _CLAUDE_HOSTS and claude):
        return None
    base = api_base
    if not base and provider == "anthropic":
        # Where litellm itself sends an Anthropic request with no base configured.
        base = (
            getattr(litellm, "api_base", None)
            or os.environ.get("ANTHROPIC_API_BASE")
            or os.environ.get("ANTHROPIC_BASE_URL")
        )
    try:
        url = urlsplit(base or "")
    except ValueError:
        return None  # litellm rejects a base it cannot parse, and says why.
    # Neither credentials nor a query string changes which server signs a block.
    host = url.netloc.rpartition("@")[2].lower()
    where = url._replace(netloc=host, query="", fragment="").geturl()
    return f"{provider}|{where.rstrip('/')}|{name}"


def _signing_endpoint(
    model: str | None,
    custom_llm_provider: str | None,
    api_base: str | None,
    base_model: str | None = None,
) -> str | None:
    """A digest of ``_endpoint_name``, so a stored block never carries an api_base.

    Saved histories keep it, so changing what goes into the name strands their blocks.
    """
    name = _endpoint_name(model, custom_llm_provider, api_base, base_model)
    return None if name is None else hashlib.sha256(name.encode()).hexdigest()[:16]


def _signed_thinking_blocks(blocks: Any) -> list[dict[str, Any]]:
    """Keep the thinking blocks a provider can verify, with only the keys it defined.

    A ``thinking`` block counts only with a non-empty ``signature``, which is what
    Anthropic and Bedrock check, and a ``redacted_thinking`` block with its ``data``.
    """
    if not isinstance(blocks, list | tuple):
        return []
    kept: list[dict[str, Any]] = []
    for block in blocks:
        block_type = _get_field(block, "type")
        if block_type == "thinking":
            signature = _get_field(block, "signature")
            thinking = _get_field(block, "thinking")
            if isinstance(signature, str) and signature:
                kept.append(
                    {
                        "type": "thinking",
                        "thinking": thinking if isinstance(thinking, str) else "",
                        "signature": signature,
                    }
                )
        elif block_type == "redacted_thinking":
            data = _get_field(block, "data")
            if isinstance(data, str) and data:
                kept.append({"type": "redacted_thinking", "data": data})
    return kept


def _stored_thinking_blocks(blocks: Any, origin: str) -> list[dict[str, Any]]:
    """The signed blocks of a response, each marked with the endpoint that signed it.

    The mark goes on every block, not once on the message: merging stream chunks
    joins two strings end to end but keeps list items apart.
    """
    return [{**block, _ORIGIN: origin} for block in _signed_thinking_blocks(blocks)]


class _ThinkingBlockAssembler:
    """Rebuild whole thinking blocks from the deltas of one stream.

    litellm streams a thinking block as unsigned text fragments and closes it with a
    signed block, which Anthropic fills with the whole text again and Bedrock
    converse leaves empty. A delta therefore contributes a block only once it is
    complete, so the summed chunks equal what a non-streaming call stores. Each call
    builds its own, so fragments from two streams never meet.
    """

    def __init__(self, origin: str) -> None:
        self._origin = origin
        self._pending: list[str] = []

    def feed(self, blocks: Any) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        if not isinstance(blocks, list | tuple):
            return completed
        for block in blocks:
            block_type = _get_field(block, "type")
            if block_type == "redacted_thinking":
                completed.extend(_stored_thinking_blocks([block], self._origin))
                self._pending = []
            elif block_type == "thinking":
                closed = _stored_thinking_blocks([block], self._origin)
                if closed:
                    closed[0]["thinking"] = closed[0]["thinking"] or "".join(
                        self._pending
                    )
                    completed.extend(closed)
                    self._pending = []
                else:
                    thinking = _get_field(block, "thinking")
                    if isinstance(thinking, str):
                        self._pending.append(thinking)
        return completed


def _keep_thinking_blocks(
    result: ChatResult, response: Mapping[str, Any], endpoint: str | None
) -> ChatResult:
    """Store each choice's signed blocks on its message, marked with ``endpoint``.

    This runs on whatever ``_create_chat_result`` returns, so a subclass that
    overrides it keeps its signature.
    """
    if endpoint is None:
        return result
    choices = response["choices"]
    if len(result.generations) != len(choices):
        logger.debug("Not keeping thinking blocks: generations and choices differ.")
        return result
    for generation, choice in zip(result.generations, choices, strict=True):
        blocks = _stored_thinking_blocks(
            _get_field(_get_field(choice, "message"), "thinking_blocks"), endpoint
        )
        if blocks:
            generation.message.additional_kwargs["thinking_blocks"] = blocks
    return result


def _attach_thinking_blocks(
    messages: Sequence[BaseMessage],
    message_dicts: list[dict[str, Any]],
    endpoint: str | None,
) -> None:
    """Hand back the thinking blocks that ``endpoint``, this request's only one, signed.

    A turn gets its blocks back only when every one of them came from that endpoint.
    Its empty tool-call content then goes out as ``None``: litellm writes placeholder
    text into an empty string, which would edit the turn the signature covers.
    """
    if not any(
        isinstance(message, AIMessage)
        and message.additional_kwargs.get("thinking_blocks")
        for message in messages
    ):
        return
    if endpoint is None:
        logger.debug("Not replaying thinking blocks: no single signing endpoint.")
        return
    if len(messages) != len(message_dicts):
        logger.debug("Not replaying thinking blocks: messages and dicts differ.")
        return
    for message, message_dict in zip(messages, message_dicts, strict=True):
        if not isinstance(message, AIMessage):
            continue
        stored = message.additional_kwargs.get("thinking_blocks")
        if not isinstance(stored, list | tuple) or not stored:
            continue
        if any(_get_field(block, _ORIGIN) != endpoint for block in stored):
            logger.debug(
                "Not replaying a turn's thinking blocks: another endpoint signed them."
            )
            continue
        blocks = _signed_thinking_blocks(stored)
        if not blocks:
            continue
        message_dict["thinking_blocks"] = blocks
        if message_dict.get("tool_calls") and message_dict.get("content") == "":
            message_dict["content"] = None


def _cost_metadata(response: Any) -> dict[str, Any]:
    """Name what a call cost, from whichever field litellm recorded it in.

    A complete response carries the figure in `_hidden_params`; a stream leaves it
    there unset and reports it under `usage` on the trailing usage chunk.
    """
    cost = _get_field(_get_field(response, "_hidden_params"), "response_cost")
    if cost is None:
        cost = _get_field(_get_field(response, "usage"), "cost")
    return {"response_cost": cost} if cost is not None else {}


class ChatLiteLLMException(Exception):
    """Exception raised for errors in the LiteLLM integration."""


def _copy_containers(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Copy dicts and lists recursively, leaving anything else shared.

    Deep enough that a caller cannot mutate this model's configuration through the
    params it is handed, and shallow enough not to fail on a client object or any
    other value that cannot be copied. ``copy.copy`` rather than a fresh literal so
    a defaultdict keeps its factory, and ``memo`` so a self-referential value ends.
    """
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        copied_dict = copy.copy(value)
        memo[id(value)] = copied_dict
        for key, item in value.items():
            copied_dict[key] = _copy_containers(item, memo)
        return copied_dict
    if isinstance(value, list):
        copied_list = copy.copy(value)
        memo[id(value)] = copied_list
        for index, item in enumerate(value):
            copied_list[index] = _copy_containers(item, memo)
        return copied_list
    return value


def _create_retry_decorator(
    llm: ChatLiteLLM,
    run_manager: AsyncCallbackManagerForLLMRun | CallbackManagerForLLMRun | None = None,
) -> Callable[[Any], Any]:
    """Return a tenacity retry decorator preconfigured for LiteLLM transient errors."""

    errors = [
        litellm.Timeout,
        litellm.APIError,
        litellm.APIConnectionError,
        litellm.RateLimitError,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        content = _dict.get("content", "") or ""

        additional_kwargs = {}
        tool_calls = []
        invalid_tool_calls: list[InvalidToolCall] = []

        if _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(_dict["function_call"])

        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]

            # Populate standard tool_calls attribute
            for tc in _dict["tool_calls"]:
                try:
                    # Handle both dict and object (Pydantic) access safely
                    tc_id = (
                        tc.get("id")
                        if isinstance(tc, dict)
                        else getattr(tc, "id", None)
                    )

                    func = (
                        tc.get("function")
                        if isinstance(tc, dict)
                        else getattr(tc, "function", None)
                    )
                    if func:
                        func_name = (
                            func.get("name")
                            if isinstance(func, dict)
                            else getattr(func, "name", None)
                        )
                        func_args = (
                            func.get("arguments")
                            if isinstance(func, dict)
                            else getattr(func, "arguments", None)
                        )

                        # Handle JSON String arguments (e.g., OpenAI)
                        if isinstance(func_args, str):
                            try:
                                func_args = json.loads(func_args)
                            except json.JSONDecodeError:
                                # Arguments that don't parse cannot be recovered,
                                # so report the call as invalid rather than
                                # dispatching the tool with no arguments. Matches
                                # langchain_core's default_tool_parser, which keeps
                                # the raw string for inspection.
                                invalid_tool_calls.append(
                                    invalid_tool_call(
                                        name=func_name,
                                        args=func_args,
                                        id=tc_id,
                                        error=None,
                                    )
                                )
                                continue

                        # Ensure args is a dict (e.g., already parsed Dict from Vertex)
                        if not isinstance(func_args, dict):
                            func_args = {}

                        tool_calls.append(
                            ToolCall(
                                name=func_name or "", args=func_args, id=tc_id or ""
                            )
                        )
                except Exception:  # noqa: BLE001, S110
                    # Prevent crash on malformed tool call
                    pass

        if _dict.get("reasoning_content"):
            additional_kwargs["reasoning_content"] = _dict["reasoning_content"]

        # Check standard field first, then fallback to Vertex specific field
        provider_specific_fields = _dict.get("provider_specific_fields")
        if not provider_specific_fields:
            provider_specific_fields = _dict.get("vertex_ai_grounding_metadata")

        if provider_specific_fields:
            additional_kwargs["provider_specific_fields"] = provider_specific_fields

        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )

    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    elif role == "tool":
        return ToolMessage(content=_dict["content"], tool_call_id=_dict["tool_call_id"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
    delta: Delta | dict[str, Any],
    default_class: type[BaseMessageChunk],
    thinking: _ThinkingBlockAssembler | None = None,
) -> BaseMessageChunk:
    # Handle both Delta objects and dicts
    if isinstance(delta, dict):
        role = delta.get("role")
        content = delta.get("content") or ""
        function_call = delta.get("function_call")
        raw_tool_calls = delta.get("tool_calls")
        reasoning_content = delta.get("reasoning_content")
        # Check standard field first, then fallback to Vertex specific field
        provider_specific_fields = delta.get("provider_specific_fields")
        if not provider_specific_fields:
            provider_specific_fields = delta.get("vertex_ai_grounding_metadata")
    else:
        # Safely access attributes with getattr
        role = getattr(delta, "role", None)
        content = getattr(delta, "content", "") or ""
        function_call = getattr(delta, "function_call", None)
        raw_tool_calls = getattr(delta, "tool_calls", None)
        reasoning_content = getattr(delta, "reasoning_content", None)

        # Check standard field first, then fallback to Vertex specific field
        provider_specific_fields = getattr(delta, "provider_specific_fields", None)
        if not provider_specific_fields:
            provider_specific_fields = getattr(
                delta, "vertex_ai_grounding_metadata", None
            )

    additional_kwargs: dict[str, Any] = {}
    if function_call:
        additional_kwargs["function_call"] = dict(function_call)
    if reasoning_content:
        additional_kwargs["reasoning_content"] = reasoning_content
    if thinking is not None:
        thinking_blocks = thinking.feed(_get_field(delta, "thinking_blocks"))
        if thinking_blocks:
            additional_kwargs["thinking_blocks"] = thinking_blocks

    if provider_specific_fields is not None:
        additional_kwargs["provider_specific_fields"] = provider_specific_fields

    tool_call_chunks = []
    if raw_tool_calls:
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                ToolCallChunk(
                    name=rtc["function"]["name"]
                    if isinstance(rtc, dict)
                    else rtc.function.name,
                    args=rtc["function"]["arguments"]
                    if isinstance(rtc, dict)
                    else rtc.function.arguments,
                    id=rtc["id"] if isinstance(rtc, dict) else rtc.id,
                    index=rtc["index"] if isinstance(rtc, dict) else rtc.index,
                )
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        if isinstance(delta, dict):
            func_args = function_call.get("arguments", "") if function_call else ""
            func_name = function_call.get("name", "") if function_call else ""
        else:
            func_args = function_call.arguments if function_call else ""
            func_name = function_call.name if function_call else ""
        return FunctionMessageChunk(content=func_args, name=func_name)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    # Capture the original content from the message
    content = message.content

    # Handle multimodal content conversion if the content is a list
    if isinstance(content, list):
        new_content: list[Any] = []
        for item in content:
            if isinstance(item, dict):
                # Check for LiteLLM's native format which expects a 'file' key
                # Preserve this format exactly as-is to avoid breaking existing LiteLLM implementations
                if item.get("type") == "file" and "file" in item:
                    new_content.append(item)

                # Check for LangChain's standard multimodal format (e.g., 'media', 'image_url')
                # Convert these to the OpenAI/LiteLLM compatible format using the core utility
                elif is_data_content_block(item):
                    converted = convert_to_openai_data_block(item)
                    if isinstance(file_payload := converted.get("file"), dict):
                        # Core normalization moves nested file fields (e.g.,
                        # format and video_metadata) into file_-prefixed extras.
                        # Restore them for LiteLLM after converting back.
                        file_payload.update(
                            {
                                key.removeprefix("file_"): value
                                for key, value in (item.get("extras") or {}).items()
                                if key.startswith("file_")
                            }
                        )
                    new_content.append(converted)

                # Skip tool_use / tool_call blocks — these are handled via
                # message.tool_calls and must not leak into content sent to
                # providers that don't understand them (e.g. OpenAI).
                elif item.get("type") in (
                    "tool_use",
                    "tool_call",
                    "thinking",
                    "redacted_thinking",
                ):
                    continue

                # Pass through standard text blocks or other unrecognized dict formats unchanged
                else:
                    new_content.append(item)
            else:
                # Append non-dict items (like strings) directly
                new_content.append(item)

        # Update content with the processed list.
        # If filtering removed all blocks, collapse to empty string so the
        # provider doesn't receive an empty list.
        content = new_content or ""

    # Initialize the message dictionary with the processed content
    message_dict: dict[str, Any] = {"content": content}

    # Determine the role and specific attributes based on the message type
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        # specific handling for function and tool calls in AI messages
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ]
        # A call that failed to parse was never dispatched, so it must not go back:
        # the provider fails the same parse on the same raw arguments.
        elif (
            "tool_calls" in message.additional_kwargs and not message.invalid_tool_calls
        ):
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
        # Read by OpenAI-compatible reasoning providers and Gemini. litellm never
        # builds an Anthropic or Bedrock thinking block from it: see thinking_blocks.
        if "reasoning_content" in message.additional_kwargs:
            message_dict["reasoning_content"] = message.additional_kwargs[
                "reasoning_content"
            ]
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
        message_dict["name"] = message.name
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
    else:
        # ValueError, not TypeError: callers already catch this one.
        raise ValueError(f"Got unknown type {message}")  # noqa: TRY004

    # Attach the name field if it exists in additional arguments
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]

    return message_dict


class ChatLiteLLM(BaseChatModel):
    """Chat model that uses the LiteLLM API."""

    client: Any = None  #: :meta private:
    model: str = "gpt-3.5-turbo"
    model_name: str | None = None
    stream_options: dict[str, Any] | None = None
    """Model name to use."""
    openai_api_key: str | None = Field(default=None, repr=False)
    azure_api_key: str | None = Field(default=None, repr=False)
    anthropic_api_key: str | None = Field(default=None, repr=False)
    replicate_api_key: str | None = Field(default=None, repr=False)
    cohere_api_key: str | None = Field(default=None, repr=False)
    openrouter_api_key: str | None = Field(default=None, repr=False)
    huggingface_api_key: str | None = Field(default=None, repr=False)
    together_ai_api_key: str | None = Field(default=None, repr=False)
    api_key: str | None = Field(default=None, repr=False)
    streaming: bool = False
    api_base: str | None = None
    """Endpoint override for the upstream provider.

    Also accepts ``base_url`` as an alias (normalized in ``validate_environment``)
    for consistency with the rest of the LangChain ecosystem (e.g. ``ChatOpenAI``,
    ``ChatAnthropic``) and with ``init_chat_model(..., base_url=...)``. A non-None
    ``api_base`` wins; ``base_url`` fills in when ``api_base`` is unset or None,
    so a config built from ``os.getenv`` still reaches the endpoint."""
    organization: str | None = None
    custom_llm_provider: str | None = None
    base_model: str | None = None
    extra_headers: dict[str, str] | None = Field(default=None, repr=False)
    request_timeout: float | tuple[float, float] | None = None
    temperature: float | None = None
    """Run inference with this temperature. Must be in the closed
       interval [0.0, 2.0]."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""
    top_p: float | None = None
    """Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0]."""
    top_k: int | None = None
    """Decode using top-k sampling: consider the set of top_k most probable tokens.
       Must be positive."""
    n: int | None = None
    """Number of chat completions to generate for each prompt. Note that the API may
       not return the full n completions if duplicates are generated."""
    max_tokens: int | None = None
    """The maximum number of tokens to generate in the reply."""
    num_ctx: int | None = None
    """Context window size (e.g. for Ollama models)."""

    max_retries: int = 1

    def _thinking_config(self) -> dict[str, Any]:
        thinking_config = self.model_kwargs.get("thinking")
        return thinking_config if isinstance(thinking_config, dict) else {}

    def _is_claude_model(self) -> bool:
        return "claude" in (self.model_name or self.model).lower()

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for the LiteLLM completion call."""
        set_model_value = self.model
        if self.model_name is not None:
            set_model_value = self.model_name
        params: dict[str, Any] = {
            "model": set_model_value,
            "timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            "custom_llm_provider": self.custom_llm_provider,
            "num_ctx": self.num_ctx,
            "base_model": self.base_model,
        }
        # litellm rejects these for watsonx on key presence, so an unset one has
        # to be absent rather than None.
        for name, value in (("top_p", self.top_p), ("top_k", self.top_k)):
            if value is not None:
                params[name] = value
        # Copy containers at every level: a caller mutating the returned params,
        # however deeply, must not reach back into this instance's model_kwargs.
        return {
            **params,
            **{
                key: _copy_containers(value) for key, value in self.model_kwargs.items()
            },
        }

    def _constructor_destination(self) -> tuple[str | None, str | None]:
        """The model and provider this instance sends to when a call overrides neither.

        ``model_kwargs`` is merged last into ``_default_params``, so an entry there
        for ``model`` or ``custom_llm_provider`` is what actually reaches litellm
        and must decide the credential too.
        """
        overrides = self.model_kwargs or {}
        model = overrides.get("model") or (
            self.model_name if self.model_name is not None else self.model
        )
        provider = overrides.get("custom_llm_provider") or self.custom_llm_provider
        return model, provider

    def _resolve_api_key(
        self,
        model: str | None = None,
        custom_llm_provider: str | None = None,
    ) -> str | None:
        """Resolve the key to send to litellm for this call.

        ``api_key`` wins when set. Otherwise fall back to the provider-specific
        field matching the provider, so a key supplied as, say,
        ``openai_api_key`` is actually used rather than silently ignored.

        ``model`` and ``custom_llm_provider`` are the EFFECTIVE values for this
        call, either of which a caller may override per call. Both decide where
        litellm sends the request, so both must decide which key travels with it.

        Returns ``None`` when the caller configured nothing, leaving every
        environment variable to litellm's own resolution.
        """
        if self.api_key:
            return self.api_key
        # A generic key supplied through model_kwargs is as provider-agnostic and as
        # explicit as the field, so it takes the same precedence rather than being
        # replaced by a provider-scoped resolution when a call redirects.
        explicit = (self.model_kwargs or {}).get("api_key")
        if explicit:
            return explicit

        # With no provider-scoped key set there is nothing to attribute, and asking
        # litellm to attribute a proxy deployment name costs a banner on stdout.
        fields = type(self).model_fields
        if not any(
            getattr(self, name, None) for name in fields if name.endswith("_api_key")
        ):
            return None

        default_model, default_provider = self._constructor_destination()
        provider = custom_llm_provider or default_provider
        if not provider:
            effective_model = model or default_model
            if not effective_model:
                return None
            try:
                _, provider, _, _ = litellm.get_llm_provider(model=effective_model)
            except litellm.BadRequestError:
                # The one expected failure: litellm cannot attribute this model, so
                # defer to its own credential resolution rather than guess.
                logger.debug(
                    "No provider attributed to %r; leaving api_key unset.",
                    effective_model,
                )
                return None

        field = _provider_api_key_field(type(self), provider)
        if field is None:
            return None
        return getattr(self, field, None) or None

    def _merge_call_params(
        self, params: dict[str, Any], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge per-call kwargs over the client params, rescoping the destination.

        ``_client_params`` is built from constructor state, so every value in it
        that depends on WHERE the request goes — the credential, the endpoint, the
        organization, the provider-specific headers, the cost-tracking model — is
        resolved against the constructor's destination. A caller may redirect the
        request per call with ``model`` or ``custom_llm_provider``.

        Only the provider-scoped credential is INFERRED from that destination, so only
        it is re-resolved when the destination changes. ``api_base``, ``organization``
        and ``extra_headers`` exist because a caller set them, so they survive a
        redirect untouched, and a pinned ``api_base`` holds the credential with it:
        choosing an endpoint and a key together is choosing them for each other.

        A ``None`` override means "not supplied", matching how litellm reads params.
        """
        merged = {**params, **kwargs}

        # None means omitted: fall back rather than sending a null destination.
        for key in _DESTINATION_KEYS:
            if key in kwargs and kwargs[key] is None:
                merged[key] = params.get(key)

        redirected = {
            key: kwargs[key] for key in _DESTINATION_KEYS if kwargs.get(key) is not None
        }
        if not redirected:
            return merged

        # A caller who pinned an endpoint and supplied a key chose them together, so
        # a redirect must not swap one of them for a provider-scoped key.
        if kwargs.get("api_key") is None and merged.get("api_base") is None:
            merged["api_key"] = self._resolve_api_key(
                model=redirected.get("model"),
                custom_llm_provider=redirected.get("custom_llm_provider"),
            )
        return merged

    def _thinking_endpoint(self, params: dict[str, Any]) -> str | None:
        """The one endpoint this request reaches, when it checks replayed thinking.

        ``params`` must be the merged per-call params, since a call may redirect.
        litellm re-sends the same messages to any fallback, sends a model_list to
        every deployment in it, and hands deployment_id or the azure flag to Azure,
        so any of those replays nothing. An alias is named by the model it maps to.
        """
        if (
            params.get("fallbacks")
            or params.get("context_window_fallback_dict")
            or params.get("model_list")
            or params.get("deployment_id")
            or params.get("azure")
            or getattr(litellm, "model_fallbacks", None)
        ):
            return None
        model = params.get("model")
        aliases = getattr(litellm, "model_alias_map", None) or {}
        return _signing_endpoint(
            aliases.get(model, model),
            params.get("custom_llm_provider"),
            # litellm sends to base_url over api_base when a caller sets both.
            params.get("base_url") or params.get("api_base"),
            params.get("base_model"),
        )

    @property
    def _client_params(self) -> dict[str, Any]:
        """Get the per-call parameters passed to litellm.completion."""
        creds: dict[str, Any] = {
            "api_base": self.api_base,
            "api_key": self._resolve_api_key(),
            "organization": self.organization,
            "extra_headers": self.extra_headers,
        }
        # Only override a default that is actually configured. An unset credential
        # would otherwise clobber the same key supplied through model_kwargs, and
        # litellm treats a missing param and a None one identically anyway.
        return {
            **self._default_params,
            **{key: value for key, value in creds.items() if value is not None},
        }

    def completion_with_retry(
        self, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.completion(**kwargs)

        return _completion_with_retry(**kwargs)

    async def acompletion_with_retry(
        self, run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the async completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            return await self.client.acompletion(**kwargs)

        return await _completion_with_retry(**kwargs)

    @model_validator(mode="after")
    def _set_litellm_version(self) -> Self:
        """Set package version in metadata."""
        self._add_version("langchain-litellm", __version__)
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Any) -> Any:
        """Normalize the base_url alias, collect credentials, and check the ranges.

        A ``mode="before"`` validator sees only what the caller passed, so pydantic's
        own ``model_fields_set`` stays truthful and langchain-core can tell a chosen
        ``streaming=False`` from the default.
        """
        if not isinstance(values, dict):
            return values

        # A config built from JSON or os.getenv carries None for an unset value.
        # Dropping it leaves the default in place without marking the field set.
        for name in [key for key, value in values.items() if value is None]:
            field = cls.model_fields.get(name)
            if field is None or field.is_required():
                continue
            if type(None) not in get_args(field.annotation):
                del values[name]

        # Accept `base_url` as an alias for `api_base` for cross-provider
        # consistency (e.g. `init_chat_model(..., base_url=...)`). Without this,
        # `base_url` is silently dropped by Pydantic's `extra="ignore"`. The
        # explicit `api_base` takes precedence when both are provided.
        base_url = values.pop("base_url", None)
        if base_url is not None and values.get("api_base") is None:
            values["api_base"] = base_url

        values["client"] = litellm

        if (
            values.get("temperature") is not None
            and not 0 <= values["temperature"] <= 2
        ):
            raise ValueError("temperature must be in the range [0.0, 2.0]")

        if values.get("top_p") is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values.get("top_k") is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        return values

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
        endpoint = self._thinking_endpoint(params)
        _attach_thinking_blocks(messages, message_dicts, endpoint)
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return _keep_thinking_blocks(
            self._create_chat_result(response), response, endpoint
        )

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        token_usage = response.get("usage", {})
        usage_metadata = _create_usage_metadata(token_usage)
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            if isinstance(message, AIMessage):
                message.response_metadata = {
                    "model_name": self.model_name or self.model,
                    "model_provider": "litellm",
                    **_cost_metadata(response),
                }
                message.usage_metadata = usage_metadata
            gen = ChatGeneration(
                message=message,
                generation_info={
                    "finish_reason": res.get("finish_reason"),
                    "logprobs": res.get("logprobs"),
                },
            )
            generations.append(gen)
        set_model_value = self.model
        if self.model_name is not None:
            set_model_value = self.model_name
        llm_output = {"token_usage": token_usage, "model": set_model_value}
        # Check standard field first, then fallback to Vertex specific field
        provider_specific_fields = response.get("provider_specific_fields")
        if not provider_specific_fields:
            provider_specific_fields = response.get("vertex_ai_grounding_metadata")

        # Add provider_specific_fields if present at response level
        if provider_specific_fields:
            llm_output["provider_specific_fields"] = provider_specific_fields
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._client_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**self._merge_call_params(params, kwargs), "stream": True}
        if "stream_options" not in kwargs:
            params["stream_options"] = (
                self.stream_options
                if self.stream_options is not None
                else {"include_usage": True}
            )
        endpoint = self._thinking_endpoint(params)
        _attach_thinking_blocks(messages, message_dicts, endpoint)
        thinking = _ThinkingBlockAssembler(endpoint) if endpoint else None
        default_chunk_class = AIMessageChunk
        first_chunk_yielded = False
        cost_named = False

        for chunk in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            # Ensure chunk is a dict
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()

            # Extract usage metadata first
            usage_metadata = None
            if chunk.get("usage"):
                usage_metadata = _create_usage_metadata(chunk["usage"])

            # Read while `chunk` is still the raw response: both the usage-only
            # branch below and the content path need it. A cost named on two
            # chunks cannot be merged, since langchain raises on two floats.
            cost_metadata = {} if cost_named else _cost_metadata(chunk)

            # Handle empty choices (usage-only chunks)
            if len(chunk["choices"]) == 0:
                if usage_metadata:
                    # Create an empty chunk just to carry the metadata
                    chunk_obj = default_chunk_class(
                        content="", usage_metadata=usage_metadata
                    )
                    # A stream reports its cost here, on a chunk with no content.
                    if cost_metadata:
                        chunk_obj.response_metadata.update(cost_metadata)
                        cost_named = True
                    cg_chunk = ChatGenerationChunk(message=chunk_obj)
                    if run_manager:
                        run_manager.on_llm_new_token("", chunk=cg_chunk)
                    yield cg_chunk
                continue

            delta = chunk["choices"][0]["delta"]
            finish_reason = chunk["choices"][0].get("finish_reason")

            # Inject Root Metadata into Delta
            root_metadata = chunk.get("provider_specific_fields")
            if not root_metadata:
                root_metadata = chunk.get("vertex_ai_grounding_metadata")

            if root_metadata:
                delta["provider_specific_fields"] = root_metadata

            chunk = _convert_delta_to_message_chunk(
                delta, default_chunk_class, thinking
            )

            if usage_metadata and isinstance(chunk, AIMessageChunk):
                chunk.usage_metadata = usage_metadata

            # Set response_metadata on the first chunk only
            if not first_chunk_yielded and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata = {
                    "model_name": self.model_name or self.model,
                    "model_provider": "litellm",
                }
                first_chunk_yielded = True

            if finish_reason is not None and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata["finish_reason"] = finish_reason

            # Some providers attach the usage, and so the cost, to a content chunk.
            if cost_metadata and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata.update(cost_metadata)
                cost_named = True

            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
            yield cg_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**self._merge_call_params(params, kwargs), "stream": True}
        if "stream_options" not in kwargs:
            params["stream_options"] = (
                self.stream_options
                if self.stream_options is not None
                else {"include_usage": True}
            )
        endpoint = self._thinking_endpoint(params)
        _attach_thinking_blocks(messages, message_dicts, endpoint)
        thinking = _ThinkingBlockAssembler(endpoint) if endpoint else None
        default_chunk_class = AIMessageChunk
        first_chunk_yielded = False
        cost_named = False

        async for chunk in await self.acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            # Ensure chunk is a dict
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()

            # Extract usage metadata first
            usage_metadata = None
            if chunk.get("usage"):
                usage_metadata = _create_usage_metadata(chunk["usage"])

            # Read while `chunk` is still the raw response: both the usage-only
            # branch below and the content path need it. A cost named on two
            # chunks cannot be merged, since langchain raises on two floats.
            cost_metadata = {} if cost_named else _cost_metadata(chunk)

            # Handle empty choices (usage-only chunks)
            if len(chunk["choices"]) == 0:
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
                        await run_manager.on_llm_new_token("", chunk=cg_chunk)
                    yield cg_chunk
                continue

            delta = chunk["choices"][0]["delta"]
            finish_reason = chunk["choices"][0].get("finish_reason")

            # Inject Root Metadata into Delta
            root_metadata = chunk.get("provider_specific_fields")
            if not root_metadata:
                root_metadata = chunk.get("vertex_ai_grounding_metadata")

            if root_metadata:
                delta["provider_specific_fields"] = root_metadata

            chunk = _convert_delta_to_message_chunk(
                delta, default_chunk_class, thinking
            )

            if usage_metadata and isinstance(chunk, AIMessageChunk):
                chunk.usage_metadata = usage_metadata

            # Set response_metadata on the first chunk only
            if not first_chunk_yielded and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata = {
                    "model_name": self.model_name or self.model,
                    "model_provider": "litellm",
                }
                first_chunk_yielded = True

            if finish_reason is not None and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata["finish_reason"] = finish_reason

            # Some providers attach the usage, and so the cost, to a content chunk.
            if cost_metadata and isinstance(chunk, AIMessageChunk):
                chunk.response_metadata.update(cost_metadata)
                cost_named = True

            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
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
        endpoint = self._thinking_endpoint(params)
        _attach_thinking_blocks(messages, message_dicts, endpoint)
        response = await self.acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return _keep_thinking_blocks(
            self._create_chat_result(response), response, endpoint
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        tool_choice: dict
        | str
        | Literal["auto", "none", "required", "any"]
        | bool
        | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        LiteLLM expects tools argument in OpenAI format.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Controls tool-calling behavior. Options are:
                - str of the form ``"<<tool_name>>"``: calls <<tool_name>> tool.
                - ``"auto"``:
                    automatically selects a tool (including no tool).
                - ``"none"``:
                    does not call a tool.
                - ``"any"`` or ``"required"`` or ``True``:
                    forces at least one tool to be called.
                - dict of the form:
                ``{"type": "function", "function": {"name": <<tool_name>>}}``
                - ``False`` or ``None``: no effect
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain_core.runnables.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        # Robustly handle tool_choice='any' or True for ALL providers.
        # Many providers (Gemini, Vertex, etc.) via LiteLLM reject "any" but accept "required".
        # We map "any" (or True) to "required" globally to prevent crashes.
        # A False tool_choice is left alone; it behaves like None or auto per provider.
        if tool_choice is True or tool_choice == "any":
            tool_choice = "required"

        # Handle dict tool_choice logic — validate before any downgrade so
        # typos in tool names always raise, even when thinking is enabled.
        if isinstance(tool_choice, dict):
            tool_names = [
                formatted_tool["function"]["name"] for formatted_tool in formatted_tools
            ]
            if not any(
                tool_name == tool_choice["function"]["name"] for tool_name in tool_names
            ):
                raise ValueError(
                    f"Tool choice {tool_choice} was specified, but the only "
                    f"provided tools were {tool_names}."
                )

        # When thinking/extended thinking is enabled, tool_choice="required"
        # (or a forced specific tool) suppresses chain-of-thought on Claude
        # models. Downgrade to "auto" only for Claude so other providers keep
        # their original forced tool-calling behavior.
        # Prior art: langchain-ai/langchain#35544, langchain-ai/langchain-aws#927.
        thinking_config = self._thinking_config()
        is_claude_model = self._is_claude_model()
        # "any" is already mapped to "required" above, so only check "required"
        tool_choice_is_forced = tool_choice == "required" or isinstance(
            tool_choice, dict
        )
        if (
            thinking_config.get("type") == "enabled"
            and is_claude_model
            and tool_choice_is_forced
        ):
            logger.warning(
                "tool_choice=%r is incompatible with thinking/extended "
                "thinking on Claude models. Downgrading tool_choice to 'auto' "
                "so the model can produce chain-of-thought reasoning before "
                "calling tools.",
                tool_choice,
            )
            tool_choice = "auto"

        return super().bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type | BaseModel,
        *,
        method: Literal["json_schema", "function_calling", "json_mode"]
        | None = "json_schema",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        # Remove unsupported parameters
        _ = kwargs.pop("tools", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)

        parser: Runnable[Any, Any]
        pre_parser: Callable[[AIMessage], AIMessage] | None = None
        if method == "function_calling":
            # Determine appropriate tool_choice based on model
            # Use "required" for most models, which is more widely supported than "any"
            tool_choice_value = "required"
            bind_kwargs = {"tool_choice": tool_choice_value}

            if (
                self._is_claude_model()
                and self._thinking_config().get("type") == "enabled"
            ):
                warning_message = (
                    "Structured output via function calling is not guaranteed on "
                    "Claude models when `thinking` is enabled. Tool calls may be "
                    "omitted; this runnable will raise OutputParserException when "
                    "no tool call is returned. Consider disabling `thinking` or "
                    'using `method="json_schema"`.'
                )
                warnings.warn(warning_message, stacklevel=2)
                bind_kwargs = {}

                def _raise_if_no_tool_calls(message: AIMessage) -> AIMessage:
                    if not message.tool_calls:
                        raise OutputParserException(warning_message)
                    return message

                pre_parser = _raise_if_no_tool_calls

            # pydantic
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                parser = PydanticToolsParser(
                    tools=[cast(TypeBaseModel, schema)], first_tool_only=True
                )
                llm = self.bind_tools([schema], **bind_kwargs)
            # dict or typeddict
            elif is_typeddict(schema) or isinstance(schema, dict):
                tool_def = convert_to_openai_tool(schema)  # type: ignore[arg-type]
                function_name = tool_def["function"]["name"]
                parser = JsonOutputKeyToolsParser(
                    key_name=function_name, first_tool_only=True
                )
                llm = self.bind_tools([tool_def], **bind_kwargs)
            else:
                msg = f"Unsupported schema type {type(schema)}"
                raise ValueError(msg)

        elif method == "json_schema":
            if strict is None:
                strict_flag = True
            else:
                strict_flag = strict

            # Setup parser for JSON text
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                parser = PydanticOutputParser(pydantic_object=schema)
            else:
                parser = JsonOutputParser()

            # Setup LLM with json_schema
            tool_def = convert_to_openai_tool(schema)  # type: ignore[arg-type]
            raw_schema = tool_def["function"]["parameters"]
            json_schema = _ensure_additional_properties_false(raw_schema)

            # Safe schema name extraction
            schema_name = getattr(schema, "__name__", tool_def["function"]["name"])

            llm = self.bind(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": json_schema,
                        "strict": strict_flag,
                    },
                }
            )

        elif method == "json_mode":
            # Setup parser for JSON text
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                parser = PydanticOutputParser(pydantic_object=schema)
            else:
                parser = JsonOutputParser()

            # Setup LLM with json_mode (simpler than json_schema)
            llm = self.bind(response_format={"type": "json_object"})

        else:
            msg = f"Unsupported method '{method}'. Must be 'json_schema', 'function_calling', or 'json_mode'"
            raise ValueError(msg)

        parse_chain: Runnable[Any, Any] = parser
        if pre_parser is not None:
            parse_chain = pre_parser | parser

        if include_raw:
            parser_with_fallback = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parse_chain,
                parsing_error=lambda _: None,
            ).with_fallbacks(
                [RunnablePassthrough.assign(parsed=lambda _: None)],
                exception_key="parsing_error",
            )
            return {"raw": llm} | parser_with_fallback

        return llm | parse_chain

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        set_model_value = self.model
        if self.model_name is not None:
            set_model_value = self.model_name
        return {
            "model": set_model_value,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "n": self.n,
            "num_ctx": self.num_ctx,
        }

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Return LangSmith tracing parameters for this model.

        Overrides the base implementation to set ``ls_provider`` to ``"litellm"``
        and ``ls_model_name`` to the resolved model string. These values are used
        by LangSmith for run tagging and by LangChain middleware such as
        ``SummarizationMiddleware``, which compares ``response_metadata["model_provider"]``
        against ``ls_provider`` to decide whether reported token counts should be
        trusted. Without this override, ``ls_provider`` would be absent and that
        middleware check would always short-circuit.

        ``ls_model_name`` resolves ``kwargs["model"]`` first because ``_generate``
        merges per-call kwargs over the default params, so a ``model`` passed to
        ``bind`` or ``invoke`` is the model actually requested.
        """
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "litellm"
        params["ls_model_name"] = kwargs.get("model") or self.model_name or self.model
        return params

    @property
    def _llm_type(self) -> str:
        return "litellm-chat"


def _create_usage_metadata(token_usage: Any) -> UsageMetadata:
    """Create `UsageMetadata` from a LiteLLM usage object.

    `token_usage` may be a plain dict (streaming path, after `model_dump()`) or
    a Pydantic `Usage` model (non-streaming path, where
    `ModelResponse.get("usage")` returns the raw model).

    Both are handled uniformly via `_get_field`.
    """
    input_tokens = int(_get_field(token_usage, "prompt_tokens") or 0)
    output_tokens = int(_get_field(token_usage, "completion_tokens") or 0)
    _raw_total = _get_field(token_usage, "total_tokens")
    total_tokens = (
        int(_raw_total) if _raw_total is not None else (input_tokens + output_tokens)
    )

    # ── input token details (cache) ───────────────────────────────────────
    cache_read = _get_field(token_usage, "cache_read_input_tokens")
    cache_creation = _get_field(token_usage, "cache_creation_input_tokens")

    # Fallback: some providers nest cache info inside prompt_tokens_details
    # instead of top-level keys.
    if cache_read is None or cache_creation is None:
        prompt_details = _get_field(token_usage, "prompt_tokens_details")
        if prompt_details is not None:
            if cache_read is None:
                cache_read = _get_field(prompt_details, "cached_tokens")
            if cache_creation is None:
                cache_creation = _get_field(prompt_details, "cache_creation_tokens")

    input_token_details: dict = {
        "cache_read": int(cache_read) if cache_read is not None else None,
        "cache_creation": int(cache_creation) if cache_creation is not None else None,
    }

    # ── output token details (reasoning) ──────────────────────────────────
    completion_details = _get_field(token_usage, "completion_tokens_details")
    reasoning = (
        _get_field(completion_details, "reasoning_tokens")
        if completion_details is not None
        else None
    )
    output_token_details: dict = {
        "reasoning": int(reasoning) if reasoning is not None else None,
    }

    # ── assemble ──────────────────────────────────────────────────────────
    usage_metadata = UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )

    filtered_input = {k: v for k, v in input_token_details.items() if v is not None}
    if filtered_input:
        usage_metadata["input_token_details"] = InputTokenDetails(**filtered_input)

    filtered_output = {k: v for k, v in output_token_details.items() if v is not None}
    if filtered_output:
        usage_metadata["output_token_details"] = OutputTokenDetails(**filtered_output)

    return usage_metadata


def _ensure_additional_properties_false(schema_dict: dict[str, Any]) -> dict[str, Any]:
    """Recursively ensure additionalProperties is set to false for all objects."""
    if isinstance(schema_dict, dict):
        result = schema_dict.copy()

        if result.get("type") == "object":
            result["additionalProperties"] = False

        for key, value in result.items():
            if isinstance(value, dict):
                result[key] = _ensure_additional_properties_false(value)
            elif isinstance(value, list):
                result[key] = [
                    _ensure_additional_properties_false(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]

        return result

    return schema_dict
