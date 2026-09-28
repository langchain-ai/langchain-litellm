"""A reply's url_citation annotations reach the caller.

litellm keeps web-search citations on the reply as ``annotations``: flat on its
Responses API bridge, nested under ``url_citation`` on Chat Completions. The
connector keeps them as litellm returns them in ``additional_kwargs`` and shows
them as standard citations in ``content_blocks``, leaving ``content`` a string.
The wire tests patch only the transport.
"""

import asyncio
from typing import Any

import litellm
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from openai.types.chat import ChatCompletion

from langchain_litellm.chat_models import ChatLiteLLM, ChatLiteLLMRouter
from langchain_litellm.chat_models.litellm import _convert_message_to_dict
from tests.utils import responses_api_events, responses_api_reply, serve_http

TEXT = "Sunny in Paris."
SEARCH = {
    "type": "web_search_call",
    "id": "ws_1",
    "status": "completed",
    "action": {"type": "search", "query": "weather in Paris"},
}


def _flat(url: str, start: int, end: int) -> dict[str, Any]:
    return {
        "type": "url_citation",
        "url": url,
        "title": f"About {url}",
        "start_index": start,
        "end_index": end,
    }


def _nested(url: str, start: int, end: int) -> dict[str, Any]:
    return {
        "type": "url_citation",
        "url_citation": {
            "url": url,
            "title": f"About {url}",
            "start_index": start,
            "end_index": end,
        },
    }


def _citation(url: str, start: int, end: int) -> dict[str, Any]:
    return {
        "type": "citation",
        "url": url,
        "title": f"About {url}",
        "start_index": start,
        "end_index": end,
    }


def _message_item(*parts: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": text, "annotations": annotations}
            for text, annotations in parts
        ],
    }


BRIDGED = responses_api_reply(
    SEARCH, _message_item((TEXT, [_flat("https://a.example", 0, 5)]))
)
SPLIT = responses_api_reply(
    SEARCH,
    _message_item(
        ("Sunny. ", [_flat("https://a.example", 0, 5)]),
        ("Warm too. ", [_flat("https://b.example", 0, 4)]),
        ("Dry.", [_flat("https://c.example", 0, 3)]),
    ),
)
BRIDGED_EVENTS = responses_api_events(
    {
        "type": "response.created",
        "response": {**BRIDGED, "status": "in_progress", "output": []},
    },
    {
        "type": "response.output_text.delta",
        "output_index": 1,
        "item_id": "msg_1",
        "content_index": 0,
        "delta": TEXT,
        "logprobs": [],
    },
    {
        "type": "response.output_text.annotation.added",
        "output_index": 1,
        "item_id": "msg_1",
        "content_index": 0,
        "annotation_index": 0,
        "annotation": _flat("https://a.example", 0, 5),
    },
    {"type": "response.completed", "response": BRIDGED},
)
CHAT = ChatCompletion.model_validate(
    {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-search-preview",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": TEXT,
                    "annotations": [_nested("https://a.example", 0, 5)],
                },
            }
        ],
    }
).model_dump(exclude_none=True)
_CHUNK = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "gpt-4o-search-preview",
}
CHAT_EVENTS = [
    {
        **_CHUNK,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": TEXT},
                "finish_reason": None,
            }
        ],
    },
    {
        **_CHUNK,
        "choices": [
            {
                "index": 0,
                "delta": {"annotations": [_nested("https://a.example", 0, 5)]},
                "finish_reason": None,
            }
        ],
    },
    {**_CHUNK, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
]
# The citation rides on the delta with the last of the text.
CHAT_EVENTS_CITED_TEXT = [
    {
        **_CHUNK,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Sunny "},
                "finish_reason": None,
            }
        ],
    },
    {
        **_CHUNK,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "in Paris.",
                    "annotations": [_nested("https://a.example", 0, 5)],
                },
                "finish_reason": None,
            }
        ],
    },
    {**_CHUNK, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
]
MODES = ["invoke", "ainvoke", "stream", "astream"]


def _bridged(kind: str, **kwargs: Any) -> ChatLiteLLM:
    if kind == "base":
        return ChatLiteLLM(
            model="openai/gpt-5-mini", api_key="k", use_responses_api=True, **kwargs
        )
    deployment = {"model": "openai/responses/gpt-5-mini", "api_key": "k"}
    router = litellm.Router(
        model_list=[{"model_name": "g", "litellm_params": deployment}]
    )
    return ChatLiteLLMRouter(router=router, model_name="g", **kwargs)


def _search_model(kind: str, **kwargs: Any) -> ChatLiteLLM:
    if kind == "base":
        return ChatLiteLLM(model="openai/gpt-4o-search-preview", api_key="k", **kwargs)
    deployment = {"model": "openai/gpt-4o-search-preview", "api_key": "k"}
    router = litellm.Router(
        model_list=[{"model_name": "s", "litellm_params": deployment}]
    )
    return ChatLiteLLMRouter(router=router, model_name="s", **kwargs)


def _run(llm: Any, mode: str) -> Any:
    if mode == "invoke":
        return llm.invoke("weather?")
    if mode == "ainvoke":
        return asyncio.run(llm.ainvoke("weather?"))
    if mode == "stream":
        chunks = list(llm.stream("weather?"))
    else:

        async def collect() -> list[Any]:
            return [c async for c in llm.astream("weather?")]

        chunks = asyncio.run(collect())
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


def _cited_text(message: AIMessage) -> list[tuple[str, dict[str, Any]]]:
    """Each text block's citations, with the text they point at."""
    blocks: list[dict[str, Any]] = [dict(block) for block in message.content_blocks]
    return [
        (block["text"][c["start_index"] : c["end_index"]], c)
        for block in blocks
        if block["type"] == "text"
        for c in block.get("annotations") or []
    ]


@pytest.mark.parametrize("mode", ["invoke", "ainvoke"])
@pytest.mark.parametrize("kind", ["base", "router"])
def test_a_bridged_reply_keeps_its_citations(
    monkeypatch: pytest.MonkeyPatch, kind: str, mode: str
) -> None:
    """Content stays a string; the citations are kept and shown as standard ones."""
    serve_http(monkeypatch, BRIDGED)

    message = _run(_bridged(kind), mode)

    assert message.content == TEXT
    assert message.additional_kwargs["annotations"] == [
        _flat("https://a.example", 0, 5)
    ]
    assert _cited_text(message) == [("Sunny", _citation("https://a.example", 0, 5))]


@pytest.mark.parametrize("kind", ["base", "router"])
def test_a_split_reply_shifts_later_citations_onto_the_joined_text(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The bridge gives each text part its own choice and offsets; joining the parts
    must move each part's citations past all the text before it."""
    serve_http(monkeypatch, SPLIT)

    message = _bridged(kind).invoke("weather?")

    assert message.content == "Sunny. Warm too. Dry."
    assert [text for text, _ in _cited_text(message)] == ["Sunny", "Warm", "Dry"]
    assert [a["start_index"] for a in message.additional_kwargs["annotations"]] == [
        0,
        7,
        17,
    ]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["base", "router"])
def test_a_chat_completions_reply_keeps_its_citations(
    monkeypatch: pytest.MonkeyPatch, kind: str, mode: str
) -> None:
    """Chat Completions nests each citation under url_citation, streamed or not."""
    serve_http(monkeypatch, CHAT, CHAT_EVENTS)

    message = _run(_search_model(kind), mode)

    assert message.content == TEXT
    assert message.additional_kwargs["annotations"] == [
        _nested("https://a.example", 0, 5)
    ]
    assert _cited_text(message) == [("Sunny", _citation("https://a.example", 0, 5))]


@pytest.mark.parametrize("kind", ["base", "router"])
def test_output_version_v1_puts_standard_citations_in_content(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    serve_http(monkeypatch, BRIDGED)

    message = _bridged(kind, output_version="v1").invoke("weather?")

    assert message.content == [
        {
            "type": "text",
            "text": TEXT,
            "annotations": [_citation("https://a.example", 0, 5)],
        }
    ]


@pytest.mark.parametrize(
    "events",
    [
        pytest.param(CHAT_EVENTS, id="on-their-own-delta"),
        pytest.param(CHAT_EVENTS_CITED_TEXT, id="on-a-later-text-delta"),
    ],
)
@pytest.mark.parametrize("mode", ["stream", "astream"])
@pytest.mark.parametrize("kind", ["base", "router"])
def test_output_version_v1_keeps_citations_on_a_streamed_reply(
    monkeypatch: pytest.MonkeyPatch, kind: str, mode: str, events: list[Any]
) -> None:
    """Core turns each chunk into blocks; the citations must land on the text."""
    serve_http(monkeypatch, CHAT, events)

    message = _run(_search_model(kind, output_version="v1"), mode)

    assert _cited_text(message) == [("Sunny", _citation("https://a.example", 0, 5))]


def test_a_bridged_stream_carries_no_citations_until_litellm_forwards_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """litellm's bridge drops response.output_text.annotation.added when streaming;
    this fails once it forwards them, which is the cue to cover streaming here."""
    serve_http(monkeypatch, BRIDGED, BRIDGED_EVENTS)

    message = _run(_bridged("base"), "stream")

    assert message.content == TEXT
    assert "annotations" not in message.additional_kwargs


def _plain(message: AIMessage) -> list[Any]:
    """content_blocks as langchain-core builds them with no translator."""
    metadata = {
        k: v for k, v in message.response_metadata.items() if k != "model_provider"
    }
    return message.model_copy(update={"response_metadata": metadata}).content_blocks


@pytest.mark.parametrize("cls", [AIMessage, AIMessageChunk])
def test_a_message_without_citations_keeps_core_content_blocks(cls: type) -> None:
    message = cls(
        content=TEXT,
        additional_kwargs={"reasoning_content": "Look it up."},
        tool_calls=[{"name": "f", "args": {}, "id": "call_1", "type": "tool_call"}]
        if cls is AIMessage
        else [],
        response_metadata={"model_provider": "litellm"},
    )

    assert message.content_blocks == _plain(message)


def test_citations_leave_the_rest_of_content_blocks_to_core() -> None:
    message = AIMessage(
        content=TEXT,
        additional_kwargs={
            "reasoning_content": "Look it up.",
            "annotations": [_flat("https://a.example", 0, 5)],
        },
        tool_calls=[{"name": "f", "args": {}, "id": "call_1", "type": "tool_call"}],
        response_metadata={"model_provider": "litellm"},
    )

    blocks = message.content_blocks

    assert [block["type"] for block in blocks] == [
        block["type"] for block in _plain(message)
    ]
    text = next(block for block in blocks if block["type"] == "text")
    assert text["annotations"] == [_citation("https://a.example", 0, 5)]


@pytest.mark.parametrize(
    "annotations",
    [
        pytest.param(["not a dict"], id="not-a-dict"),
        pytest.param([{"type": "url_citation", "title": "no url"}], id="no-url"),
        pytest.param([{"type": "url_citation", "url": ""}], id="empty-url"),
        pytest.param([{"type": "file_citation", "file_id": "f"}], id="other-kind"),
        pytest.param(
            [{"type": "file_citation", "url": "https://a", "file_id": "f"}],
            id="other-kind-with-a-url",
        ),
        pytest.param(
            [{"type": "url_citation", "url": "https://a", "start_index": 3}],
            id="half-a-span",
        ),
    ],
)
def test_an_annotation_that_is_not_a_url_citation_is_left_out(
    annotations: list[Any],
) -> None:
    message = AIMessage(
        content=TEXT,
        additional_kwargs={"annotations": annotations},
        response_metadata={"model_provider": "litellm"},
    )

    assert message.content_blocks == _plain(message)


def test_one_bad_annotation_leaves_the_good_ones() -> None:
    message = AIMessage(
        content=TEXT,
        additional_kwargs={
            "annotations": [
                _flat("https://a.example", 0, 5),
                {"type": "file_citation", "file_id": "f"},
            ]
        },
        response_metadata={"model_provider": "litellm"},
    )

    assert [c for _, c in _cited_text(message)] == [
        _citation("https://a.example", 0, 5)
    ]


@pytest.mark.parametrize(
    ("annotation", "citation"),
    [
        pytest.param(
            {"type": "url_citation", "url": "https://a.example"},
            {"type": "citation", "url": "https://a.example"},
            id="no-span",
        ),
        pytest.param(
            {**_flat("https://a.example", 0, 5), "title": ""},
            {
                k: v
                for k, v in _citation("https://a.example", 0, 5).items()
                if k != "title"
            },
            id="empty-title",
        ),
    ],
)
def test_a_citation_carries_only_what_litellm_gave(
    annotation: dict[str, Any], citation: dict[str, Any]
) -> None:
    message = AIMessage(
        content=TEXT,
        additional_kwargs={"annotations": [annotation]},
        response_metadata={"model_provider": "litellm"},
    )

    text = next(b for b in message.content_blocks if b["type"] == "text")
    assert text.get("annotations") == [citation]


def test_citations_with_no_text_are_left_to_core() -> None:
    message = AIMessage(
        content="",
        additional_kwargs={"annotations": [_flat("https://a.example", 0, 5)]},
        response_metadata={"model_provider": "litellm"},
    )

    assert message.content_blocks == _plain(message)


def test_citations_never_go_back_to_the_provider() -> None:
    """They are output only; a v1 history's text block would carry them too."""
    kept = AIMessage(
        content=TEXT, additional_kwargs={"annotations": [_flat("https://a", 0, 5)]}
    )
    v1 = AIMessage(
        content=[
            {
                "type": "text",
                "text": TEXT,
                "annotations": [_citation("https://a", 0, 5)],
            }
        ]
    )

    assert "annotations" not in _convert_message_to_dict(kept)
    assert _convert_message_to_dict(v1)["content"] == [{"type": "text", "text": TEXT}]
