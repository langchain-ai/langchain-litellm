"""File metadata must survive LangChain message normalization."""

from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models._utils import _normalize_messages
from langchain_core.messages import BaseMessage, HumanMessage

from langchain_litellm import ChatLiteLLM
from langchain_litellm.chat_models.litellm import _convert_message_to_dict


@pytest.fixture(params=["remote", "base64"])
def file_block(request: pytest.FixtureRequest) -> dict:
    payload: dict[str, Any] = (
        {"file_id": "gs://bucket/clip.webm"}
        if request.param == "remote"
        else {
            "file_data": "data:application/pdf;base64,JVBERi0=",
            "filename": "document.pdf",
        }
    )
    payload.update(
        format="audio/webm" if request.param == "remote" else "application/pdf",
        video_metadata={"fps": 2, "start_offset": "0s", "end_offset": "1s"},
        custom_option={"enabled": False},
        file_custom="keep-prefix",
    )
    return {"type": "file", "file": payload}


@pytest.mark.parametrize("normalize", [False, True])
def test_file_metadata_conversion(file_block: dict, normalize: bool) -> None:
    message: BaseMessage = HumanMessage(content=[file_block])
    if normalize:
        message = _normalize_messages([message])[0]
    original = deepcopy(message.content)

    assert _convert_message_to_dict(message)["content"] == [file_block]
    assert message.content == original


def test_file_metadata_ignores_unrelated_extras() -> None:
    message = HumanMessage(
        content=[
            {
                "type": "file",
                "file_id": "file-123",
                "extras": {"trace_id": "private", "file_format": "audio/webm"},
            },
            {"type": "file", "file_id": "file-456"},
            {
                "type": "image",
                "url": "https://example.com/image.png",
                "extras": {"file_format": "ignored"},
            },
        ]
    )
    assert _convert_message_to_dict(message)["content"] == [
        {"type": "file", "file": {"file_id": "file-123", "format": "audio/webm"}},
        {"type": "file", "file": {"file_id": "file-456"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]


def test_file_metadata_tolerates_extras_none() -> None:
    message = HumanMessage(
        content=[{"type": "file", "file_id": "file-123", "extras": None}]
    )
    assert _convert_message_to_dict(message)["content"] == [
        {"type": "file", "file": {"file_id": "file-123"}}
    ]


@pytest.mark.parametrize("use_async", [False, True])
async def test_invoke_preserves_file_metadata(
    file_block: dict, use_async: bool
) -> None:
    llm = ChatLiteLLM(model="gemini/gemini-3-flash")
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
    }
    method = "acompletion" if use_async else "completion"
    call = (
        AsyncMock(return_value=response)
        if use_async
        else MagicMock(return_value=response)
    )
    with patch.object(llm.client, method, new=call):
        messages = [HumanMessage(content=[file_block])]
        if use_async:
            await llm.ainvoke(messages)
        else:
            llm.invoke(messages)

    call.assert_called_once()
    assert call.call_args.kwargs["messages"] == [
        {"role": "user", "content": [file_block]}
    ]
