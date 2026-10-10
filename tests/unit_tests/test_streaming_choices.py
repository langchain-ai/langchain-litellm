"""Streaming must not concatenate independently generated choices."""

from unittest.mock import AsyncMock, patch

import pytest

from langchain_litellm import ChatLiteLLM, ChatLiteLLMRouter
from tests.utils import make_router


def _model(router: bool, **kwargs):
    if router:
        return ChatLiteLLMRouter(router=make_router(), **kwargs)
    return ChatLiteLLM(model="gpt-4o-mini", **kwargs)


@pytest.mark.parametrize("router", [False, True])
@pytest.mark.parametrize("source", ["field", "model_kwargs", "call"])
@pytest.mark.parametrize("streaming", [None, True])
def test_multiple_choices_rejected_before_sync_request(router, source, streaming):
    constructor = {} if streaming is None else {"streaming": streaming}
    call = {}
    if source == "call":
        call["n"] = 2
    elif source == "model_kwargs":
        constructor["model_kwargs"] = {"n": 2}
    else:
        constructor["n"] = 2
    llm = _model(router, **constructor)
    with (
        patch.object(
            type(llm), "completion_with_retry", return_value=iter([])
        ) as completion,
        pytest.raises(ValueError, match="n must be 1 when streaming"),
    ):
        list(llm.stream("hi", **call))
    completion.assert_not_called()


@pytest.mark.parametrize("router", [False, True])
@pytest.mark.parametrize("source", ["field", "model_kwargs", "call"])
@pytest.mark.parametrize("streaming", [None, True])
async def test_multiple_choices_rejected_before_async_request(
    router, source, streaming
):
    constructor = {} if streaming is None else {"streaming": streaming}
    call = {}
    if source == "call":
        call["n"] = 2
    elif source == "model_kwargs":
        constructor["model_kwargs"] = {"n": 2}
    else:
        constructor["n"] = 2
    llm = _model(router, **constructor)

    async def empty_stream():
        if False:
            yield

    with (
        patch.object(
            type(llm),
            "acompletion_with_retry",
            new_callable=AsyncMock,
            return_value=empty_stream(),
        ) as completion,
        pytest.raises(ValueError, match="n must be 1 when streaming"),
    ):
        async for _ in llm.astream("hi", **call):
            pass
    completion.assert_not_called()


@pytest.mark.parametrize("router", [False, True])
@pytest.mark.parametrize("constructor_n,call", [(None, {}), (1, {}), (2, {"n": 1})])
def test_single_choice_sync_stream_still_works(router, constructor_n, call):
    llm = _model(router, n=constructor_n)
    reply = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}]}
    with patch.object(type(llm), "completion_with_retry", return_value=iter([reply])):
        assert "".join(chunk.content for chunk in llm.stream("hi", **call)) == "hi"


@pytest.mark.parametrize("router", [False, True])
@pytest.mark.parametrize("constructor_n,call", [(None, {}), (1, {}), (2, {"n": 1})])
async def test_single_choice_async_stream_still_works(router, constructor_n, call):
    llm = _model(router, n=constructor_n)

    async def replies():
        yield {
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}]
        }

    with patch.object(
        type(llm),
        "acompletion_with_retry",
        new_callable=AsyncMock,
        return_value=replies(),
    ):
        chunks = [chunk async for chunk in llm.astream("hi", **call)]
        assert "".join(chunk.content for chunk in chunks) == "hi"
