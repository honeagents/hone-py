"""
Anthropic Client Wrapper for Hone.

Portions of this code adapted from LangSmith SDK
Copyright (c) 2023 LangChain
Licensed under MIT License
https://github.com/langchain-ai/langsmith-sdk

Modifications for Hone Platform:
- Replaced LangSmith client with Hone client
- Added Hone-specific metadata extraction
- Simplified to work with Hone's TrackedCall model
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from hone.client import Hone
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Completion, Message, MessageStreamEvent

from hone.models import TrackedCall

C = TypeVar("C", bound=Union["Anthropic", "AsyncAnthropic", Any])
logger = logging.getLogger(__name__)


@functools.lru_cache
def _get_not_given() -> Optional[tuple[type, ...]]:
    """Get NotGiven sentinel types used by Anthropic SDK."""
    try:
        from anthropic._types import NotGiven, Omit

        return (NotGiven, Omit)
    except ImportError:
        return None


def _strip_not_given(d: dict) -> dict:
    """Remove NotGiven values from a dictionary."""
    try:
        if not_given := _get_not_given():
            d = {
                k: v
                for k, v in d.items()
                if not any(isinstance(v, t) for t in not_given)
            }
    except Exception as e:
        logger.error(f"Error stripping NotGiven: {e}")

    # Normalize system prompt into messages
    if "system" in d:
        d["messages"] = [{"role": "system", "content": d["system"]}] + d.get(
            "messages", []
        )
        d.pop("system")

    return {k: v for k, v in d.items() if v is not None}


def _extract_usage_metadata(response: Any) -> Dict[str, Any]:
    """Extract usage metadata from Anthropic response."""
    metadata = {}

    if hasattr(response, "model"):
        metadata["model"] = response.model

    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        metadata["tokens_used"] = input_tokens + output_tokens
        metadata["input_tokens"] = input_tokens
        metadata["output_tokens"] = output_tokens

        # Check for cache tokens
        if hasattr(usage, "cache_creation_input_tokens"):
            metadata["cache_creation_tokens"] = usage.cache_creation_input_tokens
        if hasattr(usage, "cache_read_input_tokens"):
            metadata["cache_read_tokens"] = usage.cache_read_input_tokens

    if hasattr(response, "id"):
        metadata["response_id"] = response.id

    if hasattr(response, "stop_reason"):
        metadata["stop_reason"] = response.stop_reason

    return metadata


def _accumulate_event(
    *, event: "MessageStreamEvent", current_snapshot: Optional["Message"]
) -> Optional["Message"]:
    """Accumulate streaming events into a complete message."""
    try:
        from anthropic.types import ContentBlock
        from pydantic import TypeAdapter
    except ImportError:
        logger.debug("Error importing ContentBlock")
        return current_snapshot

    if current_snapshot is None:
        if event.type == "message_start":
            return event.message

        raise RuntimeError(
            f'Unexpected event order, got {event.type} before "message_start"'
        )

    if event.type == "content_block_start":
        adapter: TypeAdapter = TypeAdapter(ContentBlock)
        content_block_instance = adapter.validate_python(
            event.content_block.model_dump()
        )
        current_snapshot.content.append(content_block_instance)
    elif event.type == "content_block_delta":
        content = current_snapshot.content[event.index]
        if content.type == "text" and event.delta.type == "text_delta":
            content.text += event.delta.text
    elif event.type == "message_delta":
        current_snapshot.stop_reason = event.delta.stop_reason
        current_snapshot.stop_sequence = event.delta.stop_sequence
        current_snapshot.usage.output_tokens = event.usage.output_tokens

    return current_snapshot


def _reduce_chat_chunks(all_chunks: Sequence) -> dict:
    """Reduce streaming chunks into a single response."""
    full_message = None

    for chunk in all_chunks:
        try:
            full_message = _accumulate_event(event=chunk, current_snapshot=full_message)
        except RuntimeError as e:
            logger.debug(f"Error accumulating event in Anthropic Wrapper: {e}")
            return {"output": all_chunks}

    if full_message is None:
        return {"output": all_chunks}

    d = full_message.model_dump() if hasattr(full_message, "model_dump") else {}
    d.pop("type", None)

    return {"message": d}


def _reduce_completions(all_chunks: List["Completion"]) -> dict:
    """Reduce completion chunks into a single response."""
    all_content = []

    for chunk in all_chunks:
        content = chunk.completion
        if content is not None:
            all_content.append(content)

    content = "".join(all_content)

    if all_chunks:
        d = all_chunks[-1].model_dump() if hasattr(all_chunks[-1], "model_dump") else {}
        d["choices"] = [{"text": content}]
    else:
        d = {"choices": [{"text": content}]}

    return d


def _is_async(func: Callable) -> bool:
    """Check if a function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


def _get_messages_wrapper(
    original_create: Callable,
    hone_client: "Hone",
    name: str = "ChatAnthropic",
) -> Callable:
    """Create a wrapper for messages.create."""

    @functools.wraps(original_create)
    def create(*args, **kwargs):
        clean_kwargs = _strip_not_given(kwargs)

        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        messages = clean_kwargs.get("messages", [])
        model = clean_kwargs.get("model", "unknown")
        is_streaming = clean_kwargs.get("stream", False)

        try:
            result = original_create(*args, **kwargs)

            if is_streaming:
                return _wrap_streaming_response(
                    result,
                    hone_client,
                    name,
                    messages,
                    model,
                    started_at,
                    start_time,
                    _reduce_chat_chunks
                )
            else:
                end_time = time.perf_counter()
                duration_ms = int((end_time - start_time) * 1000)

                response_meta = _extract_usage_metadata(result)
                output = result.model_dump() if hasattr(result, "model_dump") else str(result)

                tracked_call = TrackedCall(
                    function_name=name,
                    input={"messages": messages, "model": model},
                    output=output,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={"provider": "anthropic", **response_meta},
                    model=response_meta.get("model", model),
                    tokens_used=response_meta.get("tokens_used"),
                    messages=messages,
                    project_id=hone_client.project_id,
                )

                hone_client._enqueue_call(tracked_call)

                return result

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            tracked_call = TrackedCall(
                function_name=name,
                input={"messages": messages, "model": model},
                output=None,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "anthropic"},
                model=model,
                messages=messages,
                error=f"{type(e).__name__}: {str(e)}",
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)
            raise

    @functools.wraps(original_create)
    async def acreate(*args, **kwargs):
        clean_kwargs = _strip_not_given(kwargs)

        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        messages = clean_kwargs.get("messages", [])
        model = clean_kwargs.get("model", "unknown")
        is_streaming = clean_kwargs.get("stream", False)

        try:
            result = await original_create(*args, **kwargs)

            if is_streaming:
                return _wrap_async_streaming_response(
                    result,
                    hone_client,
                    name,
                    messages,
                    model,
                    started_at,
                    start_time,
                    _reduce_chat_chunks
                )
            else:
                end_time = time.perf_counter()
                duration_ms = int((end_time - start_time) * 1000)

                response_meta = _extract_usage_metadata(result)
                output = result.model_dump() if hasattr(result, "model_dump") else str(result)

                tracked_call = TrackedCall(
                    function_name=name,
                    input={"messages": messages, "model": model},
                    output=output,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={"provider": "anthropic", **response_meta},
                    model=response_meta.get("model", model),
                    tokens_used=response_meta.get("tokens_used"),
                    messages=messages,
                    project_id=hone_client.project_id,
                )

                hone_client._enqueue_call(tracked_call)

                return result

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            tracked_call = TrackedCall(
                function_name=name,
                input={"messages": messages, "model": model},
                output=None,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "anthropic"},
                model=model,
                messages=messages,
                error=f"{type(e).__name__}: {str(e)}",
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)
            raise

    return acreate if _is_async(original_create) else create


def _wrap_streaming_response(
    stream,
    hone_client: "Hone",
    name: str,
    messages: list,
    model: str,
    started_at: datetime,
    start_time: float,
    reduce_fn: Callable,
):
    """Wrap a streaming response to track it."""
    chunks = []

    for chunk in stream:
        chunks.append(chunk)
        yield chunk

    # After streaming completes, track the call
    end_time = time.perf_counter()
    duration_ms = int((end_time - start_time) * 1000)

    # Reduce chunks to single response
    reduced = reduce_fn(chunks)

    # Extract usage from accumulated message
    response_meta = {}
    if "message" in reduced and reduced["message"]:
        msg = reduced["message"]
        if "model" in msg:
            response_meta["model"] = msg["model"]
        if "usage" in msg:
            usage = msg["usage"]
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            response_meta["tokens_used"] = input_tokens + output_tokens

    tracked_call = TrackedCall(
        function_name=name,
        input={"messages": messages, "model": model},
        output=reduced,
        duration_ms=duration_ms,
        started_at=started_at,
        metadata={"provider": "anthropic", "streaming": True, **response_meta},
        model=response_meta.get("model", model),
        tokens_used=response_meta.get("tokens_used"),
        messages=messages,
        project_id=hone_client.project_id,
    )

    hone_client._enqueue_call(tracked_call)


async def _wrap_async_streaming_response(
    stream,
    hone_client: "Hone",
    name: str,
    messages: list,
    model: str,
    started_at: datetime,
    start_time: float,
    reduce_fn: Callable,
):
    """Wrap an async streaming response to track it."""
    chunks = []

    async for chunk in stream:
        chunks.append(chunk)
        yield chunk

    # After streaming completes, track the call
    end_time = time.perf_counter()
    duration_ms = int((end_time - start_time) * 1000)

    # Reduce chunks to single response
    reduced = reduce_fn(chunks)

    # Extract usage from accumulated message
    response_meta = {}
    if "message" in reduced and reduced["message"]:
        msg = reduced["message"]
        if "model" in msg:
            response_meta["model"] = msg["model"]
        if "usage" in msg:
            usage = msg["usage"]
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            response_meta["tokens_used"] = input_tokens + output_tokens

    tracked_call = TrackedCall(
        function_name=name,
        input={"messages": messages, "model": model},
        output=reduced,
        duration_ms=duration_ms,
        started_at=started_at,
        metadata={"provider": "anthropic", "streaming": True, **response_meta},
        model=response_meta.get("model", model),
        tokens_used=response_meta.get("tokens_used"),
        messages=messages,
        project_id=hone_client.project_id,
    )

    hone_client._enqueue_call(tracked_call)


def _get_stream_wrapper(
    original_stream: Callable,
    hone_client: "Hone",
    name: str = "ChatAnthropic",
) -> Callable:
    """Create a wrapper for the streaming context manager."""
    is_async = "async" in str(original_stream).lower()

    if is_async:

        class AsyncMessageStreamManagerWrapper:
            def __init__(self, **kwargs):
                self._kwargs = kwargs
                self._clean_kwargs = _strip_not_given(kwargs)
                self._started_at = datetime.now(timezone.utc)
                self._start_time = time.perf_counter()

            async def __aenter__(self):
                self._manager = original_stream(**self._kwargs)
                self._stream = await self._manager.__aenter__()
                return AsyncMessageStreamWrapper(
                    self._stream,
                    hone_client,
                    name,
                    self._clean_kwargs.get("messages", []),
                    self._clean_kwargs.get("model", "unknown"),
                    self._started_at,
                    self._start_time,
                )

            async def __aexit__(self, *exc):
                await self._manager.__aexit__(*exc)

        class AsyncMessageStreamWrapper:
            def __init__(
                self,
                wrapped,
                hone_client: "Hone",
                name: str,
                messages: list,
                model: str,
                started_at: datetime,
                start_time: float,
            ):
                self._wrapped = wrapped
                self._hone_client = hone_client
                self._name = name
                self._messages = messages
                self._model = model
                self._started_at = started_at
                self._start_time = start_time
                self._chunks = []

            @property
            def response(self):
                return self._wrapped.response

            @property
            def text_stream(self):
                async def _text_stream():
                    async for chunk in self._wrapped.text_stream:
                        yield chunk

                return _text_stream()

            async def __anext__(self):
                return await self.__aiter__().__anext__()

            async def __aiter__(self):
                async for chunk in self._wrapped.__aiter__():
                    self._chunks.append(chunk)
                    yield chunk

                # Track after iteration completes
                self._track_call()

            async def __aenter__(self):
                await self._wrapped.__aenter__()
                return self

            async def __aexit__(self, *exc):
                await self._wrapped.__aexit__(*exc)

            async def close(self):
                await self._wrapped.close()

            async def get_final_message(self):
                return await self._wrapped.get_final_message()

            async def get_final_text(self):
                return await self._wrapped.get_final_text()

            async def until_done(self):
                await self._wrapped.until_done()

            @property
            def current_message_snapshot(self):
                return self._wrapped.current_message_snapshot

            def _track_call(self):
                end_time = time.perf_counter()
                duration_ms = int((end_time - self._start_time) * 1000)

                reduced = _reduce_chat_chunks(self._chunks)

                response_meta = {}
                if "message" in reduced and reduced["message"]:
                    msg = reduced["message"]
                    if "model" in msg:
                        response_meta["model"] = msg["model"]
                    if "usage" in msg:
                        usage = msg["usage"]
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        response_meta["tokens_used"] = input_tokens + output_tokens

                tracked_call = TrackedCall(
                    function_name=self._name,
                    input={"messages": self._messages, "model": self._model},
                    output=reduced,
                    duration_ms=duration_ms,
                    started_at=self._started_at,
                    metadata={"provider": "anthropic", "streaming": True, **response_meta},
                    model=response_meta.get("model", self._model),
                    tokens_used=response_meta.get("tokens_used"),
                    messages=self._messages,
                    project_id=self._hone_client.project_id,
                )

                self._hone_client._enqueue_call(tracked_call)

        return AsyncMessageStreamManagerWrapper

    else:

        class MessageStreamManagerWrapper:
            def __init__(self, **kwargs):
                self._kwargs = kwargs
                self._clean_kwargs = _strip_not_given(kwargs)
                self._started_at = datetime.now(timezone.utc)
                self._start_time = time.perf_counter()

            def __enter__(self):
                self._manager = original_stream(**self._kwargs)
                self._stream = self._manager.__enter__()
                return MessageStreamWrapper(
                    self._stream,
                    hone_client,
                    name,
                    self._clean_kwargs.get("messages", []),
                    self._clean_kwargs.get("model", "unknown"),
                    self._started_at,
                    self._start_time,
                )

            def __exit__(self, *exc):
                self._manager.__exit__(*exc)

        class MessageStreamWrapper:
            def __init__(
                self,
                wrapped,
                hone_client: "Hone",
                name: str,
                messages: list,
                model: str,
                started_at: datetime,
                start_time: float,
            ):
                self._wrapped = wrapped
                self._hone_client = hone_client
                self._name = name
                self._messages = messages
                self._model = model
                self._started_at = started_at
                self._start_time = start_time
                self._chunks = []

            @property
            def response(self):
                return self._wrapped.response

            @property
            def text_stream(self):
                def _text_stream():
                    yield from self._wrapped.text_stream

                return _text_stream()

            def __next__(self):
                return self.__iter__().__next__()

            def __iter__(self):
                for chunk in self._wrapped.__iter__():
                    self._chunks.append(chunk)
                    yield chunk

                # Track after iteration completes
                self._track_call()

            def __enter__(self):
                self._wrapped.__enter__()
                return self

            def __exit__(self, *exc):
                self._wrapped.__exit__(*exc)

            def close(self):
                self._wrapped.close()

            def get_final_message(self):
                return self._wrapped.get_final_message()

            def get_final_text(self):
                return self._wrapped.get_final_text()

            def until_done(self):
                return self._wrapped.until_done()

            @property
            def current_message_snapshot(self):
                return self._wrapped.current_message_snapshot

            def _track_call(self):
                end_time = time.perf_counter()
                duration_ms = int((end_time - self._start_time) * 1000)

                reduced = _reduce_chat_chunks(self._chunks)

                response_meta = {}
                if "message" in reduced and reduced["message"]:
                    msg = reduced["message"]
                    if "model" in msg:
                        response_meta["model"] = msg["model"]
                    if "usage" in msg:
                        usage = msg["usage"]
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        response_meta["tokens_used"] = input_tokens + output_tokens

                tracked_call = TrackedCall(
                    function_name=self._name,
                    input={"messages": self._messages, "model": self._model},
                    output=reduced,
                    duration_ms=duration_ms,
                    started_at=self._started_at,
                    metadata={"provider": "anthropic", "streaming": True, **response_meta},
                    model=response_meta.get("model", self._model),
                    tokens_used=response_meta.get("tokens_used"),
                    messages=self._messages,
                    project_id=self._hone_client.project_id,
                )

                self._hone_client._enqueue_call(tracked_call)

        return MessageStreamManagerWrapper


def wrap_anthropic(
    client: C,
    *,
    hone_client: "Hone",
    messages_name: str = "ChatAnthropic",
    completions_name: str = "Anthropic",
) -> C:
    """Wrap an Anthropic client to automatically track all calls to Hone.

    Supports:
        - Messages API
        - Completions API (legacy)
        - Sync and async clients
        - Streaming and non-streaming responses
        - Streaming context manager (messages.stream)

    Args:
        client: The Anthropic client to wrap.
        hone_client: The Hone client to send tracking data to.
        messages_name: The name to use for message calls in tracking.
        completions_name: The name to use for completion calls in tracking.

    Returns:
        The wrapped client (same instance, modified in place).

    Example:
        ```python
        from anthropic import Anthropic
        from hone import Hone
        from hone.wrappers import wrap_anthropic

        aix = Hone(api_key="aix_xxx", project_id="my-project")
        client = wrap_anthropic(Anthropic(), hone_client=aix)

        # Now all calls are automatically tracked
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello!"}]
        )

        # Streaming is also supported
        with client.messages.stream(
            model="claude-3-5-sonnet-latest",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello!"}]
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
        ```
    """
    # Wrap messages.create
    client.messages.create = _get_messages_wrapper(  # type: ignore[method-assign]
        client.messages.create,
        hone_client,
        messages_name,
    )

    # Wrap messages.stream
    client.messages.stream = _get_stream_wrapper(  # type: ignore[method-assign]
        client.messages.stream,
        hone_client,
        messages_name,
    )

    # Wrap legacy completions.create if it exists
    if hasattr(client, "completions") and hasattr(client.completions, "create"):
        client.completions.create = _get_messages_wrapper(  # type: ignore[method-assign]
            client.completions.create,
            hone_client,
            completions_name,
        )

    # Wrap beta.messages.create if it exists
    if (
        hasattr(client, "beta")
        and hasattr(client.beta, "messages")
        and hasattr(client.beta.messages, "create")
    ):
        client.beta.messages.create = _get_messages_wrapper(  # type: ignore[method-assign]
            client.beta.messages.create,
            hone_client,
            messages_name,
        )

    return client
