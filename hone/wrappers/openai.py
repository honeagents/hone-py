"""
OpenAI Client Wrapper for Hone.

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
from collections import defaultdict
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
    from openai import AsyncOpenAI, OpenAI
    from openai.types.chat.chat_completion_chunk import (
        ChatCompletionChunk,
        Choice,
        ChoiceDeltaToolCall,
    )
    from openai.types.completion import Completion

from hone.models import TrackedCall

# Any is used since it may work with Azure or other providers
C = TypeVar("C", bound=Union["OpenAI", "AsyncOpenAI", Any])
logger = logging.getLogger(__name__)


@functools.lru_cache
def _get_omit_types() -> tuple[type, ...]:
    """Get NotGiven/Omit sentinel types used by OpenAI SDK."""
    types: list[type[Any]] = []
    try:
        from openai._types import NotGiven, Omit

        types.append(NotGiven)
        types.append(Omit)
    except ImportError:
        pass

    return tuple(types)


def _strip_not_given(d: dict) -> dict:
    """Remove NotGiven values from a dictionary."""
    try:
        omit_types = _get_omit_types()
        if not omit_types:
            return d
        return {
            k: v
            for k, v in d.items()
            if not (isinstance(v, omit_types) or (k.startswith("extra_") and v is None))
        }
    except Exception as e:
        logger.error(f"Error stripping NotGiven: {e}")
        return d


def _extract_usage_metadata(response: Any) -> Dict[str, Any]:
    """Extract usage metadata from OpenAI response."""
    metadata = {}

    if hasattr(response, "model"):
        metadata["model"] = response.model

    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        if hasattr(usage, "total_tokens"):
            metadata["tokens_used"] = usage.total_tokens
        if hasattr(usage, "prompt_tokens"):
            metadata["input_tokens"] = usage.prompt_tokens
        if hasattr(usage, "completion_tokens"):
            metadata["output_tokens"] = usage.completion_tokens

    if hasattr(response, "id"):
        metadata["response_id"] = response.id

    return metadata


def _reduce_choices(choices: List["Choice"]) -> dict:
    """Reduce streaming choices into a single message."""
    reversed_choices = list(reversed(choices))
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "",
    }

    for c in reversed_choices:
        if hasattr(c, "delta") and getattr(c.delta, "role", None):
            message["role"] = c.delta.role
            break

    tool_calls: defaultdict[int, list] = defaultdict(list)

    for c in choices:
        if hasattr(c, "delta"):
            if getattr(c.delta, "content", None):
                message["content"] += c.delta.content
            if getattr(c.delta, "function_call", None):
                if not message.get("function_call"):
                    message["function_call"] = {"name": "", "arguments": ""}
                name_ = getattr(c.delta.function_call, "name", None)
                if name_:
                    message["function_call"]["name"] += name_
                arguments_ = getattr(c.delta.function_call, "arguments", None)
                if arguments_:
                    message["function_call"]["arguments"] += arguments_
            if getattr(c.delta, "tool_calls", None):
                tool_calls_list = c.delta.tool_calls
                if tool_calls_list is not None:
                    for tool_call in tool_calls_list:
                        tool_calls[tool_call.index].append(tool_call)

    if tool_calls:
        message["tool_calls"] = [None for _ in range(max(tool_calls.keys()) + 1)]
        for index, tool_call_chunks in tool_calls.items():
            message["tool_calls"][index] = {
                "index": index,
                "id": next((c.id for c in tool_call_chunks if c.id), None),
                "type": next((c.type for c in tool_call_chunks if c.type), None),
                "function": {"name": "", "arguments": ""},
            }
            for chunk in tool_call_chunks:
                if getattr(chunk, "function", None):
                    name_ = getattr(chunk.function, "name", None)
                    if name_:
                        message["tool_calls"][index]["function"]["name"] += name_
                    arguments_ = getattr(chunk.function, "arguments", None)
                    if arguments_:
                        message["tool_calls"][index]["function"]["arguments"] += (
                            arguments_
                        )

    return {
        "index": getattr(choices[0], "index", 0) if choices else 0,
        "finish_reason": next(
            (
                c.finish_reason
                for c in reversed_choices
                if getattr(c, "finish_reason", None)
            ),
            None,
        ),
        "message": message,
    }


def _reduce_chat_chunks(all_chunks: List["ChatCompletionChunk"]) -> dict:
    """Reduce chat completion chunks into a single response."""
    choices_by_index: defaultdict[int, list] = defaultdict(list)

    for chunk in all_chunks:
        for choice in chunk.choices:
            choices_by_index[choice.index].append(choice)

    if all_chunks:
        d = all_chunks[-1].model_dump() if hasattr(all_chunks[-1], "model_dump") else {}
        d["choices"] = [
            _reduce_choices(choices) for choices in choices_by_index.values()
        ]
    else:
        d = {"choices": [{"message": {"role": "assistant", "content": ""}}]}

    return d


def _reduce_completions(all_chunks: List["Completion"]) -> dict:
    """Reduce completion chunks into a single response."""
    all_content = []

    for chunk in all_chunks:
        content = chunk.choices[0].text
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


def _get_chat_wrapper(
    original_create: Callable,
    hone_client: "Hone",
    name: str = "ChatOpenAI",
) -> Callable:
    """Create a wrapper for chat completions."""

    @functools.wraps(original_create)
    def create(*args, **kwargs):
        # Strip sentinel values
        clean_kwargs = _strip_not_given(kwargs)

        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        # Extract messages for tracking
        messages = clean_kwargs.get("messages", [])
        model = clean_kwargs.get("model", "unknown")
        is_streaming = clean_kwargs.get("stream", False)

        try:
            result = original_create(*args, **kwargs)

            if is_streaming:
                # For streaming, we need to collect chunks
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
                # For non-streaming, track immediately
                end_time = time.perf_counter()
                duration_ms = int((end_time - start_time) * 1000)

                # Extract metadata
                response_meta = _extract_usage_metadata(result)

                # Serialize output
                output = result.model_dump() if hasattr(result, "model_dump") else str(result)

                # Create tracked call
                tracked_call = TrackedCall(
                    function_name=name,
                    input={"messages": messages, "model": model},
                    output=output,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={"provider": "openai", **response_meta},
                    model=response_meta.get("model", model),
                    tokens_used=response_meta.get("tokens_used"),
                    messages=messages,
                    project_id=hone_client.project_id,
                )

                hone_client._enqueue_call(tracked_call)

                return result

        except Exception as e:
            # Track errors too
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            tracked_call = TrackedCall(
                function_name=name,
                input={"messages": messages, "model": model},
                output=None,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "openai"},
                model=model,
                messages=messages,
                error=f"{type(e).__name__}: {str(e)}",
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)
            raise

    @functools.wraps(original_create)
    async def acreate(*args, **kwargs):
        # Strip sentinel values
        clean_kwargs = _strip_not_given(kwargs)

        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        # Extract messages for tracking
        messages = clean_kwargs.get("messages", [])
        model = clean_kwargs.get("model", "unknown")
        is_streaming = clean_kwargs.get("stream", False)

        try:
            result = await original_create(*args, **kwargs)

            if is_streaming:
                # For streaming, we need to collect chunks
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
                # For non-streaming, track immediately
                end_time = time.perf_counter()
                duration_ms = int((end_time - start_time) * 1000)

                # Extract metadata
                response_meta = _extract_usage_metadata(result)

                # Serialize output
                output = result.model_dump() if hasattr(result, "model_dump") else str(result)

                # Create tracked call
                tracked_call = TrackedCall(
                    function_name=name,
                    input={"messages": messages, "model": model},
                    output=output,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={"provider": "openai", **response_meta},
                    model=response_meta.get("model", model),
                    tokens_used=response_meta.get("tokens_used"),
                    messages=messages,
                    project_id=hone_client.project_id,
                )

                hone_client._enqueue_call(tracked_call)

                return result

        except Exception as e:
            # Track errors too
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            tracked_call = TrackedCall(
                function_name=name,
                input={"messages": messages, "model": model},
                output=None,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "openai"},
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

    # Extract usage from last chunk if available
    response_meta = {}
    if chunks:
        last_chunk = chunks[-1]
        if hasattr(last_chunk, "model"):
            response_meta["model"] = last_chunk.model
        if hasattr(last_chunk, "usage") and last_chunk.usage:
            response_meta["tokens_used"] = getattr(last_chunk.usage, "total_tokens", None)

    tracked_call = TrackedCall(
        function_name=name,
        input={"messages": messages, "model": model},
        output=reduced,
        duration_ms=duration_ms,
        started_at=started_at,
        metadata={"provider": "openai", "streaming": True, **response_meta},
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

    # Extract usage from last chunk if available
    response_meta = {}
    if chunks:
        last_chunk = chunks[-1]
        if hasattr(last_chunk, "model"):
            response_meta["model"] = last_chunk.model
        if hasattr(last_chunk, "usage") and last_chunk.usage:
            response_meta["tokens_used"] = getattr(last_chunk.usage, "total_tokens", None)

    tracked_call = TrackedCall(
        function_name=name,
        input={"messages": messages, "model": model},
        output=reduced,
        duration_ms=duration_ms,
        started_at=started_at,
        metadata={"provider": "openai", "streaming": True, **response_meta},
        model=response_meta.get("model", model),
        tokens_used=response_meta.get("tokens_used"),
        messages=messages,
        project_id=hone_client.project_id,
    )

    hone_client._enqueue_call(tracked_call)


def _get_completions_wrapper(
    original_create: Callable,
    hone_client: "Hone",
    name: str = "OpenAI",
) -> Callable:
    """Create a wrapper for completions."""

    @functools.wraps(original_create)
    def create(*args, **kwargs):
        clean_kwargs = _strip_not_given(kwargs)

        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        prompt = clean_kwargs.get("prompt", "")
        model = clean_kwargs.get("model", "unknown")
        is_streaming = clean_kwargs.get("stream", False)

        try:
            result = original_create(*args, **kwargs)

            if is_streaming:
                return _wrap_streaming_response(
                    result,
                    hone_client,
                    name,
                    [{"role": "user", "content": prompt}],
                    model,
                    started_at,
                    start_time,
                    _reduce_completions
                )
            else:
                end_time = time.perf_counter()
                duration_ms = int((end_time - start_time) * 1000)

                response_meta = _extract_usage_metadata(result)
                output = result.model_dump() if hasattr(result, "model_dump") else str(result)

                tracked_call = TrackedCall(
                    function_name=name,
                    input={"prompt": prompt, "model": model},
                    output=output,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={"provider": "openai", **response_meta},
                    model=response_meta.get("model", model),
                    tokens_used=response_meta.get("tokens_used"),
                    messages=[{"role": "user", "content": prompt}] if prompt else None,
                    project_id=hone_client.project_id,
                )

                hone_client._enqueue_call(tracked_call)

                return result

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            tracked_call = TrackedCall(
                function_name=name,
                input={"prompt": prompt, "model": model},
                output=None,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "openai"},
                model=model,
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

        prompt = clean_kwargs.get("prompt", "")
        model = clean_kwargs.get("model", "unknown")
        is_streaming = clean_kwargs.get("stream", False)

        try:
            result = await original_create(*args, **kwargs)

            if is_streaming:
                return _wrap_async_streaming_response(
                    result,
                    hone_client,
                    name,
                    [{"role": "user", "content": prompt}],
                    model,
                    started_at,
                    start_time,
                    _reduce_completions
                )
            else:
                end_time = time.perf_counter()
                duration_ms = int((end_time - start_time) * 1000)

                response_meta = _extract_usage_metadata(result)
                output = result.model_dump() if hasattr(result, "model_dump") else str(result)

                tracked_call = TrackedCall(
                    function_name=name,
                    input={"prompt": prompt, "model": model},
                    output=output,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={"provider": "openai", **response_meta},
                    model=response_meta.get("model", model),
                    tokens_used=response_meta.get("tokens_used"),
                    messages=[{"role": "user", "content": prompt}] if prompt else None,
                    project_id=hone_client.project_id,
                )

                hone_client._enqueue_call(tracked_call)

                return result

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            tracked_call = TrackedCall(
                function_name=name,
                input={"prompt": prompt, "model": model},
                output=None,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "openai"},
                model=model,
                error=f"{type(e).__name__}: {str(e)}",
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)
            raise

    return acreate if _is_async(original_create) else create


def wrap_openai(
    client: C,
    *,
    hone_client: "Hone",
    chat_name: str = "ChatOpenAI",
    completions_name: str = "OpenAI",
) -> C:
    """Wrap an OpenAI client to automatically track all calls to Hone.

    Supports:
        - Chat completions API
        - Completions API (legacy)
        - Sync and async OpenAI clients
        - Streaming and non-streaming responses

    Args:
        client: The OpenAI client to wrap.
        hone_client: The Hone client to send tracking data to.
        chat_name: The name to use for chat completion calls in tracking.
        completions_name: The name to use for completion calls in tracking.

    Returns:
        The wrapped client (same instance, modified in place).

    Example:
        ```python
        from openai import OpenAI
        from hone import Hone
        from hone.wrappers import wrap_openai

        aix = Hone(api_key="aix_xxx", project_id="my-project")
        client = wrap_openai(OpenAI(), hone_client=aix)

        # Now all calls are automatically tracked
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        ```
    """
    # Detect Azure client
    ls_provider = "openai"
    try:
        from openai import AsyncAzureOpenAI, AzureOpenAI

        if isinstance(client, AzureOpenAI) or isinstance(client, AsyncAzureOpenAI):
            ls_provider = "azure"
            chat_name = "AzureChatOpenAI"
            completions_name = "AzureOpenAI"
    except ImportError:
        pass

    # Wrap chat completions
    client.chat.completions.create = _get_chat_wrapper(  # type: ignore[method-assign]
        client.chat.completions.create,
        hone_client,
        chat_name,
    )

    # Wrap legacy completions
    client.completions.create = _get_completions_wrapper(  # type: ignore[method-assign]
        client.completions.create,
        hone_client,
        completions_name,
    )

    # Wrap beta.chat.completions.parse if it exists
    if (
        hasattr(client, "beta")
        and hasattr(client.beta, "chat")
        and hasattr(client.beta.chat, "completions")
        and hasattr(client.beta.chat.completions, "parse")
    ):
        client.beta.chat.completions.parse = _get_chat_wrapper(  # type: ignore[method-assign]
            client.beta.chat.completions.parse,
            hone_client,
            chat_name,
        )

    # Wrap chat.completions.parse if it exists
    if (
        hasattr(client, "chat")
        and hasattr(client.chat, "completions")
        and hasattr(client.chat.completions, "parse")
    ):
        client.chat.completions.parse = _get_chat_wrapper(  # type: ignore[method-assign]
            client.chat.completions.parse,
            hone_client,
            chat_name,
        )

    # Wrap responses API if it exists
    if hasattr(client, "responses"):
        if hasattr(client.responses, "create"):
            client.responses.create = _get_chat_wrapper(  # type: ignore[method-assign]
                client.responses.create,
                hone_client,
                chat_name,
            )

    return client
