"""
LiteLLM Wrapper for Hone.

This is a new wrapper that patches LiteLLM's completion function to automatically
track all LLM calls to Hone. LiteLLM provides a unified interface to multiple
LLM providers (OpenAI, Anthropic, Cohere, etc.), so wrapping LiteLLM covers all
of them with a single integration.
"""

from __future__ import annotations

import functools
import logging
import time
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

if TYPE_CHECKING:
    from hone.client import Hone

from hone.models import TrackedCall

logger = logging.getLogger(__name__)

# Store reference to original functions
_original_completion: Optional[Callable] = None
_original_acompletion: Optional[Callable] = None
_original_embedding: Optional[Callable] = None
_original_aembedding: Optional[Callable] = None


def _extract_usage_metadata(response: Any) -> Dict[str, Any]:
    """Extract usage metadata from LiteLLM response."""
    metadata = {}

    if hasattr(response, "model"):
        metadata["model"] = response.model

    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        if hasattr(usage, "total_tokens"):
            metadata["tokens_used"] = usage.total_tokens
        elif hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
            metadata["tokens_used"] = usage.prompt_tokens + usage.completion_tokens

        if hasattr(usage, "prompt_tokens"):
            metadata["input_tokens"] = usage.prompt_tokens
        if hasattr(usage, "completion_tokens"):
            metadata["output_tokens"] = usage.completion_tokens

    if hasattr(response, "id"):
        metadata["response_id"] = response.id

    # Extract provider info
    if hasattr(response, "_hidden_params"):
        hidden = response._hidden_params
        if "custom_llm_provider" in hidden:
            metadata["provider"] = hidden["custom_llm_provider"]
        if "model_id" in hidden:
            metadata["model_id"] = hidden["model_id"]

    return metadata


def _reduce_streaming_chunks(all_chunks: list) -> dict:
    """Reduce streaming chunks into a single response."""
    if not all_chunks:
        return {"choices": [{"message": {"role": "assistant", "content": ""}}]}

    # Accumulate content
    content = ""
    tool_calls = []
    function_call = None
    model = None
    usage = None

    for chunk in all_chunks:
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "delta"):
                delta = choice.delta
                if hasattr(delta, "content") and delta.content:
                    content += delta.content
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.index >= len(tool_calls):
                            tool_calls.append({"index": tc.index, "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if tc.id:
                            tool_calls[tc.index]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls[tc.index]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments
                if hasattr(delta, "function_call") and delta.function_call:
                    if function_call is None:
                        function_call = {"name": "", "arguments": ""}
                    if delta.function_call.name:
                        function_call["name"] += delta.function_call.name
                    if delta.function_call.arguments:
                        function_call["arguments"] += delta.function_call.arguments

        if hasattr(chunk, "model") and chunk.model:
            model = chunk.model

        if hasattr(chunk, "usage") and chunk.usage:
            usage = chunk.usage

    result = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "stop",
        }],
    }

    if model:
        result["model"] = model

    if usage:
        result["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    if tool_calls:
        result["choices"][0]["message"]["tool_calls"] = tool_calls

    if function_call:
        result["choices"][0]["message"]["function_call"] = function_call

    return result


def _create_completion_wrapper(
    original_fn: Callable,
    hone_client: "Hone",
    name: str = "LiteLLM",
) -> Callable:
    """Create a wrapper for litellm.completion."""

    @functools.wraps(original_fn)
    def completion(*args, **kwargs):
        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")
        is_streaming = kwargs.get("stream", False)

        try:
            result = original_fn(*args, **kwargs)

            if is_streaming:
                return _wrap_streaming_response(
                    result,
                    hone_client,
                    name,
                    messages,
                    model,
                    started_at,
                    start_time,
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
                    metadata={"provider": response_meta.get("provider", "litellm"), **response_meta},
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
                metadata={"provider": "litellm"},
                model=model,
                messages=messages,
                error=f"{type(e).__name__}: {str(e)}",
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)
            raise

    return completion


def _create_acompletion_wrapper(
    original_fn: Callable,
    hone_client: "Hone",
    name: str = "LiteLLM",
) -> Callable:
    """Create a wrapper for litellm.acompletion."""

    @functools.wraps(original_fn)
    async def acompletion(*args, **kwargs):
        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")
        is_streaming = kwargs.get("stream", False)

        try:
            result = await original_fn(*args, **kwargs)

            if is_streaming:
                return _wrap_async_streaming_response(
                    result,
                    hone_client,
                    name,
                    messages,
                    model,
                    started_at,
                    start_time,
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
                    metadata={"provider": response_meta.get("provider", "litellm"), **response_meta},
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
                metadata={"provider": "litellm"},
                model=model,
                messages=messages,
                error=f"{type(e).__name__}: {str(e)}",
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)
            raise

    return acompletion


def _wrap_streaming_response(
    stream,
    hone_client: "Hone",
    name: str,
    messages: list,
    model: str,
    started_at: datetime,
    start_time: float,
):
    """Wrap a streaming response to track it."""
    chunks = []

    for chunk in stream:
        chunks.append(chunk)
        yield chunk

    # After streaming completes, track the call
    end_time = time.perf_counter()
    duration_ms = int((end_time - start_time) * 1000)

    # Reduce chunks
    reduced = _reduce_streaming_chunks(chunks)

    # Extract metadata
    response_meta = {}
    if "model" in reduced:
        response_meta["model"] = reduced["model"]
    if "usage" in reduced:
        response_meta["tokens_used"] = reduced["usage"].get("total_tokens")

    tracked_call = TrackedCall(
        function_name=name,
        input={"messages": messages, "model": model},
        output=reduced,
        duration_ms=duration_ms,
        started_at=started_at,
        metadata={"provider": "litellm", "streaming": True, **response_meta},
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
):
    """Wrap an async streaming response to track it."""
    chunks = []

    async for chunk in stream:
        chunks.append(chunk)
        yield chunk

    # After streaming completes, track the call
    end_time = time.perf_counter()
    duration_ms = int((end_time - start_time) * 1000)

    # Reduce chunks
    reduced = _reduce_streaming_chunks(chunks)

    # Extract metadata
    response_meta = {}
    if "model" in reduced:
        response_meta["model"] = reduced["model"]
    if "usage" in reduced:
        response_meta["tokens_used"] = reduced["usage"].get("total_tokens")

    tracked_call = TrackedCall(
        function_name=name,
        input={"messages": messages, "model": model},
        output=reduced,
        duration_ms=duration_ms,
        started_at=started_at,
        metadata={"provider": "litellm", "streaming": True, **response_meta},
        model=response_meta.get("model", model),
        tokens_used=response_meta.get("tokens_used"),
        messages=messages,
        project_id=hone_client.project_id,
    )

    hone_client._enqueue_call(tracked_call)


def _create_embedding_wrapper(
    original_fn: Callable,
    hone_client: "Hone",
    name: str = "LiteLLMEmbedding",
) -> Callable:
    """Create a wrapper for litellm.embedding."""

    @functools.wraps(original_fn)
    def embedding(*args, **kwargs):
        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        input_text = kwargs.get("input", "")
        model = kwargs.get("model", "unknown")

        try:
            result = original_fn(*args, **kwargs)

            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            response_meta = _extract_usage_metadata(result)

            # Don't store the actual embeddings (too large)
            output = {
                "model": getattr(result, "model", model),
                "usage": result.usage.model_dump() if hasattr(result, "usage") and result.usage else None,
                "data_count": len(result.data) if hasattr(result, "data") else 0,
            }

            tracked_call = TrackedCall(
                function_name=name,
                input={"input": input_text[:500] if isinstance(input_text, str) else f"[{len(input_text)} items]", "model": model},
                output=output,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "litellm", "type": "embedding", **response_meta},
                model=response_meta.get("model", model),
                tokens_used=response_meta.get("tokens_used"),
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)

            return result

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            tracked_call = TrackedCall(
                function_name=name,
                input={"input": str(input_text)[:500], "model": model},
                output=None,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "litellm", "type": "embedding"},
                model=model,
                error=f"{type(e).__name__}: {str(e)}",
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)
            raise

    return embedding


def _create_aembedding_wrapper(
    original_fn: Callable,
    hone_client: "Hone",
    name: str = "LiteLLMEmbedding",
) -> Callable:
    """Create a wrapper for litellm.aembedding."""

    @functools.wraps(original_fn)
    async def aembedding(*args, **kwargs):
        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        input_text = kwargs.get("input", "")
        model = kwargs.get("model", "unknown")

        try:
            result = await original_fn(*args, **kwargs)

            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            response_meta = _extract_usage_metadata(result)

            # Don't store the actual embeddings (too large)
            output = {
                "model": getattr(result, "model", model),
                "usage": result.usage.model_dump() if hasattr(result, "usage") and result.usage else None,
                "data_count": len(result.data) if hasattr(result, "data") else 0,
            }

            tracked_call = TrackedCall(
                function_name=name,
                input={"input": input_text[:500] if isinstance(input_text, str) else f"[{len(input_text)} items]", "model": model},
                output=output,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "litellm", "type": "embedding", **response_meta},
                model=response_meta.get("model", model),
                tokens_used=response_meta.get("tokens_used"),
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)

            return result

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            tracked_call = TrackedCall(
                function_name=name,
                input={"input": str(input_text)[:500], "model": model},
                output=None,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "litellm", "type": "embedding"},
                model=model,
                error=f"{type(e).__name__}: {str(e)}",
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)
            raise

    return aembedding


def wrap_litellm(
    hone_client: "Hone",
    *,
    completion_name: str = "LiteLLM",
    embedding_name: str = "LiteLLMEmbedding",
) -> None:
    """Wrap LiteLLM's completion and embedding functions to automatically track all calls to Hone.

    This function patches the LiteLLM module globally. After calling this function,
    all LiteLLM calls in your application will be automatically tracked.

    Supports:
        - litellm.completion() - synchronous completion
        - litellm.acompletion() - async completion
        - litellm.embedding() - synchronous embedding
        - litellm.aembedding() - async embedding
        - Streaming responses
        - All LiteLLM-supported providers (OpenAI, Anthropic, Cohere, etc.)

    Args:
        hone_client: The Hone client to send tracking data to.
        completion_name: The name to use for completion calls in tracking.
        embedding_name: The name to use for embedding calls in tracking.

    Returns:
        None. LiteLLM is patched in place.

    Example:
        ```python
        import litellm
        from hone import Hone
        from hone.wrappers import wrap_litellm

        aix = Hone(api_key="aix_xxx", project_id="my-project")
        wrap_litellm(aix)

        # Now all litellm calls are automatically tracked
        response = litellm.completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )

        # Works with any provider LiteLLM supports
        response = litellm.completion(
            model="anthropic/claude-3-5-sonnet-latest",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        ```
    """
    global _original_completion, _original_acompletion, _original_embedding, _original_aembedding

    try:
        import litellm
    except ImportError:
        raise ImportError(
            "LiteLLM is not installed. Please install it with: pip install litellm"
        )

    # Store and wrap completion
    if _original_completion is None:
        _original_completion = litellm.completion
    litellm.completion = _create_completion_wrapper(_original_completion, hone_client, completion_name)

    # Store and wrap acompletion
    if _original_acompletion is None:
        _original_acompletion = litellm.acompletion
    litellm.acompletion = _create_acompletion_wrapper(_original_acompletion, hone_client, completion_name)

    # Store and wrap embedding
    if _original_embedding is None:
        _original_embedding = litellm.embedding
    litellm.embedding = _create_embedding_wrapper(_original_embedding, hone_client, embedding_name)

    # Store and wrap aembedding
    if _original_aembedding is None:
        _original_aembedding = litellm.aembedding
    litellm.aembedding = _create_aembedding_wrapper(_original_aembedding, hone_client, embedding_name)

    logger.info("LiteLLM wrapped successfully. All calls will be tracked to Hone.")


def unwrap_litellm() -> None:
    """Restore LiteLLM's original functions.

    Call this to stop tracking LiteLLM calls.
    """
    global _original_completion, _original_acompletion, _original_embedding, _original_aembedding

    try:
        import litellm
    except ImportError:
        return

    if _original_completion is not None:
        litellm.completion = _original_completion
        _original_completion = None

    if _original_acompletion is not None:
        litellm.acompletion = _original_acompletion
        _original_acompletion = None

    if _original_embedding is not None:
        litellm.embedding = _original_embedding
        _original_embedding = None

    if _original_aembedding is not None:
        litellm.aembedding = _original_aembedding
        _original_aembedding = None

    logger.info("LiteLLM unwrapped. Calls will no longer be tracked.")
