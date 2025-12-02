"""
Google Gemini Client Wrapper for Hone.

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

import base64
import functools
import json
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
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from hone.client import Hone
    from google import genai

from hone.models import TrackedCall

C = TypeVar("C", bound=Union["genai.Client", Any])
logger = logging.getLogger(__name__)


def _strip_none(d: dict) -> dict:
    """Remove None values from dictionary."""
    return {k: v for k, v in d.items() if v is not None}


def _convert_config_for_tracing(kwargs: dict) -> None:
    """Convert GenerateContentConfig to dict for compatibility."""
    if "config" in kwargs and not isinstance(kwargs["config"], dict):
        kwargs["config"] = vars(kwargs["config"])


def _process_gemini_inputs(inputs: dict) -> dict:
    """Process Gemini inputs to normalize them for tracking.

    Converts Gemini's content format to a standardized messages format.
    """
    contents = inputs.get("contents")
    if not contents:
        return inputs

    # Handle string input
    if isinstance(contents, str):
        return {
            "messages": [{"role": "user", "content": contents}],
            "model": inputs.get("model"),
            **({k: v for k, v in inputs.items() if k not in ("contents", "model")}),
        }

    # Handle list of content objects
    if isinstance(contents, list):
        # Simple list of strings
        if all(isinstance(item, str) for item in contents):
            return {
                "messages": [{"role": "user", "content": item} for item in contents],
                "model": inputs.get("model"),
                **({k: v for k, v in inputs.items() if k not in ("contents", "model")}),
            }

        # Complex multimodal case
        messages = []
        for content in contents:
            if isinstance(content, dict):
                role = content.get("role", "user")
                parts = content.get("parts", [])

                text_parts = []
                content_parts = []

                for part in parts:
                    if isinstance(part, dict):
                        if "text" in part and part["text"]:
                            text_parts.append(part["text"])
                            content_parts.append({"type": "text", "text": part["text"]})
                        elif "inline_data" in part:
                            inline_data = part["inline_data"]
                            mime_type = inline_data.get("mime_type", "image/jpeg")
                            data = inline_data.get("data", b"")

                            if isinstance(data, bytes):
                                data_b64 = base64.b64encode(data).decode("utf-8")
                            else:
                                data_b64 = data

                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{data_b64}",
                                    "detail": "high",
                                },
                            })
                        elif "function_call" in part or "functionCall" in part:
                            function_call = part.get("function_call") or part.get("functionCall")
                            if function_call is not None:
                                if not isinstance(function_call, dict):
                                    function_call = function_call.to_dict() if hasattr(function_call, "to_dict") else {}
                                content_parts.append({
                                    "type": "function_call",
                                    "function_call": {
                                        "id": function_call.get("id"),
                                        "name": function_call.get("name"),
                                        "arguments": function_call.get("args", {}),
                                    },
                                })
                    elif isinstance(part, str):
                        text_parts.append(part)
                        content_parts.append({"type": "text", "text": part})

                # Use simple string format for text-only
                if content_parts and all(p.get("type") == "text" for p in content_parts):
                    message_content = "\n".join(text_parts)
                else:
                    message_content = content_parts if content_parts else ""

                messages.append({"role": role, "content": message_content})

        return {
            "messages": messages,
            "model": inputs.get("model"),
            **({k: v for k, v in inputs.items() if k not in ("contents", "model")}),
        }

    return inputs


def _extract_usage_metadata(response: Any) -> Dict[str, Any]:
    """Extract usage metadata from Gemini response."""
    metadata = {}

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata

        if hasattr(usage, "prompt_token_count"):
            metadata["input_tokens"] = usage.prompt_token_count
        if hasattr(usage, "candidates_token_count"):
            metadata["output_tokens"] = usage.candidates_token_count
        if hasattr(usage, "total_token_count"):
            metadata["tokens_used"] = usage.total_token_count
        elif "input_tokens" in metadata and "output_tokens" in metadata:
            metadata["tokens_used"] = metadata["input_tokens"] + metadata["output_tokens"]

        # Check for cached content tokens
        if hasattr(usage, "cached_content_token_count"):
            metadata["cached_tokens"] = usage.cached_content_token_count

        # Check for thinking/reasoning tokens
        if hasattr(usage, "thoughts_token_count"):
            metadata["reasoning_tokens"] = usage.thoughts_token_count

    return metadata


def _process_gemini_response(response: Any) -> dict:
    """Process Gemini response for tracking."""
    try:
        if hasattr(response, "to_dict"):
            rdict = response.to_dict()
        elif hasattr(response, "model_dump"):
            rdict = response.model_dump()
        else:
            rdict = {"text": getattr(response, "text", str(response))}

        # Extract content from candidates
        content_result = ""
        content_parts = []
        finish_reason = None

        if "candidates" in rdict and rdict["candidates"]:
            candidate = rdict["candidates"][0]
            if "content" in candidate:
                content = candidate["content"]
                if "parts" in content and content["parts"]:
                    for part in content["parts"]:
                        if "text" in part and part["text"]:
                            content_result += part["text"]
                            content_parts.append({"type": "text", "text": part["text"]})
                        elif "inline_data" in part and part["inline_data"] is not None:
                            inline_data = part["inline_data"]
                            mime_type = inline_data.get("mime_type", "image/jpeg")
                            data = inline_data.get("data", b"")

                            if isinstance(data, bytes):
                                data_b64 = base64.b64encode(data).decode("utf-8")
                            else:
                                data_b64 = data

                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{data_b64}",
                                    "detail": "high",
                                },
                            })
                        elif "function_call" in part or "functionCall" in part:
                            function_call = part.get("function_call") or part.get("functionCall")
                            if function_call is not None:
                                if not isinstance(function_call, dict):
                                    function_call = function_call.to_dict() if hasattr(function_call, "to_dict") else {}
                                content_parts.append({
                                    "type": "function_call",
                                    "function_call": {
                                        "id": function_call.get("id"),
                                        "name": function_call.get("name"),
                                        "arguments": function_call.get("args", {}),
                                    },
                                })

            if "finish_reason" in candidate:
                finish_reason = candidate["finish_reason"]
        elif "text" in rdict:
            content_result = rdict["text"]
            content_parts.append({"type": "text", "text": content_result})

        # Build result in OpenAI-compatible format
        tool_calls = [p for p in content_parts if p.get("type") == "function_call"]
        if tool_calls:
            return {
                "content": content_result or None,
                "role": "assistant",
                "finish_reason": finish_reason,
                "tool_calls": [
                    {
                        "id": tc["function_call"].get("id") or f"call_{i}",
                        "type": "function",
                        "index": i,
                        "function": {
                            "name": tc["function_call"]["name"],
                            "arguments": json.dumps(tc["function_call"]["arguments"]),
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            }
        elif len(content_parts) > 1 or (content_parts and content_parts[0]["type"] != "text"):
            return {
                "content": content_parts,
                "role": "assistant",
                "finish_reason": finish_reason,
            }
        else:
            return {
                "content": content_result,
                "role": "assistant",
                "finish_reason": finish_reason,
            }

    except Exception as e:
        logger.debug(f"Error processing Gemini response: {e}")
        return {"output": str(response)}


def _reduce_streaming_chunks(all_chunks: list) -> dict:
    """Reduce streaming chunks into a single response."""
    if not all_chunks:
        return {"content": ""}

    full_text = ""
    last_chunk = None

    for chunk in all_chunks:
        try:
            if hasattr(chunk, "text") and chunk.text:
                full_text += chunk.text
            last_chunk = chunk
        except Exception as e:
            logger.debug(f"Error processing chunk: {e}")

    return {"content": full_text}


def _is_async(func: Callable) -> bool:
    """Check if a function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


def _get_generate_wrapper(
    original_generate: Callable,
    hone_client: "Hone",
    name: str = "ChatGoogleGenerativeAI",
    is_streaming: bool = False,
) -> Callable:
    """Create a wrapper for generate_content methods."""

    @functools.wraps(original_generate)
    def generate(*args, **kwargs):
        # Handle config object
        _convert_config_for_tracing(kwargs)
        clean_kwargs = _strip_none(kwargs)

        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        # Process inputs
        processed_inputs = _process_gemini_inputs(clean_kwargs)
        messages = processed_inputs.get("messages", [])
        model = clean_kwargs.get("model", "unknown")

        try:
            result = original_generate(*args, **kwargs)

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
                output = _process_gemini_response(result)

                tracked_call = TrackedCall(
                    function_name=name,
                    input=processed_inputs,
                    output=output,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={"provider": "google", **response_meta},
                    model=model,
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
                input=processed_inputs,
                output=None,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "google"},
                model=model,
                messages=messages,
                error=f"{type(e).__name__}: {str(e)}",
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)
            raise

    @functools.wraps(original_generate)
    async def agenerate(*args, **kwargs):
        # Handle config object
        _convert_config_for_tracing(kwargs)
        clean_kwargs = _strip_none(kwargs)

        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        # Process inputs
        processed_inputs = _process_gemini_inputs(clean_kwargs)
        messages = processed_inputs.get("messages", [])
        model = clean_kwargs.get("model", "unknown")

        try:
            result = await original_generate(*args, **kwargs)

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
                output = _process_gemini_response(result)

                tracked_call = TrackedCall(
                    function_name=name,
                    input=processed_inputs,
                    output=output,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={"provider": "google", **response_meta},
                    model=model,
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
                input=processed_inputs,
                output=None,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={"provider": "google"},
                model=model,
                messages=messages,
                error=f"{type(e).__name__}: {str(e)}",
                project_id=hone_client.project_id,
            )

            hone_client._enqueue_call(tracked_call)
            raise

    return agenerate if _is_async(original_generate) else generate


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

    # Extract usage from last chunk
    response_meta = {}
    if chunks:
        last_chunk = chunks[-1]
        response_meta = _extract_usage_metadata(last_chunk)

    tracked_call = TrackedCall(
        function_name=name,
        input={"messages": messages, "model": model},
        output=reduced,
        duration_ms=duration_ms,
        started_at=started_at,
        metadata={"provider": "google", "streaming": True, **response_meta},
        model=model,
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

    # Extract usage from last chunk
    response_meta = {}
    if chunks:
        last_chunk = chunks[-1]
        response_meta = _extract_usage_metadata(last_chunk)

    tracked_call = TrackedCall(
        function_name=name,
        input={"messages": messages, "model": model},
        output=reduced,
        duration_ms=duration_ms,
        started_at=started_at,
        metadata={"provider": "google", "streaming": True, **response_meta},
        model=model,
        tokens_used=response_meta.get("tokens_used"),
        messages=messages,
        project_id=hone_client.project_id,
    )

    hone_client._enqueue_call(tracked_call)


def wrap_gemini(
    client: C,
    *,
    hone_client: "Hone",
    chat_name: str = "ChatGoogleGenerativeAI",
) -> C:
    """Wrap a Google Gemini client to automatically track all calls to Hone.

    NOTE: This wrapper is in beta and targets the google-genai SDK.

    Supports:
        - generate_content and generate_content_stream methods
        - Sync and async clients
        - Streaming and non-streaming responses
        - Tool/function calling
        - Multimodal inputs (text + images)

    Args:
        client: The Google Gen AI client to wrap.
        hone_client: The Hone client to send tracking data to.
        chat_name: The name to use for chat calls in tracking.

    Returns:
        The wrapped client (same instance, modified in place).

    Example:
        ```python
        from google import genai
        from hone import Hone
        from hone.wrappers import wrap_gemini

        aix = Hone(api_key="aix_xxx", project_id="my-project")
        client = wrap_gemini(genai.Client(api_key="your-api-key"), hone_client=aix)

        # Now all calls are automatically tracked
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello!",
        )
        print(response.text)
        ```
    """
    # Check if already wrapped
    if (
        hasattr(client, "models")
        and hasattr(client.models, "generate_content")
        and hasattr(client.models.generate_content, "__wrapped__")
    ):
        raise ValueError(
            "This Google Gen AI client has already been wrapped. "
            "Wrapping a client multiple times is not supported."
        )

    # Wrap synchronous methods
    if hasattr(client, "models") and hasattr(client.models, "generate_content"):
        client.models.generate_content = _get_generate_wrapper(  # type: ignore[method-assign]
            client.models.generate_content,
            hone_client,
            chat_name,
            is_streaming=False,
        )

    if hasattr(client, "models") and hasattr(client.models, "generate_content_stream"):
        client.models.generate_content_stream = _get_generate_wrapper(  # type: ignore[method-assign]
            client.models.generate_content_stream,
            hone_client,
            chat_name,
            is_streaming=True,
        )

    # Wrap async methods (aio namespace)
    if (
        hasattr(client, "aio")
        and hasattr(client.aio, "models")
        and hasattr(client.aio.models, "generate_content")
    ):
        client.aio.models.generate_content = _get_generate_wrapper(  # type: ignore[method-assign]
            client.aio.models.generate_content,
            hone_client,
            chat_name,
            is_streaming=False,
        )

    if (
        hasattr(client, "aio")
        and hasattr(client.aio, "models")
        and hasattr(client.aio.models, "generate_content_stream")
    ):
        client.aio.models.generate_content_stream = _get_generate_wrapper(  # type: ignore[method-assign]
            client.aio.models.generate_content_stream,
            hone_client,
            chat_name,
            is_streaming=True,
        )

    return client
