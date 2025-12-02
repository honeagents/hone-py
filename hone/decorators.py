"""
Decorators for tracking LLM calls.

Portions of this code adapted from LangSmith SDK
Copyright (c) 2023 LangChain
Licensed under MIT License
https://github.com/langchain-ai/langsmith-sdk
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
import traceback
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    TypeVar,
    Union,
    overload,
)

from hone.models import TrackedCall, extract_openai_metadata, extract_anthropic_metadata

if TYPE_CHECKING:
    from hone.client import Hone

F = TypeVar("F", bound=Callable[..., Any])


def _get_function_args(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """
    Extract function arguments as a dictionary.

    Args:
        func: The function being called.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        Dictionary of argument names to values.
    """
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _extract_response_metadata(response: Any) -> Dict[str, Any]:
    """
    Extract metadata from LLM response objects.

    Supports OpenAI and Anthropic response formats.

    Args:
        response: LLM API response object.

    Returns:
        Dictionary with extracted metadata.
    """
    metadata = {}

    # Try OpenAI format first
    openai_meta = extract_openai_metadata(response)
    if openai_meta:
        metadata.update(openai_meta)

    # Try Anthropic format
    anthropic_meta = extract_anthropic_metadata(response)
    if anthropic_meta:
        metadata.update(anthropic_meta)

    return metadata


def _extract_messages(input_args: Dict[str, Any]) -> Optional[list]:
    """
    Extract messages array from input arguments.

    Args:
        input_args: Dictionary of function arguments.

    Returns:
        Messages list if found, None otherwise.
    """
    # Check for messages directly
    if "messages" in input_args:
        return input_args["messages"]

    # Check for nested in kwargs
    for value in input_args.values():
        if isinstance(value, dict) and "messages" in value:
            return value["messages"]

    return None


class TrackDecorator:
    """
    Decorator class for tracking function calls.

    This class wraps functions to capture their inputs, outputs, timing,
    and metadata. It supports both sync and async functions.
    """

    def __init__(
        self,
        client: "Hone",
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the track decorator.

        Args:
            client: Hone client instance.
            name: Optional custom name for the tracked call.
            metadata: Optional additional metadata.
        """
        self._client = client
        self._name = name
        self._metadata = metadata or {}

    def __call__(self, func: F) -> F:
        """
        Decorate a function for tracking.

        Args:
            func: Function to decorate.

        Returns:
            Decorated function.
        """
        if asyncio.iscoroutinefunction(func):
            return self._wrap_async(func)  # type: ignore
        else:
            return self._wrap_sync(func)  # type: ignore

    def _wrap_sync(self, func: Callable) -> Callable:
        """Wrap a synchronous function."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function name
            func_name = self._name or func.__name__

            # Capture start time
            start_time = time.perf_counter()
            started_at = datetime.now(timezone.utc)

            # Get input arguments
            try:
                input_args = _get_function_args(func, args, kwargs)
            except Exception:
                input_args = {"args": args, "kwargs": kwargs}

            # Extract messages
            messages = _extract_messages(input_args)

            # Execute function
            error: Optional[str] = None
            output: Any = None

            try:
                output = func(*args, **kwargs)
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                raise
            finally:
                # Calculate duration
                end_time = time.perf_counter()
                duration_ms = int((end_time - start_time) * 1000)

                # Extract response metadata
                response_meta = {}
                if output is not None and error is None:
                    response_meta = _extract_response_metadata(output)

                # Create tracked call
                tracked_call = TrackedCall(
                    function_name=func_name,
                    input=input_args,
                    output=output if error is None else None,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={**self._metadata, **response_meta.get("metadata", {})},
                    model=response_meta.get("model"),
                    tokens_used=response_meta.get("tokens_used"),
                    messages=messages,
                    error=error,
                    project_id=self._client.project_id,
                )

                # Enqueue for upload
                self._client._enqueue_call(tracked_call)

            return output

        return wrapper

    def _wrap_async(self, func: Callable) -> Callable:
        """Wrap an asynchronous function."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function name
            func_name = self._name or func.__name__

            # Capture start time
            start_time = time.perf_counter()
            started_at = datetime.now(timezone.utc)

            # Get input arguments
            try:
                input_args = _get_function_args(func, args, kwargs)
            except Exception:
                input_args = {"args": args, "kwargs": kwargs}

            # Extract messages
            messages = _extract_messages(input_args)

            # Execute function
            error: Optional[str] = None
            output: Any = None

            try:
                output = await func(*args, **kwargs)
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                raise
            finally:
                # Calculate duration
                end_time = time.perf_counter()
                duration_ms = int((end_time - start_time) * 1000)

                # Extract response metadata
                response_meta = {}
                if output is not None and error is None:
                    response_meta = _extract_response_metadata(output)

                # Create tracked call
                tracked_call = TrackedCall(
                    function_name=func_name,
                    input=input_args,
                    output=output if error is None else None,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={**self._metadata, **response_meta.get("metadata", {})},
                    model=response_meta.get("model"),
                    tokens_used=response_meta.get("tokens_used"),
                    messages=messages,
                    error=error,
                    project_id=self._client.project_id,
                )

                # Enqueue for upload
                self._client._enqueue_call(tracked_call)

            return output

        return wrapper


def create_track_decorator(
    client: "Hone",
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Create a track decorator for the given client.

    Args:
        client: Hone client instance.
        name: Optional custom name for tracked calls.
        metadata: Optional additional metadata.

    Returns:
        Decorator function.
    """
    decorator = TrackDecorator(client, name=name, metadata=metadata)
    return decorator
