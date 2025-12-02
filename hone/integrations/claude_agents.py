"""
Claude Agent SDK Integration for Hone.

Portions of this code adapted from LangSmith SDK
Copyright (c) 2023 LangChain
Licensed under MIT License
https://github.com/langchain-ai/langsmith-sdk

Modifications for Hone Platform:
- Replaced LangSmith client with Hone client
- Simplified tracing to work with Hone's TrackedCall model
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from hone.client import Hone

from hone.models import TrackedCall

logger = logging.getLogger(__name__)

# Module-level configuration
_hone_client: Optional["Hone"] = None
_config: Dict[str, Any] = {}


def set_tracing_config(
    hone_client: "Hone",
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> None:
    """Set the tracing configuration."""
    global _hone_client, _config
    _hone_client = hone_client
    _config = {
        "name": name,
        "metadata": metadata or {},
        "tags": tags or [],
    }


def _wrap_conversation_stream(original_method: Callable) -> Callable:
    """Wrap the conversation_stream method to add tracing."""

    @functools.wraps(original_method)
    async def wrapped(self, *args, **kwargs):
        global _hone_client, _config

        if _hone_client is None:
            # Not configured, just call original
            async for item in original_method(self, *args, **kwargs):
                yield item
            return

        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        # Extract input from args/kwargs
        messages = kwargs.get("messages", args[0] if args else [])

        # Track conversation turns
        collected_output = []
        tool_calls = []
        model = None

        try:
            async for event in original_method(self, *args, **kwargs):
                yield event

                # Collect relevant events
                if hasattr(event, "type"):
                    if event.type == "content":
                        collected_output.append({
                            "type": "content",
                            "text": getattr(event, "text", ""),
                        })
                    elif event.type == "tool_use":
                        tool_calls.append({
                            "name": getattr(event, "name", "unknown"),
                            "input": getattr(event, "input", {}),
                        })
                    elif event.type == "model":
                        model = getattr(event, "model", None)

            # Track the completed conversation
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            output = {
                "content": collected_output,
                "tool_calls": tool_calls if tool_calls else None,
            }

            name = _config.get("name", "ClaudeAgent")

            tracked_call = TrackedCall(
                function_name=name,
                input={"messages": messages},
                output=output,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={
                    "provider": "anthropic_agents",
                    "type": "conversation",
                    **_config.get("metadata", {}),
                },
                model=model,
                messages=messages if isinstance(messages, list) else None,
                project_id=_hone_client.project_id,
            )

            _hone_client._enqueue_call(tracked_call)

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            name = _config.get("name", "ClaudeAgent")

            tracked_call = TrackedCall(
                function_name=name,
                input={"messages": messages},
                output=None,
                duration_ms=duration_ms,
                started_at=started_at,
                metadata={
                    "provider": "anthropic_agents",
                    "type": "conversation",
                    **_config.get("metadata", {}),
                },
                messages=messages if isinstance(messages, list) else None,
                error=f"{type(e).__name__}: {str(e)}",
                project_id=_hone_client.project_id,
            )

            _hone_client._enqueue_call(tracked_call)
            raise

    return wrapped


def _instrument_claude_client(original_class: type) -> type:
    """Instrument the ClaudeSDKClient class with tracing."""

    class InstrumentedClaudeSDKClient(original_class):
        """ClaudeSDKClient with Hone tracing enabled."""

        async def conversation_stream(self, *args, **kwargs):
            """Wrapped conversation_stream with tracing."""
            global _hone_client, _config

            if _hone_client is None:
                # Not configured, just call original
                async for item in super().conversation_stream(*args, **kwargs):
                    yield item
                return

            start_time = time.perf_counter()
            started_at = datetime.now(timezone.utc)

            messages = kwargs.get("messages", args[0] if args else [])
            collected_output = []
            tool_calls = []
            model = None

            try:
                async for event in super().conversation_stream(*args, **kwargs):
                    yield event

                    if hasattr(event, "type"):
                        if event.type == "content":
                            collected_output.append({
                                "type": "content",
                                "text": getattr(event, "text", ""),
                            })
                        elif event.type == "tool_use":
                            tool_calls.append({
                                "name": getattr(event, "name", "unknown"),
                                "input": getattr(event, "input", {}),
                            })
                        elif event.type == "model":
                            model = getattr(event, "model", None)

                end_time = time.perf_counter()
                duration_ms = int((end_time - start_time) * 1000)

                output = {
                    "content": collected_output,
                    "tool_calls": tool_calls if tool_calls else None,
                }

                name = _config.get("name", "ClaudeAgent")

                tracked_call = TrackedCall(
                    function_name=name,
                    input={"messages": messages},
                    output=output,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={
                        "provider": "anthropic_agents",
                        "type": "conversation",
                        **_config.get("metadata", {}),
                    },
                    model=model,
                    messages=messages if isinstance(messages, list) else None,
                    project_id=_hone_client.project_id,
                )

                _hone_client._enqueue_call(tracked_call)

            except Exception as e:
                end_time = time.perf_counter()
                duration_ms = int((end_time - start_time) * 1000)

                name = _config.get("name", "ClaudeAgent")

                tracked_call = TrackedCall(
                    function_name=name,
                    input={"messages": messages},
                    output=None,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={
                        "provider": "anthropic_agents",
                        "type": "conversation",
                        **_config.get("metadata", {}),
                    },
                    messages=messages if isinstance(messages, list) else None,
                    error=f"{type(e).__name__}: {str(e)}",
                    project_id=_hone_client.project_id,
                )

                _hone_client._enqueue_call(tracked_call)
                raise

    return InstrumentedClaudeSDKClient


def configure_claude_agent_sdk(
    hone_client: "Hone",
    *,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> bool:
    """Enable Hone tracing for the Claude Agent SDK.

    This function instruments the Claude Agent SDK to automatically trace:
    - Chain runs for each conversation stream
    - All tool calls
    - Model responses

    Args:
        hone_client: The Hone client to send tracking data to.
        name: Name of the root trace.
        metadata: Metadata to associate with all traces.
        tags: Tags to associate with all traces.

    Returns:
        True if configuration was successful, False otherwise.

    Example:
        ```python
        from aix import Hone
        from hone.integrations import configure_claude_agent_sdk

        aix = Hone(api_key="hone_xxx", project_id="my-project")
        configure_claude_agent_sdk(aix)

        # Now use claude_agent_sdk as normal - tracing is automatic
        from claude_agent_sdk import ClaudeSDKClient

        client = ClaudeSDKClient()
        async for event in client.conversation_stream(messages=[...]):
            print(event)
        ```
    """
    try:
        import claude_agent_sdk
    except ImportError:
        logger.warning("Claude Agent SDK not installed.")
        return False

    if not hasattr(claude_agent_sdk, "ClaudeSDKClient"):
        logger.warning("Claude Agent SDK missing ClaudeSDKClient.")
        return False

    # Set configuration
    set_tracing_config(
        hone_client=hone_client,
        name=name,
        metadata=metadata,
        tags=tags,
    )

    # Get original class
    original = getattr(claude_agent_sdk, "ClaudeSDKClient", None)
    if not original:
        return False

    # Create instrumented class
    instrumented = _instrument_claude_client(original)

    # Replace in the module
    setattr(claude_agent_sdk, "ClaudeSDKClient", instrumented)

    # Replace in any modules that have already imported it
    for module in list(sys.modules.values()):
        try:
            if module and getattr(module, "ClaudeSDKClient", None) is original:
                setattr(module, "ClaudeSDKClient", instrumented)
        except Exception:
            continue

    logger.info("Claude Agent SDK configured for Hone tracing.")
    return True
