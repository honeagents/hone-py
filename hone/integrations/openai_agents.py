"""
OpenAI Agents SDK Integration for Hone.

Portions of this code adapted from LangSmith SDK
Copyright (c) 2023 LangChain
Licensed under MIT License
https://github.com/langchain-ai/langsmith-sdk

Modifications for Hone Platform:
- Replaced LangSmith client with Hone client
- Simplified tracing to work with Hone's TrackedCall model
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Any

if TYPE_CHECKING:
    from hone.client import Hone

from hone.models import TrackedCall

logger = logging.getLogger(__name__)

# Check if the agents SDK is available
try:
    from agents import tracing

    required = (
        "TracingProcessor",
        "Trace",
        "Span",
        "ResponseSpanData",
    )
    if not all(hasattr(tracing, name) for name in required):
        raise ImportError("The `agents` package is missing required attributes.")

    HAVE_AGENTS = True
except ImportError:
    HAVE_AGENTS = False


if not HAVE_AGENTS:

    class OpenAIAgentsTracingProcessor:
        """Tracing processor for the OpenAI Agents SDK.

        Traces all intermediate steps of your OpenAI Agent to Hone.

        Requirements: Make sure to install `pip install openai-agents`.

        Args:
            hone_client: An instance of `aix.Hone`.

        Example:
            ```python
            from agents import (
                Agent,
                Runner,
                function_tool,
                set_trace_processors,
            )
            from aix import Hone
            from hone.integrations import OpenAIAgentsTracingProcessor

            aix = Hone(api_key="hone_xxx", project_id="my-project")
            set_trace_processors([OpenAIAgentsTracingProcessor(hone_client=aix)])

            @function_tool
            def get_weather(city: str) -> str:
                return f"The weather in {city} is sunny"

            agent = Agent(
                name="Weather Agent",
                instructions="Help users with weather information",
                model="o3-mini",
                tools=[get_weather],
            )

            result = await Runner.run(agent, "What's the weather in San Francisco?")
            ```
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The `agents` package is not installed. "
                "Please install it with `pip install openai-agents`."
            )

else:

    def _extract_span_data(span: tracing.Span) -> Dict[str, Any]:
        """Extract relevant data from a span."""
        result: Dict[str, Any] = {}

        span_data = span.span_data
        if span_data is None:
            return result

        # Handle different span types
        if hasattr(span_data, "input"):
            result["inputs"] = {"input": span_data.input}

        if hasattr(span_data, "output"):
            result["outputs"] = {"output": span_data.output}

        if hasattr(span_data, "messages"):
            result["inputs"] = {"messages": span_data.messages}

        if hasattr(span_data, "response"):
            result["outputs"] = {"response": span_data.response}

        if hasattr(span_data, "model"):
            result["model"] = span_data.model

        return result

    def _get_run_type(span: tracing.Span) -> str:
        """Determine the run type based on span type."""
        span_data = span.span_data

        if isinstance(span_data, tracing.ResponseSpanData):
            return "llm"
        elif hasattr(tracing, "GenerationSpanData") and isinstance(span_data, tracing.GenerationSpanData):
            return "llm"
        elif hasattr(tracing, "FunctionSpanData") and isinstance(span_data, tracing.FunctionSpanData):
            return "tool"
        elif hasattr(tracing, "HandoffSpanData") and isinstance(span_data, tracing.HandoffSpanData):
            return "chain"
        else:
            return "chain"

    def _get_run_name(span: tracing.Span) -> str:
        """Get a name for the span."""
        if hasattr(span, "name") and span.name:
            return span.name

        span_data = span.span_data
        if hasattr(span_data, "name") and span_data.name:
            return span_data.name

        # Fallback based on span type
        if isinstance(span_data, tracing.ResponseSpanData):
            return "Response"
        elif hasattr(tracing, "GenerationSpanData") and isinstance(span_data, tracing.GenerationSpanData):
            return "Generation"
        elif hasattr(tracing, "FunctionSpanData") and isinstance(span_data, tracing.FunctionSpanData):
            return "Function"
        elif hasattr(tracing, "HandoffSpanData") and isinstance(span_data, tracing.HandoffSpanData):
            return "Handoff"
        else:
            return "Span"

    class OpenAIAgentsTracingProcessor(tracing.TracingProcessor):
        """Tracing processor for the OpenAI Agents SDK.

        Traces all intermediate steps of your OpenAI Agent to Hone.

        Args:
            hone_client: An instance of `aix.Hone`.
            metadata: Metadata to associate with all traces.
            tags: Tags to associate with all traces.
            name: Name of the root trace.

        Example:
            ```python
            from agents import (
                Agent,
                Runner,
                function_tool,
                set_trace_processors,
            )
            from aix import Hone
            from hone.integrations import OpenAIAgentsTracingProcessor

            aix = Hone(api_key="hone_xxx", project_id="my-project")
            set_trace_processors([OpenAIAgentsTracingProcessor(hone_client=aix)])

            @function_tool
            def get_weather(city: str) -> str:
                return f"The weather in {city} is sunny"

            agent = Agent(
                name="Weather Agent",
                instructions="Help users with weather information",
                model="o3-mini",
                tools=[get_weather],
            )

            result = await Runner.run(agent, "What's the weather in San Francisco?")
            ```
        """

        def __init__(
            self,
            hone_client: "Hone",
            *,
            metadata: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None,
            name: Optional[str] = None,
        ):
            self.hone_client = hone_client
            self._metadata = metadata or {}
            self._tags = tags or []
            self._name = name

            # Track active traces and spans
            self._traces: Dict[str, Dict[str, Any]] = {}
            self._spans: Dict[str, Dict[str, Any]] = {}

            # Store first/last inputs/outputs for traces
            self._first_inputs: Dict[str, Dict] = {}
            self._last_outputs: Dict[str, Dict] = {}

        def on_trace_start(self, trace: tracing.Trace) -> None:
            """Called when a trace starts."""
            trace_dict = trace.export() if hasattr(trace, "export") else {}

            # Determine name
            if self._name:
                name = self._name
            elif trace.name:
                name = trace.name
            else:
                name = "Agent workflow"

            self._traces[trace.trace_id] = {
                "name": name,
                "started_at": datetime.now(timezone.utc),
                "metadata": {
                    **self._metadata,
                    "openai_trace_id": trace.trace_id,
                },
                "tags": self._tags,
            }

            if trace_dict.get("group_id"):
                self._traces[trace.trace_id]["metadata"]["thread_id"] = trace_dict["group_id"]

        def on_trace_end(self, trace: tracing.Trace) -> None:
            """Called when a trace ends."""
            trace_info = self._traces.pop(trace.trace_id, None)
            if not trace_info:
                return

            try:
                inputs = self._first_inputs.pop(trace.trace_id, {})
                outputs = self._last_outputs.pop(trace.trace_id, {})

                # Calculate duration
                started_at = trace_info["started_at"]
                ended_at = datetime.now(timezone.utc)
                duration_ms = int((ended_at - started_at).total_seconds() * 1000)

                tracked_call = TrackedCall(
                    function_name=trace_info["name"],
                    input=inputs,
                    output=outputs,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={
                        "provider": "openai_agents",
                        "type": "trace",
                        **trace_info["metadata"],
                    },
                    project_id=self.hone_client.project_id,
                )

                self.hone_client._enqueue_call(tracked_call)

            except Exception as e:
                logger.exception(f"Error tracking trace end: {e}")

        def on_span_start(self, span: tracing.Span) -> None:
            """Called when a span starts."""
            self._spans[span.span_id] = {
                "started_at": datetime.now(timezone.utc),
                "trace_id": span.trace_id,
                "parent_id": span.parent_id,
            }

        def on_span_end(self, span: tracing.Span) -> None:
            """Called when a span ends."""
            span_info = self._spans.pop(span.span_id, None)
            if not span_info:
                return

            try:
                # Extract span data
                extracted = _extract_span_data(span)
                inputs = extracted.get("inputs", {})
                outputs = extracted.get("outputs", {})

                run_name = _get_run_name(span)
                run_type = _get_run_type(span)

                # Calculate duration
                started_at = span_info["started_at"]
                ended_at = datetime.now(timezone.utc)
                duration_ms = int((ended_at - started_at).total_seconds() * 1000)

                # Track first/last inputs/outputs for the trace
                if isinstance(span.span_data, tracing.ResponseSpanData):
                    trace_id = span.trace_id
                    if trace_id not in self._first_inputs:
                        self._first_inputs[trace_id] = inputs
                    self._last_outputs[trace_id] = outputs

                tracked_call = TrackedCall(
                    function_name=run_name,
                    input=inputs,
                    output=outputs,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    metadata={
                        "provider": "openai_agents",
                        "type": run_type,
                        "openai_trace_id": span.trace_id,
                        "openai_span_id": span.span_id,
                        "openai_parent_id": span.parent_id,
                        **self._metadata,
                    },
                    model=extracted.get("model"),
                    error=str(span.error) if span.error else None,
                    project_id=self.hone_client.project_id,
                )

                self.hone_client._enqueue_call(tracked_call)

            except Exception as e:
                logger.exception(f"Error tracking span end: {e}")

        def shutdown(self) -> None:
            """Flush any pending data."""
            self.hone_client.flush()

        def force_flush(self) -> None:
            """Force flush pending data."""
            self.hone_client.flush()
