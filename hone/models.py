"""
Data models for Hone SDK.

Portions of this code adapted from LangSmith SDK
Copyright (c) 2023 LangChain
Licensed under MIT License
https://github.com/langchain-ai/langsmith-sdk
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _generate_uuid() -> str:
    """Generate a unique identifier for tracked calls."""
    return str(uuid.uuid4())


@dataclass
class TrackedCall:
    """
    Represents a tracked LLM call.

    This dataclass captures all relevant information about an LLM call
    including inputs, outputs, timing, and optional metadata.
    """

    function_name: str
    input: Dict[str, Any]
    output: Any
    duration_ms: int
    started_at: datetime
    id: str = field(default_factory=_generate_uuid)
    ended_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    messages: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    project_id: Optional[str] = None

    def __post_init__(self):
        """Set ended_at if not provided."""
        if self.ended_at is None:
            self.ended_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TrackedCall to a dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "id": self.id,
            "function_name": self.function_name,
            "input": self.input,
            "output": self._serialize_output(self.output),
            "duration_ms": self.duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "metadata": self.metadata,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "messages": self.messages,
            "error": self.error,
            "project_id": self.project_id,
        }

    @staticmethod
    def _serialize_output(output: Any) -> Any:
        """
        Serialize output to a JSON-compatible format.

        Handles common LLM response objects from OpenAI, Anthropic, etc.
        """
        if output is None:
            return None

        # Handle primitive types
        if isinstance(output, (str, int, float, bool)):
            return output

        # Handle list/tuple
        if isinstance(output, (list, tuple)):
            return [TrackedCall._serialize_output(item) for item in output]

        # Handle dict
        if isinstance(output, dict):
            return {k: TrackedCall._serialize_output(v) for k, v in output.items()}

        # Handle objects with __dict__ (like OpenAI responses)
        if hasattr(output, "__dict__"):
            try:
                return {k: TrackedCall._serialize_output(v) for k, v in output.__dict__.items() if not k.startswith("_")}
            except Exception:
                return str(output)

        # Handle objects with model_dump (Pydantic v2)
        if hasattr(output, "model_dump"):
            try:
                return output.model_dump()
            except Exception:
                return str(output)

        # Handle objects with dict method (Pydantic v1)
        if hasattr(output, "dict"):
            try:
                return output.dict()
            except Exception:
                return str(output)

        # Fallback to string representation
        return str(output)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackedCall":
        """
        Create a TrackedCall from a dictionary.

        Args:
            data: Dictionary with TrackedCall fields.

        Returns:
            TrackedCall instance.
        """
        started_at = data.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))

        ended_at = data.get("ended_at")
        if isinstance(ended_at, str):
            ended_at = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))

        return cls(
            id=data.get("id", _generate_uuid()),
            function_name=data["function_name"],
            input=data["input"],
            output=data["output"],
            duration_ms=data["duration_ms"],
            started_at=started_at,
            ended_at=ended_at,
            metadata=data.get("metadata"),
            model=data.get("model"),
            tokens_used=data.get("tokens_used"),
            cost_usd=data.get("cost_usd"),
            messages=data.get("messages"),
            error=data.get("error"),
            project_id=data.get("project_id"),
        )


def extract_openai_metadata(response: Any) -> Dict[str, Any]:
    """
    Extract metadata from OpenAI-style response objects.

    Args:
        response: OpenAI API response object.

    Returns:
        Dictionary with extracted metadata (model, tokens_used, etc.).
    """
    metadata = {}

    # Extract model
    if hasattr(response, "model"):
        metadata["model"] = response.model

    # Extract usage/tokens
    if hasattr(response, "usage"):
        usage = response.usage
        if hasattr(usage, "total_tokens"):
            metadata["tokens_used"] = usage.total_tokens
        elif hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
            # Anthropic style
            metadata["tokens_used"] = usage.input_tokens + usage.output_tokens

    return metadata


def extract_anthropic_metadata(response: Any) -> Dict[str, Any]:
    """
    Extract metadata from Anthropic-style response objects.

    Args:
        response: Anthropic API response object.

    Returns:
        Dictionary with extracted metadata.
    """
    metadata = {}

    if hasattr(response, "model"):
        metadata["model"] = response.model

    if hasattr(response, "usage"):
        usage = response.usage
        if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
            metadata["tokens_used"] = usage.input_tokens + usage.output_tokens

    return metadata
