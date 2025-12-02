"""
Tests for Hone SDK data models.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from hone.models import (
    TrackedCall,
    extract_openai_metadata,
    extract_anthropic_metadata,
)


class TestTrackedCall:
    """Tests for the TrackedCall dataclass."""

    def test_create_tracked_call_basic(self):
        """Test creating a basic TrackedCall."""
        now = datetime.now(timezone.utc)
        call = TrackedCall(
            function_name="test_func",
            input={"message": "Hello"},
            output="Hi!",
            duration_ms=100,
            started_at=now,
        )

        assert call.function_name == "test_func"
        assert call.input == {"message": "Hello"}
        assert call.output == "Hi!"
        assert call.duration_ms == 100
        assert call.started_at == now
        assert call.id is not None  # UUID should be auto-generated

    def test_create_tracked_call_with_all_fields(self):
        """Test creating a TrackedCall with all optional fields."""
        now = datetime.now(timezone.utc)
        call = TrackedCall(
            function_name="test_func",
            input={"messages": [{"role": "user", "content": "Hi"}]},
            output="Hello!",
            duration_ms=150,
            started_at=now,
            metadata={"version": "1.0"},
            model="gpt-4",
            tokens_used=100,
            cost_usd=0.01,
            messages=[{"role": "user", "content": "Hi"}],
            project_id="test-project",
        )

        assert call.metadata == {"version": "1.0"}
        assert call.model == "gpt-4"
        assert call.tokens_used == 100
        assert call.cost_usd == 0.01
        assert call.messages == [{"role": "user", "content": "Hi"}]
        assert call.project_id == "test-project"

    def test_tracked_call_ended_at_auto_set(self):
        """Test that ended_at is automatically set if not provided."""
        now = datetime.now(timezone.utc)
        call = TrackedCall(
            function_name="test_func",
            input={},
            output=None,
            duration_ms=0,
            started_at=now,
        )

        assert call.ended_at is not None
        assert isinstance(call.ended_at, datetime)

    def test_tracked_call_to_dict(self):
        """Test TrackedCall serialization to dictionary."""
        now = datetime.now(timezone.utc)
        call = TrackedCall(
            id="test-uuid",
            function_name="test_func",
            input={"key": "value"},
            output="result",
            duration_ms=100,
            started_at=now,
            model="gpt-4",
            tokens_used=50,
        )

        data = call.to_dict()

        assert data["id"] == "test-uuid"
        assert data["function_name"] == "test_func"
        assert data["input"] == {"key": "value"}
        assert data["output"] == "result"
        assert data["duration_ms"] == 100
        assert data["model"] == "gpt-4"
        assert data["tokens_used"] == 50
        assert "started_at" in data
        assert "ended_at" in data

    def test_tracked_call_from_dict(self):
        """Test TrackedCall deserialization from dictionary."""
        data = {
            "id": "test-uuid",
            "function_name": "test_func",
            "input": {"key": "value"},
            "output": "result",
            "duration_ms": 100,
            "started_at": "2024-01-01T12:00:00+00:00",
            "model": "gpt-4",
        }

        call = TrackedCall.from_dict(data)

        assert call.id == "test-uuid"
        assert call.function_name == "test_func"
        assert call.input == {"key": "value"}
        assert call.output == "result"
        assert call.duration_ms == 100
        assert call.model == "gpt-4"

    def test_serialize_output_primitives(self):
        """Test output serialization handles primitive types."""
        assert TrackedCall._serialize_output("string") == "string"
        assert TrackedCall._serialize_output(42) == 42
        assert TrackedCall._serialize_output(3.14) == 3.14
        assert TrackedCall._serialize_output(True) is True
        assert TrackedCall._serialize_output(None) is None

    def test_serialize_output_list(self):
        """Test output serialization handles lists."""
        result = TrackedCall._serialize_output(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_serialize_output_dict(self):
        """Test output serialization handles dictionaries."""
        result = TrackedCall._serialize_output({"key": "value"})
        assert result == {"key": "value"}

    def test_serialize_output_nested(self):
        """Test output serialization handles nested structures."""
        data = {"items": [{"id": 1}, {"id": 2}]}
        result = TrackedCall._serialize_output(data)
        assert result == {"items": [{"id": 1}, {"id": 2}]}


class TestExtractOpenAIMetadata:
    """Tests for OpenAI metadata extraction."""

    def test_extract_openai_metadata(self, mock_openai_response):
        """Test extracting metadata from OpenAI response."""
        metadata = extract_openai_metadata(mock_openai_response)

        assert metadata["model"] == "gpt-4"
        assert metadata["tokens_used"] == 150

    def test_extract_openai_metadata_no_usage(self):
        """Test extraction when response has no usage info."""
        response = MagicMock(spec=["model"])
        response.model = "gpt-4"

        metadata = extract_openai_metadata(response)

        assert metadata["model"] == "gpt-4"
        assert "tokens_used" not in metadata

    def test_extract_openai_metadata_empty_response(self):
        """Test extraction from empty response returns empty dict."""
        response = MagicMock(spec=[])
        metadata = extract_openai_metadata(response)
        assert metadata == {}


class TestExtractAnthropicMetadata:
    """Tests for Anthropic metadata extraction."""

    def test_extract_anthropic_metadata(self, mock_anthropic_response):
        """Test extracting metadata from Anthropic response."""
        metadata = extract_anthropic_metadata(mock_anthropic_response)

        assert metadata["model"] == "claude-3-sonnet"
        assert metadata["tokens_used"] == 150  # 50 + 100

    def test_extract_anthropic_metadata_no_usage(self):
        """Test extraction when response has no usage info."""
        response = MagicMock(spec=["model"])
        response.model = "claude-3"

        metadata = extract_anthropic_metadata(response)

        assert metadata["model"] == "claude-3"
        assert "tokens_used" not in metadata
