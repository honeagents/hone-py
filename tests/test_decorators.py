"""
Tests for Hone SDK decorators.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from hone.decorators import (
    TrackDecorator,
    create_track_decorator,
    _get_function_args,
    _extract_messages,
    _extract_response_metadata,
)


class TestGetFunctionArgs:
    """Tests for _get_function_args helper."""

    def test_get_args_simple_function(self):
        """Test extracting args from simple function."""
        def func(a, b):
            pass

        result = _get_function_args(func, (1, 2), {})
        assert result == {"a": 1, "b": 2}

    def test_get_args_with_kwargs(self):
        """Test extracting args with keyword arguments."""
        def func(a, b, c=3):
            pass

        result = _get_function_args(func, (1,), {"b": 2})
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_get_args_with_defaults(self):
        """Test extracting args applies defaults."""
        def func(a, b=10, c=20):
            pass

        result = _get_function_args(func, (1,), {})
        assert result == {"a": 1, "b": 10, "c": 20}


class TestExtractMessages:
    """Tests for _extract_messages helper."""

    def test_extract_messages_direct(self):
        """Test extracting messages from direct messages key."""
        input_args = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = _extract_messages(input_args)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_extract_messages_nested(self):
        """Test extracting messages from nested dict."""
        input_args = {
            "kwargs": {
                "messages": [{"role": "user", "content": "Hi"}],
            },
        }

        result = _extract_messages(input_args)
        assert result == [{"role": "user", "content": "Hi"}]

    def test_extract_messages_not_found(self):
        """Test returns None when messages not found."""
        input_args = {"text": "Hello"}
        result = _extract_messages(input_args)
        assert result is None


class TestExtractResponseMetadata:
    """Tests for _extract_response_metadata helper."""

    def test_extract_openai_response(self, mock_openai_response):
        """Test extracting metadata from OpenAI response."""
        metadata = _extract_response_metadata(mock_openai_response)
        assert metadata["model"] == "gpt-4"
        assert metadata["tokens_used"] == 150

    def test_extract_anthropic_response(self, mock_anthropic_response):
        """Test extracting metadata from Anthropic response."""
        metadata = _extract_response_metadata(mock_anthropic_response)
        assert metadata["model"] == "claude-3-sonnet"

    def test_extract_empty_response(self):
        """Test extracting from response with no metadata."""
        response = MagicMock(spec=[])
        metadata = _extract_response_metadata(response)
        assert metadata == {}


class TestTrackDecorator:
    """Tests for TrackDecorator class."""

    def test_decorator_wraps_sync_function(self):
        """Test decorator wraps synchronous function."""
        mock_client = MagicMock()
        mock_client.project_id = "test"

        decorator = TrackDecorator(client=mock_client)

        @decorator
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_decorator_tracks_call(self):
        """Test decorator enqueues tracked call."""
        mock_client = MagicMock()
        mock_client.project_id = "test"

        decorator = TrackDecorator(client=mock_client)

        @decorator
        def my_func(x):
            return x * 2

        result = my_func(5)

        # Verify _enqueue_call was called
        mock_client._enqueue_call.assert_called_once()
        call = mock_client._enqueue_call.call_args[0][0]
        assert call.function_name == "my_func"
        assert call.output == 10

    def test_decorator_with_custom_name(self):
        """Test decorator uses custom name."""
        mock_client = MagicMock()
        mock_client.project_id = "test"

        decorator = TrackDecorator(client=mock_client, name="custom-name")

        @decorator
        def my_func(x):
            return x * 2

        my_func(5)

        call = mock_client._enqueue_call.call_args[0][0]
        assert call.function_name == "custom-name"

    def test_decorator_with_metadata(self):
        """Test decorator includes custom metadata."""
        mock_client = MagicMock()
        mock_client.project_id = "test"

        decorator = TrackDecorator(
            client=mock_client,
            metadata={"version": "1.0"},
        )

        @decorator
        def my_func(x):
            return x * 2

        my_func(5)

        call = mock_client._enqueue_call.call_args[0][0]
        assert "version" in call.metadata
        assert call.metadata["version"] == "1.0"

    def test_decorator_captures_duration(self):
        """Test decorator captures execution duration."""
        mock_client = MagicMock()
        mock_client.project_id = "test"

        decorator = TrackDecorator(client=mock_client)

        @decorator
        def slow_func():
            import time
            time.sleep(0.01)  # 10ms
            return "done"

        slow_func()

        call = mock_client._enqueue_call.call_args[0][0]
        assert call.duration_ms >= 10

    def test_decorator_captures_exception(self):
        """Test decorator captures exception information."""
        mock_client = MagicMock()
        mock_client.project_id = "test"

        decorator = TrackDecorator(client=mock_client)

        @decorator
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()

        call = mock_client._enqueue_call.call_args[0][0]
        assert call.error is not None
        assert "ValueError" in call.error
        assert "Test error" in call.error

    def test_decorator_preserves_function_name(self):
        """Test decorator preserves original function metadata."""
        mock_client = MagicMock()
        mock_client.project_id = "test"

        decorator = TrackDecorator(client=mock_client)

        @decorator
        def my_documented_func():
            """This is my docstring."""
            return 42

        assert my_documented_func.__name__ == "my_documented_func"
        assert my_documented_func.__doc__ == "This is my docstring."


class TestTrackDecoratorAsync:
    """Tests for async function decoration."""

    @pytest.mark.asyncio
    async def test_decorator_wraps_async_function(self):
        """Test decorator wraps async function."""
        mock_client = MagicMock()
        mock_client.project_id = "test"

        decorator = TrackDecorator(client=mock_client)

        @decorator
        async def my_async_func(x):
            await asyncio.sleep(0.001)
            return x * 2

        result = await my_async_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_decorator_tracks_async_call(self):
        """Test decorator tracks async function call."""
        mock_client = MagicMock()
        mock_client.project_id = "test"

        decorator = TrackDecorator(client=mock_client)

        @decorator
        async def my_async_func(x):
            return x * 2

        await my_async_func(5)

        mock_client._enqueue_call.assert_called_once()
        call = mock_client._enqueue_call.call_args[0][0]
        assert call.function_name == "my_async_func"
        assert call.output == 10

    @pytest.mark.asyncio
    async def test_decorator_captures_async_exception(self):
        """Test decorator captures exception from async function."""
        mock_client = MagicMock()
        mock_client.project_id = "test"

        decorator = TrackDecorator(client=mock_client)

        @decorator
        async def failing_async_func():
            raise RuntimeError("Async error")

        with pytest.raises(RuntimeError):
            await failing_async_func()

        call = mock_client._enqueue_call.call_args[0][0]
        assert call.error is not None
        assert "RuntimeError" in call.error


class TestCreateTrackDecorator:
    """Tests for create_track_decorator function."""

    def test_create_decorator(self):
        """Test creating a track decorator."""
        mock_client = MagicMock()
        decorator = create_track_decorator(client=mock_client)
        assert callable(decorator)

    def test_create_decorator_with_options(self):
        """Test creating decorator with options."""
        mock_client = MagicMock()
        decorator = create_track_decorator(
            client=mock_client,
            name="custom",
            metadata={"key": "value"},
        )
        assert callable(decorator)
