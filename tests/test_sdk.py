"""
Tests for Hone Python SDK.

Following TDD approach - these tests define the expected behavior
before implementation.
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

# Import SDK components (to be implemented)
from hone import Hone
from hone.models import TrackedCall
from hone.exceptions import AIXError, AIXConnectionError, AIXValidationError


class TestTrackedCallModel:
    """Tests for TrackedCall dataclass."""

    def test_tracked_call_creation(self):
        """TrackedCall can be created with required fields."""
        call = TrackedCall(
            function_name="my_function",
            input={"message": "hello"},
            output="world",
            duration_ms=150,
            started_at=datetime.now(timezone.utc),
        )
        assert call.function_name == "my_function"
        assert call.input == {"message": "hello"}
        assert call.output == "world"
        assert call.duration_ms == 150
        assert call.id is not None  # Auto-generated UUID

    def test_tracked_call_with_metadata(self):
        """TrackedCall can include optional metadata."""
        call = TrackedCall(
            function_name="my_function",
            input={"query": "test"},
            output="response",
            duration_ms=100,
            started_at=datetime.now(timezone.utc),
            metadata={"user_id": "123", "session": "abc"},
            model="gpt-4",
            tokens_used=150,
            cost_usd=0.003,
        )
        assert call.metadata == {"user_id": "123", "session": "abc"}
        assert call.model == "gpt-4"
        assert call.tokens_used == 150
        assert call.cost_usd == 0.003

    def test_tracked_call_to_dict(self):
        """TrackedCall can be serialized to dictionary."""
        call = TrackedCall(
            function_name="test_func",
            input={"test": "input"},
            output="test output",
            duration_ms=50,
            started_at=datetime.now(timezone.utc),
        )
        data = call.to_dict()
        assert isinstance(data, dict)
        assert data["function_name"] == "test_func"
        assert data["input"] == {"test": "input"}
        assert data["output"] == "test output"
        assert "id" in data
        assert "started_at" in data


class TestAIXClient:
    """Tests for main Hone client class."""

    def test_client_initialization(self):
        """Hone client initializes with required parameters."""
        client = Hone(api_key="hone_test_key", project_id="test-project")
        assert client.api_key == "hone_test_key"
        assert client.project_id == "test-project"
        assert client.api_url == "http://localhost:8000"

    def test_client_custom_url(self):
        """Hone client accepts custom API URL."""
        client = Hone(
            api_key="hone_test_key",
            project_id="test-project",
            api_url="https://api.custom.com",
        )
        assert client.api_url == "https://api.custom.com"

    def test_client_has_track_method(self):
        """Hone client has track method for decorator."""
        client = Hone(api_key="hone_test_key", project_id="test-project")
        assert hasattr(client, "track")
        assert callable(client.track)

    def test_client_has_flush_method(self):
        """Hone client has flush method."""
        client = Hone(api_key="hone_test_key", project_id="test-project")
        assert hasattr(client, "flush")
        assert callable(client.flush)

    def test_client_has_shutdown_method(self):
        """Hone client has shutdown method."""
        client = Hone(api_key="hone_test_key", project_id="test-project")
        assert hasattr(client, "shutdown")
        assert callable(client.shutdown)


class TestTrackDecorator:
    """Tests for @hone.track() decorator."""

    def test_track_decorator_captures_call(self):
        """Decorator captures function call data."""
        client = Hone(api_key="hone_test_key", project_id="test-project")
        captured_calls: List[TrackedCall] = []

        # Mock the internal queue to capture calls
        original_enqueue = client._enqueue_call
        def mock_enqueue(call: TrackedCall):
            captured_calls.append(call)
            return original_enqueue(call)
        client._enqueue_call = mock_enqueue

        @client.track()
        def my_function(message: str) -> str:
            return f"Hello, {message}!"

        result = my_function("World")

        assert result == "Hello, World!"
        assert len(captured_calls) == 1
        assert captured_calls[0].function_name == "my_function"
        assert "message" in str(captured_calls[0].input) or captured_calls[0].input.get("message") == "World"
        assert captured_calls[0].output == "Hello, World!"

        client.shutdown()

    def test_track_decorator_returns_result(self):
        """Decorated function returns original result."""
        client = Hone(api_key="hone_test_key", project_id="test-project")

        @client.track()
        def add_numbers(a: int, b: int) -> int:
            return a + b

        result = add_numbers(5, 3)
        assert result == 8

        client.shutdown()

    def test_track_decorator_with_custom_name(self):
        """Decorator accepts custom name parameter."""
        client = Hone(api_key="hone_test_key", project_id="test-project")
        captured_calls: List[TrackedCall] = []

        original_enqueue = client._enqueue_call
        def mock_enqueue(call: TrackedCall):
            captured_calls.append(call)
            return original_enqueue(call)
        client._enqueue_call = mock_enqueue

        @client.track(name="custom_operation")
        def some_function():
            return "result"

        some_function()
        assert len(captured_calls) == 1
        assert captured_calls[0].function_name == "custom_operation"

        client.shutdown()

    def test_track_decorator_captures_duration(self):
        """Decorator captures execution duration."""
        client = Hone(api_key="hone_test_key", project_id="test-project")
        captured_calls: List[TrackedCall] = []

        original_enqueue = client._enqueue_call
        def mock_enqueue(call: TrackedCall):
            captured_calls.append(call)
            return original_enqueue(call)
        client._enqueue_call = mock_enqueue

        @client.track()
        def slow_function():
            time.sleep(0.1)  # Sleep 100ms
            return "done"

        slow_function()
        assert len(captured_calls) == 1
        assert captured_calls[0].duration_ms >= 100

        client.shutdown()

    def test_track_decorator_handles_exceptions(self):
        """Decorator handles exceptions and still tracks the call."""
        client = Hone(api_key="hone_test_key", project_id="test-project")
        captured_calls: List[TrackedCall] = []

        original_enqueue = client._enqueue_call
        def mock_enqueue(call: TrackedCall):
            captured_calls.append(call)
            return original_enqueue(call)
        client._enqueue_call = mock_enqueue

        @client.track()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        assert len(captured_calls) == 1
        assert captured_calls[0].error is not None
        assert "Test error" in str(captured_calls[0].error)

        client.shutdown()


class TestAsyncTrackDecorator:
    """Tests for @hone.track() with async functions."""

    @pytest.mark.asyncio
    async def test_track_async_function(self):
        """Decorator works with async functions."""
        client = Hone(api_key="hone_test_key", project_id="test-project")
        captured_calls: List[TrackedCall] = []

        original_enqueue = client._enqueue_call
        def mock_enqueue(call: TrackedCall):
            captured_calls.append(call)
            return original_enqueue(call)
        client._enqueue_call = mock_enqueue

        @client.track()
        async def async_function(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2

        result = await async_function(5)
        assert result == 10
        assert len(captured_calls) == 1
        assert captured_calls[0].function_name == "async_function"

        client.shutdown()


class TestBackgroundThread:
    """Tests for background batch upload thread."""

    def test_background_thread_batches(self):
        """Calls are batched before upload."""
        # For this test, we verify batching by checking queue behavior
        client = Hone(
            api_key="hone_test_key",
            project_id="test-project",
            batch_size=5,
            flush_interval=10.0,  # Long interval to ensure batching
        )

        batches_captured: List[int] = []

        # Create original send function reference before patching
        original_send = client._send_to_api

        def mock_send(batch):
            batches_captured.append(len(batch))
            return True  # Simulate successful upload

        # Patch at the client level
        client._send_to_api = mock_send

        @client.track()
        def simple_func(x):
            return x

        # Make 10 calls
        for i in range(10):
            simple_func(i)

        # Force flush to process all pending calls
        client.flush(timeout=10.0)
        time.sleep(0.2)  # Give background thread time to process

        client.shutdown()

        # Should have processed calls in batches (batch_size=5)
        total_calls = sum(batches_captured)
        assert total_calls == 10, f"Expected 10 calls, got {total_calls}"

    def test_flush_sends_immediately(self):
        """flush() sends all pending calls."""
        client = Hone(
            api_key="hone_test_key",
            project_id="test-project",
            batch_size=100,  # Large batch to prevent auto-flush
            flush_interval=0.5,  # Short interval for faster test
        )

        sent_calls: List[TrackedCall] = []

        def mock_send(batch):
            sent_calls.extend(batch)
            return True

        client._send_to_api = mock_send

        @client.track()
        def func(x):
            return x

        # Make a few calls
        for i in range(3):
            func(i)

        # Give small delay for enqueue and processing
        time.sleep(0.2)

        # Force flush with longer timeout
        client.flush(timeout=10.0)

        # Additional wait for processing
        time.sleep(0.5)

        client.shutdown()

        # All calls should have been sent
        assert len(sent_calls) == 3, f"Expected 3 calls, got {len(sent_calls)}"

    def test_shutdown_graceful(self):
        """shutdown() waits for pending uploads."""
        sent_calls: List[TrackedCall] = []

        client = Hone(
            api_key="hone_test_key",
            project_id="test-project",
        )

        def mock_send(batch):
            sent_calls.extend(batch)
            return True

        client._send_to_api = mock_send

        @client.track()
        def func(x):
            return x

        # Make several calls
        for i in range(5):
            func(i)

        # Shutdown should wait for all to complete
        client.shutdown()

        # All calls should have been sent
        assert len(sent_calls) == 5

    def test_background_thread_retry_on_failure(self):
        """Background thread retries failed uploads."""
        client = Hone(
            api_key="hone_test_key",
            project_id="test-project",
            max_retries=3,
        )

        attempt_count = [0]

        def failing_then_success(batch):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise AIXConnectionError("Connection failed")
            return True

        client._send_to_api = failing_then_success

        @client.track()
        def func(x):
            return x

        func(1)

        # Give time for background thread to process with retries
        time.sleep(2.5)  # Allow time for retries with exponential backoff
        client.flush()

        client.shutdown()

        # Should have retried at least twice
        assert attempt_count[0] >= 2, f"Expected at least 2 attempts, got {attempt_count[0]}"


class TestHighThroughput:
    """Tests for high-throughput performance requirements."""

    def test_high_throughput(self):
        """Can handle 100 calls/sec without blocking."""
        client = Hone(
            api_key="hone_test_key",
            project_id="test-project",
        )

        with patch.object(client, "_send_to_api") as mock_send:
            mock_send.return_value = True

            @client.track()
            def fast_func(x):
                return x * 2

            # Measure time for 100 calls
            start = time.perf_counter()

            for i in range(100):
                fast_func(i)

            elapsed = time.perf_counter() - start

            # Should complete in under 1 second (100 calls/sec)
            # Adding overhead buffer for test environment
            assert elapsed < 1.0, f"100 calls took {elapsed:.2f}s, expected < 1s"

        client.shutdown()

    def test_decorator_overhead(self):
        """Decorator adds less than 5ms overhead per call."""
        client = Hone(
            api_key="hone_test_key",
            project_id="test-project",
        )

        with patch.object(client, "_send_to_api") as mock_send:
            mock_send.return_value = True

            def plain_func(x):
                return x * 2

            @client.track()
            def tracked_func(x):
                return x * 2

            # Warm up
            for _ in range(10):
                plain_func(1)
                tracked_func(1)

            # Measure plain function
            plain_times = []
            for _ in range(100):
                start = time.perf_counter()
                plain_func(1)
                plain_times.append(time.perf_counter() - start)

            # Measure tracked function
            tracked_times = []
            for _ in range(100):
                start = time.perf_counter()
                tracked_func(1)
                tracked_times.append(time.perf_counter() - start)

            avg_plain = sum(plain_times) / len(plain_times) * 1000  # ms
            avg_tracked = sum(tracked_times) / len(tracked_times) * 1000  # ms
            overhead = avg_tracked - avg_plain

            # Overhead should be < 5ms
            assert overhead < 5.0, f"Decorator overhead: {overhead:.2f}ms, expected < 5ms"

        client.shutdown()


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_tracking(self):
        """Multiple threads can track calls concurrently."""
        client = Hone(
            api_key="hone_test_key",
            project_id="test-project",
        )

        captured_calls: List[TrackedCall] = []
        lock = threading.Lock()

        original_enqueue = client._enqueue_call
        def mock_enqueue(call: TrackedCall):
            with lock:
                captured_calls.append(call)
            return original_enqueue(call)
        client._enqueue_call = mock_enqueue

        @client.track()
        def thread_func(thread_id: int):
            time.sleep(0.01)
            return f"result-{thread_id}"

        threads = []
        for i in range(10):
            t = threading.Thread(target=thread_func, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        client.flush()

        assert len(captured_calls) == 10

        client.shutdown()


class TestExceptions:
    """Tests for custom exception classes."""

    def test_hone_error_base(self):
        """AIXError is the base exception class."""
        error = AIXError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_hone_connection_error(self):
        """AIXConnectionError is for connection failures."""
        error = AIXConnectionError("Connection failed")
        assert isinstance(error, AIXError)

    def test_hone_validation_error(self):
        """AIXValidationError is for validation failures."""
        error = AIXValidationError("Invalid input")
        assert isinstance(error, AIXError)


class TestOpenAIIntegration:
    """Tests for capturing OpenAI response metadata."""

    def test_captures_openai_response_metadata(self):
        """Decorator captures tokens and model from OpenAI-style response."""
        client = Hone(api_key="hone_test_key", project_id="test-project")
        captured_calls: List[TrackedCall] = []

        original_enqueue = client._enqueue_call
        def mock_enqueue(call: TrackedCall):
            captured_calls.append(call)
            return original_enqueue(call)
        client._enqueue_call = mock_enqueue

        # Mock OpenAI response object
        class MockChoice:
            def __init__(self):
                self.message = MagicMock()
                self.message.content = "Test response"

        class MockUsage:
            prompt_tokens = 10
            completion_tokens = 20
            total_tokens = 30

        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
                self.model = "gpt-4"
                self.usage = MockUsage()

        @client.track()
        def openai_call():
            return MockResponse()

        result = openai_call()
        assert len(captured_calls) == 1
        # The decorator should extract metadata from OpenAI-style responses
        assert captured_calls[0].tokens_used == 30 or captured_calls[0].model == "gpt-4"

        client.shutdown()
