"""
Pytest configuration and fixtures for Hone SDK tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


@pytest.fixture
def mock_httpx_client():
    """Fixture that provides a mock httpx client."""
    with patch("aix.client.HAS_HTTPX", True):
        with patch("aix.client.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok", "tracked": 1}
            mock_client.post.return_value = mock_response
            mock_httpx.Client.return_value = mock_client
            yield mock_client


@pytest.fixture
def sample_tracked_call():
    """Fixture that provides sample tracked call data."""
    return {
        "function_name": "test_function",
        "input": {"message": "Hello"},
        "output": "Hi there!",
        "duration_ms": 100,
        "started_at": datetime.now(timezone.utc),
    }


@pytest.fixture
def mock_openai_response():
    """Fixture that provides a mock OpenAI-style response."""
    # Use spec to limit attributes and prevent MagicMock auto-generation
    response = MagicMock(spec=["model", "usage", "choices"])
    response.model = "gpt-4"
    response.usage = MagicMock(spec=["total_tokens", "prompt_tokens", "completion_tokens"])
    response.usage.total_tokens = 150
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 100
    response.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]
    return response


@pytest.fixture
def mock_anthropic_response():
    """Fixture that provides a mock Anthropic-style response."""
    response = MagicMock()
    response.model = "claude-3-sonnet"
    response.usage = MagicMock()
    response.usage.input_tokens = 50
    response.usage.output_tokens = 100
    response.content = [MagicMock(text="Test response")]
    return response
