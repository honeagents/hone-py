"""Pytest configuration for Hone SDK tests."""

import pytest


@pytest.fixture
def mock_api_key() -> str:
    """Provide a mock API key for tests."""
    return "test-api-key"
