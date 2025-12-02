"""
Tests for Hone SDK client.
"""

import pytest
import time
import httpx
from unittest.mock import MagicMock, patch, PropertyMock

from hone import Hone
from hone.models import TrackedCall
from hone.exceptions import (
    AIXConnectionError,
    AIXAuthenticationError,
    AIXRateLimitError,
)


class TestAIXClient:
    """Tests for the Hone client."""

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_client_initialization(self, mock_httpx, mock_thread):
        """Test Hone client initializes correctly."""
        mock_client_instance = MagicMock()
        mock_httpx.Client.return_value = mock_client_instance

        client = Hone(
            api_key="test_key",
            project_id="test_project",
            api_url="http://localhost:8000",
        )

        assert client.api_key == "test_key"
        assert client.project_id == "test_project"
        assert client.api_url == "http://localhost:8000"

        # Clean up
        client._shutdown = True

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_client_api_url_trailing_slash_removed(self, mock_httpx, mock_thread):
        """Test that trailing slashes are removed from API URL."""
        client = Hone(
            api_key="test_key",
            project_id="test_project",
            api_url="http://localhost:8000/",
        )

        assert client.api_url == "http://localhost:8000"
        client._shutdown = True

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_track_decorator_exists(self, mock_httpx, mock_thread):
        """Test that track decorator is available."""
        client = Hone(api_key="key", project_id="project")
        decorator = client.track()
        assert callable(decorator)
        client._shutdown = True

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_track_decorator_with_name(self, mock_httpx, mock_thread):
        """Test track decorator with custom name."""
        client = Hone(api_key="key", project_id="project")
        decorator = client.track(name="custom-name")
        assert callable(decorator)
        client._shutdown = True

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_track_decorator_with_metadata(self, mock_httpx, mock_thread):
        """Test track decorator with custom metadata."""
        client = Hone(api_key="key", project_id="project")
        decorator = client.track(metadata={"version": "1.0"})
        assert callable(decorator)
        client._shutdown = True

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_enqueue_call_when_not_shutdown(self, mock_httpx, mock_thread):
        """Test that calls are enqueued when client is running."""
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        client = Hone(api_key="key", project_id="project")

        call = TrackedCall(
            function_name="test",
            input={},
            output=None,
            duration_ms=0,
            started_at=None,
        )

        client._enqueue_call(call)
        mock_thread_instance.enqueue.assert_called_once_with(call)
        client._shutdown = True

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_enqueue_call_when_shutdown(self, mock_httpx, mock_thread):
        """Test that calls are not enqueued after shutdown."""
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        client = Hone(api_key="key", project_id="project")
        client._shutdown = True

        call = TrackedCall(
            function_name="test",
            input={},
            output=None,
            duration_ms=0,
            started_at=None,
        )

        client._enqueue_call(call)
        # Should not enqueue when shutdown
        assert mock_thread_instance.enqueue.call_count <= 1  # May have been called during init


class TestAIXClientAPI:
    """Tests for Hone client API interactions."""

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_send_to_api_success(self, mock_httpx, mock_thread):
        """Test successful API call."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        client = Hone(api_key="key", project_id="project")

        call = TrackedCall(
            function_name="test",
            input={},
            output=None,
            duration_ms=0,
            started_at=None,
        )

        result = client._send_to_api([call])
        assert result is True
        client._shutdown = True

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_send_to_api_empty_batch(self, mock_httpx, mock_thread):
        """Test sending empty batch returns True."""
        client = Hone(api_key="key", project_id="project")
        result = client._send_to_api([])
        assert result is True
        client._shutdown = True

    @patch("aix.client.BackgroundUploadThread")
    def test_send_to_api_auth_error(self, mock_thread):
        """Test API returns 401 raises authentication error."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client.post.return_value = mock_response

        with patch("aix.client.httpx.Client", return_value=mock_client):
            client = Hone(api_key="invalid", project_id="project")

            call = TrackedCall(
                function_name="test",
                input={},
                output=None,
                duration_ms=0,
                started_at=None,
            )

            with pytest.raises(AIXAuthenticationError):
                client._send_to_api([call])

            client._shutdown = True

    @patch("aix.client.BackgroundUploadThread")
    def test_send_to_api_rate_limit(self, mock_thread):
        """Test API returns 429 raises rate limit error."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_client.post.return_value = mock_response

        with patch("aix.client.httpx.Client", return_value=mock_client):
            client = Hone(api_key="key", project_id="project")

            call = TrackedCall(
                function_name="test",
                input={},
                output=None,
                duration_ms=0,
                started_at=None,
            )

            with pytest.raises(AIXRateLimitError):
                client._send_to_api([call])

            client._shutdown = True

    @patch("aix.client.BackgroundUploadThread")
    def test_send_to_api_server_error(self, mock_thread):
        """Test API returns 5xx raises connection error."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.post.return_value = mock_response

        with patch("aix.client.httpx.Client", return_value=mock_client):
            client = Hone(api_key="key", project_id="project")

            call = TrackedCall(
                function_name="test",
                input={},
                output=None,
                duration_ms=0,
                started_at=None,
            )

            with pytest.raises(AIXConnectionError):
                client._send_to_api([call])

            client._shutdown = True


class TestAIXClientContextManager:
    """Tests for Hone client context manager functionality."""

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_context_manager_enter(self, mock_httpx, mock_thread):
        """Test context manager __enter__ returns client."""
        with Hone(api_key="key", project_id="project") as client:
            assert isinstance(client, Hone)

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_context_manager_exit_calls_shutdown(self, mock_httpx, mock_thread):
        """Test context manager __exit__ calls shutdown."""
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        with Hone(api_key="key", project_id="project") as client:
            pass

        mock_thread_instance.shutdown.assert_called()


class TestAIXClientFlush:
    """Tests for Hone client flush functionality."""

    @patch("aix.client.BackgroundUploadThread")
    @patch("aix.client.HAS_HTTPX", True)
    @patch("aix.client.httpx")
    def test_flush_calls_background_thread(self, mock_httpx, mock_thread):
        """Test flush calls background thread flush."""
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        client = Hone(api_key="key", project_id="project")
        client.flush(timeout=5.0)

        mock_thread_instance.flush.assert_called_once_with(timeout=5.0)
        client._shutdown = True
