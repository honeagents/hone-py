"""
Main Hone client for tracking LLM calls.

Portions of this code adapted from LangSmith SDK
Copyright (c) 2023 LangChain
Licensed under MIT License
https://github.com/langchain-ai/langsmith-sdk
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time
import weakref
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, TypeVar
from urllib.parse import urljoin

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from hone.models import TrackedCall
from hone.exceptions import (
    HoneError,
    HoneConnectionError,
    HoneAuthenticationError,
    HoneRateLimitError,
)
from hone._background_thread import BackgroundUploadThread
from hone.decorators import create_track_decorator

logger = logging.getLogger("hone.client")

F = TypeVar("F", bound=Callable[..., Any])


class Hone:
    """
    Hone client for tracking LLM calls.

    This client wraps your LLM calls with the @hone.track() decorator
    and sends tracking data to the Hone API in the background.

    Example:
        ```python
        from hone import Hone

        hone = Hone(api_key="hone_xxx", project_id="my-project")

        @hone.track()
        def my_llm_call(message: str):
            return openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": message}]
            )

        result = my_llm_call("Hello!")
        ```

    Attributes:
        api_key: API key for authentication with Hone API.
        project_id: Project identifier for organizing tracked calls.
        api_url: Base URL for the Hone API.
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        api_url: str = "http://localhost:8000",
        batch_size: int = 100,
        flush_interval: float = 1.0,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize the Hone client.

        Args:
            api_key: API key for authentication.
            project_id: Project identifier.
            api_url: Base URL for the Hone API. Defaults to localhost.
            batch_size: Maximum number of calls to batch before upload.
            flush_interval: Maximum seconds between uploads.
            max_retries: Maximum retry attempts for failed uploads.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.project_id = project_id
        self.api_url = api_url.rstrip("/")
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._timeout = timeout

        # Internal state
        self._queue: Queue[TrackedCall] = Queue()
        self._lock = threading.Lock()
        self._shutdown = False

        # Setup HTTP client
        self._session = self._create_session()

        # Setup background thread
        self._background_thread = BackgroundUploadThread(
            client_ref=weakref.ref(self),
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_retries=max_retries,
        )
        self._background_thread.start()

        # Register cleanup on exit
        atexit.register(self.shutdown)

        logger.debug(
            f"Hone client initialized: project_id={project_id}, api_url={api_url}"
        )

    def _create_session(self) -> Any:
        """Create HTTP session for API calls."""
        if HAS_HTTPX:
            return httpx.Client(timeout=self._timeout)
        elif HAS_REQUESTS:
            session = requests.Session()
            session.headers.update({
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            })
            return session
        else:
            logger.warning(
                "Neither httpx nor requests is installed. "
                "API calls will fail. Install with: pip install httpx"
            )
            return None

    def track(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable[[F], F]:
        """
        Decorator to track LLM function calls.

        Args:
            name: Optional custom name for the tracked call.
                  Defaults to the function name.
            metadata: Optional additional metadata to include.

        Returns:
            Decorator function that wraps the target function.

        Example:
            ```python
            @hone.track()
            def my_function(message: str):
                return llm.chat(message)

            @hone.track(name="custom-name", metadata={"version": "1.0"})
            def another_function(query: str):
                return llm.complete(query)
            ```
        """
        return create_track_decorator(
            client=self,
            name=name,
            metadata=metadata,
        )

    def _enqueue_call(self, call: TrackedCall) -> None:
        """
        Enqueue a tracked call for upload.

        This method is called by the track decorator.

        Args:
            call: TrackedCall to enqueue.
        """
        if self._shutdown:
            logger.warning("Client is shut down, call will not be tracked")
            return

        self._background_thread.enqueue(call)

    def _send_to_api(self, batch: List[TrackedCall]) -> bool:
        """
        Send a batch of tracked calls to the API.

        Args:
            batch: List of TrackedCall objects to send.

        Returns:
            True if successful, raises exception otherwise.

        Raises:
            HoneConnectionError: If connection fails.
            HoneAuthenticationError: If authentication fails.
            HoneRateLimitError: If rate limited.
        """
        if not batch:
            return True

        if self._session is None:
            raise HoneConnectionError("No HTTP client available")

        url = urljoin(self.api_url, "/v1/track")
        payload = {
            "project_id": self.project_id,
            "calls": [call.to_dict() for call in batch],
        }

        try:
            if HAS_HTTPX:
                response = self._session.post(
                    url,
                    json=payload,
                    headers={
                        "X-API-Key": self.api_key,
                        "Content-Type": "application/json",
                    },
                )
                status_code = response.status_code
            elif HAS_REQUESTS:
                response = self._session.post(url, json=payload)
                status_code = response.status_code
            else:
                raise HoneConnectionError("No HTTP client available")

            if status_code == 401:
                raise HoneAuthenticationError("Invalid API key")
            elif status_code == 429:
                raise HoneRateLimitError("Rate limit exceeded")
            elif status_code >= 500:
                raise HoneConnectionError(f"Server error: {status_code}")
            elif status_code >= 400:
                raise HoneError(f"API error: {status_code}")

            logger.debug(f"Successfully uploaded batch of {len(batch)} calls")
            return True

        except (httpx.RequestError if HAS_HTTPX else Exception) as e:
            if HAS_HTTPX and isinstance(e, httpx.RequestError):
                raise HoneConnectionError(f"Connection failed: {e}") from e
            raise
        except (requests.RequestException if HAS_REQUESTS else Exception) as e:
            if HAS_REQUESTS and isinstance(e, requests.RequestException):
                raise HoneConnectionError(f"Connection failed: {e}") from e
            raise

    def _upload_batch(self, batch: List[TrackedCall]) -> None:
        """
        Upload a batch of tracked calls.

        Used internally by the background thread.

        Args:
            batch: List of TrackedCall objects.
        """
        self._send_to_api(batch)

    def flush(self, timeout: Optional[float] = 5.0) -> None:
        """
        Force upload of all pending tracked calls.

        This method blocks until all pending calls have been uploaded
        or the timeout is reached.

        Args:
            timeout: Maximum seconds to wait. None for no timeout.
        """
        self._background_thread.flush(timeout=timeout)

    def shutdown(self, timeout: Optional[float] = 5.0) -> None:
        """
        Gracefully shut down the client.

        This method:
        1. Stops accepting new calls
        2. Uploads all pending calls
        3. Closes network connections

        Args:
            timeout: Maximum seconds to wait for pending uploads.
        """
        if self._shutdown:
            return

        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True

        logger.debug("Shutting down Hone client")

        # Shutdown background thread
        self._background_thread.shutdown(timeout=timeout)

        # Close HTTP session
        if self._session is not None:
            try:
                if HAS_HTTPX:
                    self._session.close()
                elif HAS_REQUESTS:
                    self._session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")

        logger.debug("Hone client shut down complete")

    def __enter__(self) -> "Hone":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            if not self._shutdown:
                self.shutdown(timeout=1.0)
        except Exception:
            pass
