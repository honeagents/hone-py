"""
Background thread for batch upload of tracked calls.

Portions of this code adapted from LangSmith SDK
Copyright (c) 2023 LangChain
Licensed under MIT License
https://github.com/langchain-ai/langsmith-sdk
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
import weakref
from queue import Empty, Queue
from typing import TYPE_CHECKING, Callable, List, Optional

from hone.models import TrackedCall
from hone.exceptions import HoneConnectionError

if TYPE_CHECKING:
    from hone.client import Hone

logger = logging.getLogger("hone.background_thread")


class BackgroundUploadThread:
    """
    Background thread that batches TrackedCall objects and uploads them
    to the Hone API.

    Inspired by LangSmith's background thread implementation, this class:
    - Collects calls in a queue
    - Batches them based on size or time interval
    - Uploads asynchronously to not block the main thread
    - Handles retries for failed uploads
    - Supports graceful shutdown
    """

    def __init__(
        self,
        client_ref: weakref.ref,
        batch_size: int = 100,
        flush_interval: float = 1.0,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        """
        Initialize the background upload thread.

        Args:
            client_ref: Weak reference to the Hone client.
            batch_size: Maximum number of calls to batch before uploading.
            flush_interval: Maximum seconds to wait before uploading a batch.
            max_retries: Maximum number of retry attempts for failed uploads.
            retry_delay: Initial delay between retries (exponential backoff).
        """
        self._client_ref = client_ref
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._queue: Queue[TrackedCall] = Queue()
        self._shutdown_event = threading.Event()
        self._flush_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._started = False

    @property
    def queue(self) -> Queue[TrackedCall]:
        """Access the internal queue for testing."""
        return self._queue

    def start(self) -> None:
        """Start the background thread."""
        with self._lock:
            if self._started:
                return

            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="Hone-BackgroundUpload",
            )
            self._thread.start()
            self._started = True

            # Register atexit handler for graceful shutdown
            atexit.register(self.shutdown)

    def enqueue(self, call: TrackedCall) -> None:
        """
        Add a tracked call to the upload queue.

        Args:
            call: TrackedCall to be uploaded.
        """
        if not self._started:
            self.start()

        self._queue.put(call)

        # If queue is full, trigger a flush
        if self._queue.qsize() >= self._batch_size:
            self._flush_event.set()

    def flush(self, timeout: Optional[float] = 5.0) -> None:
        """
        Force upload of all pending calls.

        Args:
            timeout: Maximum seconds to wait for flush to complete.
        """
        if not self._started or self._queue.empty():
            return

        # Signal flush
        self._flush_event.set()

        # Wait for queue to drain
        start = time.monotonic()
        while not self._queue.empty():
            if timeout and (time.monotonic() - start) > timeout:
                logger.warning("Flush timeout exceeded, some calls may not be uploaded")
                break
            time.sleep(0.01)

    def shutdown(self, timeout: Optional[float] = 5.0) -> None:
        """
        Gracefully shutdown the background thread.

        Waits for all pending uploads to complete.

        Args:
            timeout: Maximum seconds to wait for shutdown.
        """
        if not self._started:
            return

        # Signal shutdown
        self._shutdown_event.set()
        self._flush_event.set()

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._started = False

    def _run(self) -> None:
        """Main loop for the background thread."""
        last_flush_time = time.monotonic()

        while not self._shutdown_event.is_set():
            try:
                # Wait for flush event or timeout
                self._flush_event.wait(timeout=self._flush_interval)
                self._flush_event.clear()

                # Check if it's time to flush
                elapsed = time.monotonic() - last_flush_time
                should_flush = (
                    self._queue.qsize() >= self._batch_size
                    or elapsed >= self._flush_interval
                    or self._shutdown_event.is_set()
                )

                if should_flush and not self._queue.empty():
                    batch = self._drain_queue(self._batch_size)
                    if batch:
                        self._upload_batch_with_retry(batch)
                    last_flush_time = time.monotonic()

            except Exception as e:
                logger.error(
                    f"Error in background upload thread: {e}",
                    exc_info=True,
                )

        # Final drain on shutdown
        while not self._queue.empty():
            batch = self._drain_queue(self._batch_size)
            if batch:
                self._upload_batch_with_retry(batch)

        logger.debug("Background upload thread shutting down")

    def _drain_queue(self, limit: int) -> List[TrackedCall]:
        """
        Drain items from the queue up to the limit.

        Args:
            limit: Maximum number of items to drain.

        Returns:
            List of TrackedCall objects.
        """
        batch: List[TrackedCall] = []

        while len(batch) < limit:
            try:
                call = self._queue.get_nowait()
                batch.append(call)
                self._queue.task_done()
            except Empty:
                break

        return batch

    def _upload_batch_with_retry(self, batch: List[TrackedCall]) -> bool:
        """
        Upload a batch with retry logic.

        Args:
            batch: List of TrackedCall objects to upload.

        Returns:
            True if upload succeeded, False otherwise.
        """
        client = self._client_ref()
        if client is None:
            logger.warning("Client reference is gone, cannot upload batch")
            return False

        for attempt in range(self._max_retries):
            try:
                client._send_to_api(batch)
                return True
            except HoneConnectionError as e:
                delay = self._retry_delay * (2 ** attempt)
                logger.warning(
                    f"Upload attempt {attempt + 1}/{self._max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
            except Exception as e:
                logger.error(f"Unexpected error during upload: {e}", exc_info=True)
                return False

        logger.error(f"Failed to upload batch after {self._max_retries} attempts")
        return False


def create_background_thread(
    client: "Hone",
    batch_size: int = 100,
    flush_interval: float = 1.0,
    max_retries: int = 3,
) -> BackgroundUploadThread:
    """
    Create and start a background upload thread.

    Args:
        client: Hone client instance.
        batch_size: Maximum batch size.
        flush_interval: Flush interval in seconds.
        max_retries: Maximum retry attempts.

    Returns:
        Started BackgroundUploadThread instance.
    """
    thread = BackgroundUploadThread(
        client_ref=weakref.ref(client),
        batch_size=batch_size,
        flush_interval=flush_interval,
        max_retries=max_retries,
    )
    thread.start()
    return thread
