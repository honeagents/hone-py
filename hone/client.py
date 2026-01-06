"""
Hone Client Module.

Provides the main Client class for interacting with the Hone API.
This is a thin wrapper around LangSmith's Client with Hone defaults.
"""

from typing import Optional, Any, Dict, List

from langsmith import Client as LangSmithClient
from langsmith import AsyncClient as LangSmithAsyncClient

from hone._config import HONE_DEFAULT_ENDPOINT
from hone._patch import _get_hone_env_var


class Client(LangSmithClient):
    """
    Hone API Client.

    A client for tracking LLM calls and managing evaluations with Hone.
    This extends LangSmith's Client with Hone-specific defaults.

    Example:
        ```python
        from hone import Client

        # Using environment variables
        # export HONE_API_KEY=hone_xxx
        client = Client()

        # Or explicit configuration
        client = Client(
            api_key="hone_xxx",
            api_url="https://api.honeagents.ai"
        )
        ```

    Attributes:
        api_url: The Hone API endpoint URL
        api_key: API key for authentication
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the Hone Client.

        Args:
            api_url: Hone API URL. Defaults to HONE_ENDPOINT env var
                    or https://api.honeagents.ai
            api_key: API key. Defaults to HONE_API_KEY env var
            **kwargs: Additional arguments passed to LangSmith Client
        """
        # Get API URL with Hone defaults
        if api_url is None:
            api_url = _get_hone_env_var("ENDPOINT", HONE_DEFAULT_ENDPOINT)

        # Get API key with Hone defaults
        if api_key is None:
            api_key = _get_hone_env_var("API_KEY")

        super().__init__(api_url=api_url, api_key=api_key, **kwargs)


class AsyncClient(LangSmithAsyncClient):
    """
    Async Hone API Client.

    An async client for tracking LLM calls and managing evaluations with Hone.
    This extends LangSmith's AsyncClient with Hone-specific defaults.

    Example:
        ```python
        from hone import AsyncClient

        async def main():
            client = AsyncClient()
            # Use client for async operations
        ```

    Attributes:
        api_url: The Hone API endpoint URL
        api_key: API key for authentication
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the Async Hone Client.

        Args:
            api_url: Hone API URL. Defaults to HONE_ENDPOINT env var
                    or https://api.honeagents.ai
            api_key: API key. Defaults to HONE_API_KEY env var
            **kwargs: Additional arguments passed to LangSmith AsyncClient
        """
        # Get API URL with Hone defaults
        if api_url is None:
            api_url = _get_hone_env_var("ENDPOINT", HONE_DEFAULT_ENDPOINT)

        # Get API key with Hone defaults
        if api_key is None:
            api_key = _get_hone_env_var("API_KEY")

        super().__init__(api_url=api_url, api_key=api_key, **kwargs)
