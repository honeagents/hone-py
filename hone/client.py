"""
Hone SDK Client.

Exact replica of TypeScript client.ts - provides the main Hone class
for interacting with the Hone API.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypeVar

import httpx

from .types import (
    GetAgentOptions,
    HoneConfig,
    Message,
    AgentRequest,
    AgentResponse,
    TrackConversationOptions,
    TrackRequest,
)
from .agent import (
    evaluate_agent,
    format_agent_request,
    get_agent_node,
    update_agent_nodes,
)

DEFAULT_BASE_URL = "https://honeagents.ai/api"
DEFAULT_TIMEOUT = 10000  # milliseconds
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9."
    "CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0"
)

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class Hone:
    """
    The main Hone client for interacting with the Hone API.

    Implements the HoneClient protocol with agent() and track() methods.
    """

    def __init__(self, config: HoneConfig) -> None:
        """
        Initialize the Hone client.

        Args:
            config: Configuration including api_key, optional base_url, and timeout.
        """
        self._api_key = config["api_key"]
        # Allow override from env var for local dev, then config, then default
        self._base_url = (
            os.environ.get("HONE_API_URL")
            or config.get("base_url")
            or DEFAULT_BASE_URL
        )
        self._timeout = config.get("timeout", DEFAULT_TIMEOUT)

    async def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Make an HTTP request to the Hone API.

        Args:
            endpoint: The API endpoint path
            method: HTTP method (GET, POST, etc.)
            body: Optional request body

        Returns:
            The parsed JSON response

        Raises:
            Exception: If the request fails or times out
        """
        url = f"{self._base_url}{endpoint}"
        print(f"Hone API Request: {method} {url}")

        headers = {
            "apikey": SUPABASE_ANON_KEY,  # Supabase RPC requires lowercase "apikey"
            "x-api-key": self._api_key,  # Your RPC function reads this for project auth
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "hone-sdk-python/0.1.0",
        }

        timeout_seconds = self._timeout / 1000.0

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body,
                )

                if not response.is_success:
                    try:
                        error_data = response.json()
                        message = error_data.get("message", response.reason_phrase)
                    except Exception:
                        message = response.reason_phrase
                    raise Exception(
                        f"Hone API error ({response.status_code}): {message}"
                    )

                return response.json()

        except httpx.TimeoutException:
            raise Exception(
                f"Hone API request timed out after {self._timeout}ms"
            )

    async def agent(self, id: str, options: GetAgentOptions) -> str:
        """
        Fetches and evaluates an agent by its ID with the given options.

        Args:
            id: The unique identifier for the agent.
            options: Options for fetching and evaluating the agent.

        Returns:
            The evaluated prompt string.
        """
        node = get_agent_node(id, options)

        try:
            formatted_request = format_agent_request(node)
            new_agent_map: AgentResponse = await self._make_request(
                "/sync_agents",
                "POST",
                formatted_request,
            )

            def update_with_remote(agent_node: Dict[str, Any]) -> Dict[str, Any]:
                remote_agent = new_agent_map.get(agent_node["id"], {})
                return {
                    **agent_node,
                    "prompt": remote_agent.get("prompt", agent_node["prompt"]),
                }

            updated_agent_node = update_agent_nodes(node, update_with_remote)
            # Params are inserted client-side for flexibility and security
            return evaluate_agent(updated_agent_node)

        except Exception as error:
            print(f"Error fetching agent, using fallback: {error}")
            return evaluate_agent(node)

    async def prompt(self, id: str, options: GetAgentOptions) -> str:
        """
        Deprecated: Use agent() instead.

        Fetches and evaluates an agent by its ID with the given options.

        Args:
            id: The unique identifier for the agent.
            options: Options for fetching and evaluating the agent.

        Returns:
            The evaluated prompt string.
        """
        return await self.agent(id, options)

    async def track(
        self,
        id: str,
        messages: List[Message],
        options: TrackConversationOptions,
    ) -> None:
        """
        Adds messages to track a conversation under the given ID.

        Args:
            id: The unique identifier for the conversation to track.
            messages: An array of Message objects representing the conversation.
            options: TrackConversationOptions such as sessionId.
        """
        request: TrackRequest = {
            "id": id,
            "messages": messages,
            "sessionId": options["session_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await self._make_request("/insert_runs", "POST", request)


def create_hone_client(config: HoneConfig) -> Hone:
    """
    Factory function for easier initialization.

    Args:
        config: Configuration for the Hone client.

    Returns:
        A new Hone client instance.
    """
    return Hone(config)
