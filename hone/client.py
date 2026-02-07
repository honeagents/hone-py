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
    AgentResult,
    GetAgentOptions,
    GetToolOptions,
    GetTextPromptOptions,
    HoneConfig,
    Message,
    EntityV2Response,
    ToolResult,
    TrackConversationOptions,
    TrackRequest,
)
from .agent import (
    format_entity_v2_request,
    get_agent_node,
    get_tool_node,
    get_text_prompt_node,
)

DEFAULT_BASE_URL = "https://honeagents.ai/api"
DEFAULT_TIMEOUT = 10000  # milliseconds

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class Hone:
    """
    The main Hone client for interacting with the Hone API.

    Implements the HoneClient protocol with agent(), tool(), prompt(), and track() methods.
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
            "x-api-key": self._api_key,
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
                        message = error_data.get("error", error_data.get("message", response.reason_phrase))
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

    async def agent(self, id: str, options: GetAgentOptions) -> AgentResult:
        """
        Fetches and evaluates an agent by its ID with the given options.

        Args:
            id: The unique identifier for the agent.
            options: Options for fetching and evaluating the agent. Model and provider are required.

        Returns:
            An AgentResult containing the evaluated system prompt and hyperparameters.
        """
        node = get_agent_node(id, options)

        # Format request using nested structure
        request = format_entity_v2_request(node)

        # Include extra data in the request
        extra_data = options.get("extra")
        if extra_data and request.get("data"):
            request["data"].update(extra_data)

        # Call evaluate endpoint - server handles evaluation
        response: EntityV2Response = await self._make_request(
            "/evaluate",
            "POST",
            request,
        )

        # Extract data from response
        data = response.get("data", {})

        # Build the result - Response includes evaluated prompt
        result: AgentResult = {
            "system_prompt": response["evaluatedPrompt"],
            "model": data.get("model") or options.get("model", ""),
            "provider": data.get("provider") or options.get("provider", ""),
            "temperature": data.get("temperature") if data.get("temperature") is not None else options.get("temperature"),
            "max_tokens": data.get("maxTokens") if data.get("maxTokens") is not None else options.get("max_tokens"),
            "top_p": data.get("topP") if data.get("topP") is not None else options.get("top_p"),
            "frequency_penalty": data.get("frequencyPenalty") if data.get("frequencyPenalty") is not None else options.get("frequency_penalty"),
            "presence_penalty": data.get("presencePenalty") if data.get("presencePenalty") is not None else options.get("presence_penalty"),
            "stop_sequences": data.get("stopSequences") or options.get("stop_sequences", []),
            "tools": data.get("tools") or options.get("tools", []),
        }

        # Merge any extra data from response
        known_keys = {"model", "provider", "temperature", "maxTokens", "topP",
                     "frequencyPenalty", "presencePenalty", "stopSequences", "tools"}
        extra_from_response = {k: v for k, v in data.items() if k not in known_keys}
        if extra_from_response:
            result.update(extra_from_response)

        return result

    async def tool(self, id: str, options: GetToolOptions) -> ToolResult:
        """
        Fetches and evaluates a tool by its ID with the given options.

        Args:
            id: The unique identifier for the tool.
            options: Options for fetching and evaluating the tool.

        Returns:
            A ToolResult containing the evaluated prompt.
        """
        node = get_tool_node(id, options)

        # Format request using nested structure
        request = format_entity_v2_request(node)

        # Call evaluate endpoint - server handles evaluation
        response: EntityV2Response = await self._make_request(
            "/evaluate",
            "POST",
            request,
        )

        # Response includes evaluated prompt - no client-side evaluation needed
        return {
            "prompt": response["evaluatedPrompt"],
        }

    async def prompt(self, id: str, options: GetTextPromptOptions) -> str:
        """
        Fetches and evaluates a text prompt by its ID with the given options.

        Args:
            id: The unique identifier for the prompt.
            options: Options for fetching and evaluating the prompt.

        Returns:
            The evaluated text string.
        """
        node = get_text_prompt_node(id, options)

        # Format request using nested structure
        request = format_entity_v2_request(node)

        # Call evaluate endpoint - server handles evaluation
        response: EntityV2Response = await self._make_request(
            "/evaluate",
            "POST",
            request,
        )

        # Return the evaluated text directly
        return response["evaluatedPrompt"]

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
