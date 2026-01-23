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
    EntityRequest,
    EntityResponse,
    ToolResult,
    TextPromptResult,
    TrackConversationOptions,
    TrackRequest,
)
from .agent import (
    evaluate_agent,
    evaluate_entity,
    format_entity_request,
    get_agent_node,
    get_tool_node,
    get_text_prompt_node,
    update_agent_nodes,
    update_entity_nodes,
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

        try:
            formatted_request = format_entity_request(node)

            # Include extra data in the request
            extra_data = options.get("extra")
            if extra_data:
                root_entity = formatted_request["entities"]["map"].get(id)
                if root_entity:
                    root_entity["extra"] = extra_data

            response = await self._make_request(
                "/sync_entities",
                "POST",
                formatted_request,
            )

            entity_map: EntityResponse = response.get("entities", {})

            def update_with_remote(agent_node: Dict[str, Any]) -> Dict[str, Any]:
                response_item = entity_map.get(agent_node["id"], {})
                return {
                    **agent_node,
                    "prompt": response_item.get("prompt", agent_node["prompt"]),
                    # Update hyperparameters from API response (if present)
                    "model": response_item.get("model") or agent_node.get("model"),
                    "temperature": response_item.get("temperature") if response_item.get("temperature") is not None else agent_node.get("temperature"),
                    "max_tokens": response_item.get("maxTokens") if response_item.get("maxTokens") is not None else agent_node.get("max_tokens"),
                    "top_p": response_item.get("topP") if response_item.get("topP") is not None else agent_node.get("top_p"),
                    "frequency_penalty": response_item.get("frequencyPenalty") if response_item.get("frequencyPenalty") is not None else agent_node.get("frequency_penalty"),
                    "presence_penalty": response_item.get("presencePenalty") if response_item.get("presencePenalty") is not None else agent_node.get("presence_penalty"),
                    "stop_sequences": response_item.get("stopSequences") if response_item.get("stopSequences") else agent_node.get("stop_sequences"),
                }

            updated_agent_node = update_agent_nodes(node, update_with_remote)

            # Get the root agent's hyperparameters and extra from the response
            root_response = entity_map.get(id, {})
            extra_from_response = root_response.get("extra", extra_data or {})

            # Build the result
            result: AgentResult = {
                "system_prompt": evaluate_agent(updated_agent_node),
                "model": root_response.get("model") or options.get("model", ""),
                "provider": root_response.get("provider") or options.get("provider", ""),
                "temperature": root_response.get("temperature") if root_response.get("temperature") is not None else options.get("temperature"),
                "max_tokens": root_response.get("maxTokens") if root_response.get("maxTokens") is not None else options.get("max_tokens"),
                "top_p": root_response.get("topP") if root_response.get("topP") is not None else options.get("top_p"),
                "frequency_penalty": root_response.get("frequencyPenalty") if root_response.get("frequencyPenalty") is not None else options.get("frequency_penalty"),
                "presence_penalty": root_response.get("presencePenalty") if root_response.get("presencePenalty") is not None else options.get("presence_penalty"),
                "stop_sequences": root_response.get("stopSequences") or options.get("stop_sequences", []),
                "tools": root_response.get("tools") or options.get("tools", []),
            }

            # Merge extra data
            if extra_from_response:
                result.update(extra_from_response)

            return result

        except Exception as error:
            print(f"Error fetching agent, using fallback: {error}")
            # Fallback: use local defaults
            extra_data = options.get("extra", {})
            result: AgentResult = {
                "system_prompt": evaluate_agent(node),
                "model": options.get("model", ""),
                "provider": options.get("provider", ""),
                "temperature": options.get("temperature"),
                "max_tokens": options.get("max_tokens"),
                "top_p": options.get("top_p"),
                "frequency_penalty": options.get("frequency_penalty"),
                "presence_penalty": options.get("presence_penalty"),
                "stop_sequences": options.get("stop_sequences", []),
                "tools": options.get("tools", []),
            }
            if extra_data:
                result.update(extra_data)
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

        try:
            formatted_request = format_entity_request(node)
            response = await self._make_request(
                "/sync_entities",
                "POST",
                formatted_request,
            )

            entity_map: EntityResponse = response.get("entities", {})

            def update_with_remote(entity_node: Dict[str, Any]) -> Dict[str, Any]:
                response_item = entity_map.get(entity_node["id"], {})
                return {
                    **entity_node,
                    "prompt": response_item.get("prompt", entity_node["prompt"]),
                }

            updated_tool_node = update_entity_nodes(node, update_with_remote)

            return {
                "prompt": evaluate_entity(updated_tool_node),
            }

        except Exception as error:
            print(f"Error fetching tool, using fallback: {error}")
            # Fallback: use local defaults
            return {
                "prompt": evaluate_entity(node),
            }

    async def prompt(self, id: str, options: GetTextPromptOptions) -> TextPromptResult:
        """
        Fetches and evaluates a text prompt by its ID with the given options.

        Args:
            id: The unique identifier for the prompt.
            options: Options for fetching and evaluating the prompt.

        Returns:
            A TextPromptResult containing the evaluated text.
        """
        node = get_text_prompt_node(id, options)

        try:
            formatted_request = format_entity_request(node)
            response = await self._make_request(
                "/sync_entities",
                "POST",
                formatted_request,
            )

            entity_map: EntityResponse = response.get("entities", {})

            def update_with_remote(entity_node: Dict[str, Any]) -> Dict[str, Any]:
                response_item = entity_map.get(entity_node["id"], {})
                return {
                    **entity_node,
                    "prompt": response_item.get("prompt", entity_node["prompt"]),
                }

            updated_prompt_node = update_entity_nodes(node, update_with_remote)

            return {
                "text": evaluate_entity(updated_prompt_node),
            }

        except Exception as error:
            print(f"Error fetching prompt, using fallback: {error}")
            # Fallback: use local defaults
            return {
                "text": evaluate_entity(node),
            }

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
