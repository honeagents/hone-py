"""
Agent and Entity utilities for the Hone SDK.

Handles entity tree building and formatting for the evaluate API.
"""

from typing import Callable, Dict, List, Optional, Set, Any, Union, cast

from .types import (
    GetAgentOptions,
    GetToolOptions,
    GetTextPromptOptions,
    EntityNode,
    SimpleParams,
    EntityType,
    EntityV2Request,
)

# Type alias for the combined entity node type
AgentNode = EntityNode
ToolNode = EntityNode
TextPromptNode = EntityNode


def get_agent_node(
    id: str,
    options: GetAgentOptions,
    ancestor_ids: Optional[Set[str]] = None,
) -> EntityNode:
    """
    Constructs an EntityNode (with type="agent") from the given id and GetAgentOptions.
    Traverses nested entities recursively.

    Args:
        id: the unique identifier for the agent node
        options: the GetAgentOptions containing agent details and parameters
        ancestor_ids: Set of ancestor IDs to detect circular references

    Returns:
        The constructed EntityNode with type="agent"

    Raises:
        ValueError: if a self-reference or circular reference is detected
    """
    return _get_entity_node(id, options, "agent", ancestor_ids)


def get_tool_node(
    id: str,
    options: GetToolOptions,
    ancestor_ids: Optional[Set[str]] = None,
) -> EntityNode:
    """
    Constructs an EntityNode (with type="tool") from the given id and GetToolOptions.
    Traverses nested entities recursively.

    Args:
        id: the unique identifier for the tool node
        options: the GetToolOptions containing tool details and parameters
        ancestor_ids: Set of ancestor IDs to detect circular references

    Returns:
        The constructed EntityNode with type="tool"

    Raises:
        ValueError: if a self-reference or circular reference is detected
    """
    return _get_entity_node(id, options, "tool", ancestor_ids)


def get_text_prompt_node(
    id: str,
    options: GetTextPromptOptions,
    ancestor_ids: Optional[Set[str]] = None,
) -> EntityNode:
    """
    Constructs an EntityNode (with type="prompt") from the given id and GetTextPromptOptions.
    Traverses nested entities recursively.

    Args:
        id: the unique identifier for the prompt node
        options: the GetTextPromptOptions containing prompt details and parameters
        ancestor_ids: Set of ancestor IDs to detect circular references

    Returns:
        The constructed EntityNode with type="prompt"

    Raises:
        ValueError: if a self-reference or circular reference is detected
    """
    return _get_entity_node(id, options, "prompt", ancestor_ids)


def _get_entity_node(
    id: str,
    options: Union[GetAgentOptions, GetToolOptions, GetTextPromptOptions],
    entity_type: EntityType,
    ancestor_ids: Optional[Set[str]] = None,
) -> EntityNode:
    """
    Internal function to construct an EntityNode from options.

    Args:
        id: the unique identifier for the entity node
        options: the options containing entity details and parameters
        entity_type: the type of entity ("agent", "tool", or "prompt")
        ancestor_ids: Set of ancestor IDs to detect circular references

    Returns:
        The constructed EntityNode

    Raises:
        ValueError: if a self-reference or circular reference is detected
    """
    if ancestor_ids is None:
        ancestor_ids = set()

    params = options.get("params", {})

    # Check for self-reference: if this entity's params contain a key matching its own id
    if params and id in params:
        raise ValueError(
            f'Self-referencing {entity_type} detected: {entity_type} "{id}" cannot reference itself as a parameter'
        )

    # Check for circular reference: if this id is already in the ancestor chain
    if id in ancestor_ids:
        path = " -> ".join(list(ancestor_ids) + [id])
        raise ValueError(f"Circular {entity_type} reference detected: {path}")

    children: List[EntityNode] = []
    new_ancestor_ids = ancestor_ids | {id}

    simple_params: SimpleParams = {}
    for param_id, value in (params or {}).items():
        if isinstance(value, str):
            simple_params[param_id] = value
        else:
            # It's a nested entity options - could be agent, tool, or prompt
            # Determine the child type based on the options structure
            child_type: EntityType = "prompt"  # default to prompt for nested
            if "model" in value or "provider" in value:
                child_type = "agent"
            children.append(_get_entity_node(param_id, value, child_type, new_ancestor_ids))

    node: EntityNode = {
        "id": id,
        "type": entity_type,
        "major_version": options.get("major_version"),
        "name": options.get("name"),
        "params": simple_params,
        "prompt": options.get("default_prompt", ""),
        "children": children,
    }

    # Add hyperparameters for agents
    if entity_type == "agent":
        agent_options = cast(GetAgentOptions, options)
        node["model"] = agent_options.get("model")
        node["provider"] = agent_options.get("provider")
        node["temperature"] = agent_options.get("temperature")
        node["max_tokens"] = agent_options.get("max_tokens")
        node["top_p"] = agent_options.get("top_p")
        node["frequency_penalty"] = agent_options.get("frequency_penalty")
        node["presence_penalty"] = agent_options.get("presence_penalty")
        node["stop_sequences"] = agent_options.get("stop_sequences")
        node["tools"] = agent_options.get("tools")

    return node


def format_entity_v2_request(node: EntityNode) -> EntityV2Request:
    """
    Formats an EntityNode into an EntityV2Request suitable for the /api/evaluate API.
    Uses nested structure with param values (not just keys).

    Args:
        node: The root EntityNode to format

    Returns:
        The formatted EntityV2Request
    """

    def format_node(n: EntityNode) -> EntityV2Request:
        # Build params: string values + recursively formatted children
        params: Dict[str, Any] = {}

        # Add string params
        for key, value in n["params"].items():
            params[key] = value

        # Add children as nested entities
        for child in n["children"]:
            params[child["id"]] = format_node(child)

        request: EntityV2Request = {
            "id": n["id"],
            "type": n.get("type", "agent"),
            "prompt": n["prompt"],
            "majorVersion": n.get("major_version"),
            "name": n.get("name"),
        }

        if params:
            request["params"] = params

        # Add data for agents (hyperparameters)
        if n.get("type") == "agent":
            request["data"] = {
                "model": n.get("model"),
                "provider": n.get("provider"),
                "temperature": n.get("temperature"),
                "maxTokens": n.get("max_tokens"),
                "topP": n.get("top_p"),
                "frequencyPenalty": n.get("frequency_penalty"),
                "presencePenalty": n.get("presence_penalty"),
                "stopSequences": n.get("stop_sequences"),
                "tools": n.get("tools"),
            }

        return request

    return format_node(node)


def update_entity_nodes(
    root: EntityNode,
    callback: Callable[[EntityNode], EntityNode],
) -> EntityNode:
    """
    Updates all nodes in an EntityNode tree using a callback function.

    Args:
        root: The root node of the tree
        callback: Function that transforms each node

    Returns:
        The updated tree with all nodes transformed
    """
    def update_node(node: EntityNode) -> EntityNode:
        updated_children = [update_node(child) for child in node["children"]]
        updated_node: EntityNode = {**node, "children": updated_children}
        return callback(updated_node)

    return update_node(root)


def update_agent_nodes(
    root: EntityNode,
    callback: Callable[[EntityNode], EntityNode],
) -> EntityNode:
    """
    Updates all nodes in an EntityNode tree using a callback function.
    Alias for update_entity_nodes for backwards compatibility.
    """
    return update_entity_nodes(root, callback)


