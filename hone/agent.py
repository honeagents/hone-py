"""
Agent and Entity utilities for the Hone SDK.

Exact replica of TypeScript agent.ts - handles entity tree building,
evaluation, and formatting.
"""

import re
from typing import Callable, Dict, List, Optional, Set, Any, Union

from .types import (
    GetAgentOptions,
    GetToolOptions,
    GetTextPromptOptions,
    EntityNode,
    EntityRequest,
    EntityRequestItem,
    SimpleParams,
    EntityType,
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
        agent_options = options  # type: GetAgentOptions
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


def evaluate_agent(node: EntityNode) -> str:
    """
    Evaluates an EntityNode by recursively inserting parameters and nested entities.

    Args:
        node: The root EntityNode to evaluate.

    Returns:
        The fully evaluated prompt string.

    Raises:
        ValueError: if any placeholders in the prompt don't have corresponding parameter values
    """
    return evaluate_entity(node)


def evaluate_entity(node: EntityNode) -> str:
    """
    Evaluates an EntityNode by recursively inserting parameters and nested entities.

    Args:
        node: The root EntityNode to evaluate.

    Returns:
        The fully evaluated prompt string.

    Raises:
        ValueError: if any placeholders in the prompt don't have corresponding parameter values
    """
    evaluated: Dict[str, str] = {}

    def evaluate(n: EntityNode) -> str:
        if n["id"] in evaluated:
            return evaluated[n["id"]]

        params: SimpleParams = dict(n["params"])

        # Evaluate all children first (depth-first)
        for child in n["children"]:
            params[child["id"]] = evaluate(child)

        # Validate that all placeholders have corresponding parameters
        _validate_entity_params(n["prompt"], params, n["id"])

        # Insert evaluated children into this prompt
        result = insert_params_into_prompt(n["prompt"], params)
        evaluated[n["id"]] = result
        return result

    return evaluate(node)


def _validate_entity_params(
    prompt: str,
    params: SimpleParams,
    node_id: str,
) -> None:
    """
    Validates that all placeholders in a prompt have corresponding parameter values.

    Args:
        prompt: The prompt template to validate
        params: The available parameters
        node_id: The node ID for error messaging

    Raises:
        ValueError: if any placeholders don't have corresponding parameters
    """
    # Extract all placeholders from the prompt
    placeholder_regex = re.compile(r"\{\{(\w+)\}\}")
    matches = placeholder_regex.findall(prompt)
    missing_params: List[str] = []

    for param_name in matches:
        if param_name not in params:
            missing_params.append(param_name)

    if missing_params:
        # Remove duplicates, preserve order
        unique_missing = list(dict.fromkeys(missing_params))
        plural = "s" if len(unique_missing) > 1 else ""
        raise ValueError(
            f'Missing parameter{plural} in entity "{node_id}": {", ".join(unique_missing)}'
        )


def insert_params_into_prompt(
    prompt: str,
    params: Optional[SimpleParams] = None,
) -> str:
    """
    Inserts parameters into a prompt template.

    Args:
        prompt: The prompt template containing placeholders in the form {{variableName}}.
        params: An object mapping variable names to their replacement values.

    Returns:
        The prompt with all placeholders replaced by their corresponding values.
    """
    if params is None:
        return prompt

    result = prompt
    for key, value in params.items():
        # Use re.sub with escaped key for safety, but key should only be word chars
        result = re.sub(r"\{\{" + re.escape(key) + r"\}\}", value, result)
    return result


def traverse_entity_node(
    node: EntityNode,
    callback: Callable[[EntityNode, Optional[str]], None],
    parent_id: Optional[str] = None,
) -> None:
    """
    Traverses an EntityNode tree and applies a callback to each node.

    Args:
        node: The root node to start traversal from
        callback: Function called for each node with (node, parent_id)
        parent_id: The ID of the parent node (None for root)
    """
    callback(node, parent_id)
    for child in node["children"]:
        traverse_entity_node(child, callback, node["id"])


def traverse_agent_node(
    node: EntityNode,
    callback: Callable[[EntityNode, Optional[str]], None],
    parent_id: Optional[str] = None,
) -> None:
    """
    Traverses an EntityNode tree and applies a callback to each node.
    Alias for traverse_entity_node for backwards compatibility.
    """
    traverse_entity_node(node, callback, parent_id)


def format_entity_request(node: EntityNode) -> EntityRequest:
    """
    Formats an EntityNode into an EntityRequest suitable for the /sync_entities API.

    Args:
        node: The root EntityNode to format

    Returns:
        The formatted EntityRequest
    """
    def format_node(n: EntityNode) -> EntityRequestItem:
        param_keys = list(n["params"].keys()) + [child["id"] for child in n["children"]]
        item: EntityRequestItem = {
            "id": n["id"],
            "type": n.get("type", "agent"),
            "name": n.get("name"),
            "majorVersion": n.get("major_version"),
            "prompt": n["prompt"],
            "paramKeys": param_keys,
            "childrenIds": [child["id"] for child in n["children"]],
            "childrenTypes": [child.get("type", "prompt") for child in n["children"]],
        }

        # Add hyperparameters for agents
        if n.get("type") == "agent":
            item["model"] = n.get("model")
            item["provider"] = n.get("provider")
            item["temperature"] = n.get("temperature")
            item["maxTokens"] = n.get("max_tokens")
            item["topP"] = n.get("top_p")
            item["frequencyPenalty"] = n.get("frequency_penalty")
            item["presencePenalty"] = n.get("presence_penalty")
            item["stopSequences"] = n.get("stop_sequences")
            item["tools"] = n.get("tools")

        return item

    entity_map: Dict[str, EntityRequestItem] = {}

    def add_to_map(current_node: EntityNode, parent_id: Optional[str]) -> None:
        entity_map[current_node["id"]] = format_node(current_node)

    traverse_entity_node(node, add_to_map)

    return {
        "entities": {
            "rootId": node["id"],
            "rootType": node.get("type", "agent"),
            "map": entity_map,
        }
    }


def format_agent_request(node: EntityNode) -> Dict[str, Any]:
    """
    Formats an EntityNode into an AgentRequest suitable for the /sync_agents API.
    DEPRECATED: Use format_entity_request instead.

    Args:
        node: The root EntityNode to format

    Returns:
        The formatted AgentRequest (old format)
    """
    def format_node(n: EntityNode) -> Dict[str, Any]:
        param_keys = list(n["params"].keys()) + [child["id"] for child in n["children"]]
        return {
            "id": n["id"],
            "name": n.get("name"),
            "majorVersion": n.get("major_version"),
            "prompt": n["prompt"],
            "paramKeys": param_keys,
            "childrenIds": [child["id"] for child in n["children"]],
            # Hyperparameters (using camelCase for API)
            "model": n.get("model"),
            "temperature": n.get("temperature"),
            "maxTokens": n.get("max_tokens"),
            "topP": n.get("top_p"),
            "frequencyPenalty": n.get("frequency_penalty"),
            "presencePenalty": n.get("presence_penalty"),
            "stopSequences": n.get("stop_sequences"),
        }

    agent_map: Dict[str, Any] = {}

    def add_to_map(current_node: EntityNode, parent_id: Optional[str]) -> None:
        agent_map[current_node["id"]] = format_node(current_node)

    traverse_agent_node(node, add_to_map)

    return {
        "agents": {
            "rootId": node["id"],
            "map": agent_map,
        }
    }


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


# ============================================================================
# Backwards Compatibility Aliases (deprecated)
# ============================================================================

# Deprecated validation function alias
def _validate_agent_params(
    prompt: str,
    params: SimpleParams,
    node_id: str,
) -> None:
    """DEPRECATED: Use _validate_entity_params instead."""
    _validate_entity_params(prompt, params, node_id)


# Deprecated: Use get_agent_node instead
get_prompt_node = get_agent_node

# Deprecated: Use evaluate_agent instead
evaluate_prompt = evaluate_agent

# Deprecated: Use traverse_agent_node instead
traverse_prompt_node = traverse_agent_node

# Deprecated: Use format_agent_request instead
format_prompt_request = format_agent_request

# Deprecated: Use update_agent_nodes instead
update_prompt_nodes = update_agent_nodes
