"""
Agent utilities for the Hone SDK.

Exact replica of TypeScript agent.ts - handles agent tree building,
evaluation, and formatting.
"""

import re
from typing import Callable, Dict, List, Optional, Set, Any

from .types import (
    GetAgentOptions,
    AgentNode,
    AgentRequest,
    AgentRequestItem,
    AgentRequestPayload,
    SimpleParams,
)


def get_agent_node(
    id: str,
    options: GetAgentOptions,
    ancestor_ids: Optional[Set[str]] = None,
) -> AgentNode:
    """
    Constructs an AgentNode from the given id and GetAgentOptions.
    Traverses nested agents recursively.

    Args:
        id: the unique identifier for the agent node
        options: the GetAgentOptions containing agent details and parameters
        ancestor_ids: Set of ancestor IDs to detect circular references

    Returns:
        The constructed AgentNode

    Raises:
        ValueError: if a self-reference or circular reference is detected
    """
    if ancestor_ids is None:
        ancestor_ids = set()

    params = options.get("params", {})

    # Check for self-reference: if this agent's params contain a key matching its own id
    if params and id in params:
        raise ValueError(
            f'Self-referencing agent detected: agent "{id}" cannot reference itself as a parameter'
        )

    # Check for circular reference: if this id is already in the ancestor chain
    if id in ancestor_ids:
        path = " -> ".join(list(ancestor_ids) + [id])
        raise ValueError(f"Circular agent reference detected: {path}")

    children: List[AgentNode] = []
    new_ancestor_ids = ancestor_ids | {id}

    simple_params: SimpleParams = {}
    for param_id, value in (params or {}).items():
        if isinstance(value, str):
            simple_params[param_id] = value
        else:
            # It's a nested GetAgentOptions
            children.append(get_agent_node(param_id, value, new_ancestor_ids))

    return {
        "id": id,
        "major_version": options.get("major_version"),
        "name": options.get("name"),
        "params": simple_params,
        "prompt": options.get("default_prompt", ""),
        "children": children,
        # Hyperparameters
        "model": options.get("model"),
        "temperature": options.get("temperature"),
        "max_tokens": options.get("max_tokens"),
        "top_p": options.get("top_p"),
        "frequency_penalty": options.get("frequency_penalty"),
        "presence_penalty": options.get("presence_penalty"),
        "stop_sequences": options.get("stop_sequences"),
    }


def evaluate_agent(node: AgentNode) -> str:
    """
    Evaluates an AgentNode by recursively inserting parameters and nested agents.

    Args:
        node: The root AgentNode to evaluate.

    Returns:
        The fully evaluated prompt string.

    Raises:
        ValueError: if any placeholders in the prompt don't have corresponding parameter values
    """
    evaluated: Dict[str, str] = {}

    def evaluate(n: AgentNode) -> str:
        if n["id"] in evaluated:
            return evaluated[n["id"]]

        params: SimpleParams = dict(n["params"])

        # Evaluate all children first (depth-first)
        for child in n["children"]:
            params[child["id"]] = evaluate(child)

        # Validate that all placeholders have corresponding parameters
        _validate_agent_params(n["prompt"], params, n["id"])

        # Insert evaluated children into this prompt
        result = insert_params_into_prompt(n["prompt"], params)
        evaluated[n["id"]] = result
        return result

    return evaluate(node)


def _validate_agent_params(
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
            f'Missing parameter{plural} in agent "{node_id}": {", ".join(unique_missing)}'
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


def traverse_agent_node(
    node: AgentNode,
    callback: Callable[[AgentNode, Optional[str]], None],
    parent_id: Optional[str] = None,
) -> None:
    """
    Traverses an AgentNode tree and applies a callback to each node.

    Args:
        node: The root node to start traversal from
        callback: Function called for each node with (node, parent_id)
        parent_id: The ID of the parent node (None for root)
    """
    callback(node, parent_id)
    for child in node["children"]:
        traverse_agent_node(child, callback, node["id"])


def format_agent_request(node: AgentNode) -> AgentRequest:
    """
    Formats an AgentNode into an AgentRequest suitable for the /sync_agents API.

    Args:
        node: The root AgentNode to format

    Returns:
        The formatted AgentRequest
    """
    def format_node(n: AgentNode) -> AgentRequestItem:
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

    agent_map: Dict[str, AgentRequestItem] = {}

    def add_to_map(current_node: AgentNode, parent_id: Optional[str]) -> None:
        agent_map[current_node["id"]] = format_node(current_node)

    traverse_agent_node(node, add_to_map)

    return {
        "agents": {
            "rootId": node["id"],
            "map": agent_map,
        }
    }


def update_agent_nodes(
    root: AgentNode,
    callback: Callable[[AgentNode], AgentNode],
) -> AgentNode:
    """
    Updates all nodes in an AgentNode tree using a callback function.

    Args:
        root: The root node of the tree
        callback: Function that transforms each node

    Returns:
        The updated tree with all nodes transformed
    """
    def update_node(node: AgentNode) -> AgentNode:
        updated_children = [update_node(child) for child in node["children"]]
        updated_node: AgentNode = {**node, "children": updated_children}
        return callback(updated_node)

    return update_node(root)


# ============================================================================
# Backwards Compatibility Aliases (deprecated)
# ============================================================================

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
