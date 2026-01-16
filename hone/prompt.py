"""
Prompt utilities for the Hone SDK.

Exact replica of TypeScript prompt.ts - handles prompt tree building,
evaluation, and formatting.
"""

import re
from typing import Callable, Dict, List, Optional, Set, Any

from .types import (
    GetPromptOptions,
    PromptNode,
    PromptRequest,
    PromptRequestItem,
    PromptRequestPayload,
    SimpleParams,
)


def get_prompt_node(
    id: str,
    options: GetPromptOptions,
    ancestor_ids: Optional[Set[str]] = None,
) -> PromptNode:
    """
    Constructs a PromptNode from the given id and GetPromptOptions.
    Traverses nested prompts recursively.

    Args:
        id: the unique identifier for the prompt node
        options: the GetPromptOptions containing prompt details and parameters
        ancestor_ids: Set of ancestor IDs to detect circular references

    Returns:
        The constructed PromptNode

    Raises:
        ValueError: if a self-reference or circular reference is detected
    """
    if ancestor_ids is None:
        ancestor_ids = set()

    params = options.get("params", {})

    # Check for self-reference: if this prompt's params contain a key matching its own id
    if params and id in params:
        raise ValueError(
            f'Self-referencing prompt detected: prompt "{id}" cannot reference itself as a parameter'
        )

    # Check for circular reference: if this id is already in the ancestor chain
    if id in ancestor_ids:
        path = " -> ".join(list(ancestor_ids) + [id])
        raise ValueError(f"Circular prompt reference detected: {path}")

    children: List[PromptNode] = []
    new_ancestor_ids = ancestor_ids | {id}

    simple_params: SimpleParams = {}
    for param_id, value in (params or {}).items():
        if isinstance(value, str):
            simple_params[param_id] = value
        else:
            # It's a nested GetPromptOptions
            children.append(get_prompt_node(param_id, value, new_ancestor_ids))

    return {
        "id": id,
        "version": options.get("version"),
        "name": options.get("name"),
        "params": simple_params,
        "prompt": options.get("default_prompt", ""),
        "children": children,
    }


def evaluate_prompt(node: PromptNode) -> str:
    """
    Evaluates a PromptNode by recursively inserting parameters and nested prompts.

    Args:
        node: The root PromptNode to evaluate.

    Returns:
        The fully evaluated prompt string.

    Raises:
        ValueError: if any placeholders in the prompt don't have corresponding parameter values
    """
    evaluated: Dict[str, str] = {}

    def evaluate(n: PromptNode) -> str:
        if n["id"] in evaluated:
            return evaluated[n["id"]]

        params: SimpleParams = dict(n["params"])

        # Evaluate all children first (depth-first)
        for child in n["children"]:
            params[child["id"]] = evaluate(child)

        # Validate that all placeholders have corresponding parameters
        _validate_prompt_params(n["prompt"], params, n["id"])

        # Insert evaluated children into this prompt
        result = insert_params_into_prompt(n["prompt"], params)
        evaluated[n["id"]] = result
        return result

    return evaluate(node)


def _validate_prompt_params(
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
            f'Missing parameter{plural} in prompt "{node_id}": {", ".join(unique_missing)}'
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


def traverse_prompt_node(
    node: PromptNode,
    callback: Callable[[PromptNode, Optional[str]], None],
    parent_id: Optional[str] = None,
) -> None:
    """
    Traverses a PromptNode tree and applies a callback to each node.

    Args:
        node: The root node to start traversal from
        callback: Function called for each node with (node, parent_id)
        parent_id: The ID of the parent node (None for root)
    """
    callback(node, parent_id)
    for child in node["children"]:
        traverse_prompt_node(child, callback, node["id"])


def format_prompt_request(node: PromptNode) -> PromptRequest:
    """
    Formats a PromptNode into a PromptRequest suitable for the /prompts API.

    Args:
        node: The root PromptNode to format

    Returns:
        The formatted PromptRequest
    """
    def format_node(n: PromptNode) -> PromptRequestItem:
        param_keys = list(n["params"].keys()) + [child["id"] for child in n["children"]]
        return {
            "id": n["id"],
            "name": n.get("name"),
            "version": n.get("version"),
            "prompt": n["prompt"],
            "paramKeys": param_keys,
            "childrenIds": [child["id"] for child in n["children"]],
        }

    prompt_map: Dict[str, PromptRequestItem] = {}

    def add_to_map(current_node: PromptNode, parent_id: Optional[str]) -> None:
        prompt_map[current_node["id"]] = format_node(current_node)

    traverse_prompt_node(node, add_to_map)

    return {
        "prompts": {
            "rootId": node["id"],
            "map": prompt_map,
        }
    }


def update_prompt_nodes(
    root: PromptNode,
    callback: Callable[[PromptNode], PromptNode],
) -> PromptNode:
    """
    Updates all nodes in a PromptNode tree using a callback function.

    Args:
        root: The root node of the tree
        callback: Function that transforms each node

    Returns:
        The updated tree with all nodes transformed
    """
    def update_node(node: PromptNode) -> PromptNode:
        updated_children = [update_node(child) for child in node["children"]]
        updated_node: PromptNode = {**node, "children": updated_children}
        return callback(updated_node)

    return update_node(root)
