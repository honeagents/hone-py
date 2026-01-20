"""
Hone SDK - AI Experience Engineering Platform.

Public API exports matching TypeScript index.ts.
"""

from .client import Hone, create_hone_client
from .types import (
    HoneClient,
    HoneAgent,
    HoneTrack,
    GetAgentOptions,
    AgentNode,
    AgentRequest,
    AgentResponse,
    Hyperparameters,
    # Backwards compatibility
    HonePrompt,
    GetPromptOptions,
    PromptNode,
    PromptRequest,
    PromptResponse,
)
from .agent import (
    get_agent_node,
    evaluate_agent,
    format_agent_request,
    update_agent_nodes,
    traverse_agent_node,
    insert_params_into_prompt,
    # Backwards compatibility
    get_prompt_node,
    evaluate_prompt,
    format_prompt_request,
    update_prompt_nodes,
    traverse_prompt_node,
)

__all__ = [
    # Client
    "Hone",
    "create_hone_client",
    # Types
    "HoneClient",
    "HoneAgent",
    "HoneTrack",
    "GetAgentOptions",
    "AgentNode",
    "AgentRequest",
    "AgentResponse",
    "Hyperparameters",
    # Functions
    "get_agent_node",
    "evaluate_agent",
    "format_agent_request",
    "update_agent_nodes",
    "traverse_agent_node",
    "insert_params_into_prompt",
    # Backwards compatibility
    "HonePrompt",
    "GetPromptOptions",
    "PromptNode",
    "PromptRequest",
    "PromptResponse",
    "get_prompt_node",
    "evaluate_prompt",
    "format_prompt_request",
    "update_prompt_nodes",
    "traverse_prompt_node",
]

__version__ = "0.1.0"
