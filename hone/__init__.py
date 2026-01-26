"""
Hone SDK - AI Experience Engineering Platform.

Public API exports matching TypeScript index.ts.
"""

from .client import Hone, create_hone_client
from .providers import (
    AIProvider,
    AIProviderValue,
    AI_PROVIDER_VALUES,
    is_valid_provider,
    get_provider_display_name,
)
from .types import (
    # Config
    HoneConfig,
    # Entity Types
    EntityType,
    # Agent Types
    HoneClient,
    HoneAgent,
    GetAgentOptions,
    AgentResult,
    Hyperparameters,
    # Tool Types
    HoneTool,
    GetToolOptions,
    ToolResult,
    # Text Prompt Types
    HoneTextPrompt,
    GetTextPromptOptions,
    TextPromptResult,
    # Tracking Types
    HoneTrack,
    Message,
    ToolCall,
    TrackConversationOptions,
    # Node Types
    EntityNode,
    AgentNode,
    ToolNode,
    TextPromptNode,
    # Request/Response Types
    EntityRequest,
    EntityResponse,
    EntityRequestItem,
    EntityResponseItem,
    # Params Types
    Params,
    ParamsValue,
    SimpleParams,
    # Backwards compatibility
    HonePrompt,
    GetPromptOptions,
    PromptNode,
    PromptRequest,
    PromptResponse,
    AgentRequest,
    AgentResponse,
)
from .agent import (
    # Entity functions
    get_agent_node,
    get_tool_node,
    get_text_prompt_node,
    evaluate_agent,
    evaluate_entity,
    format_entity_request,
    format_agent_request,
    update_agent_nodes,
    update_entity_nodes,
    traverse_agent_node,
    traverse_entity_node,
    insert_params_into_prompt,
    # Backwards compatibility
    get_prompt_node,
    evaluate_prompt,
    format_prompt_request,
    update_prompt_nodes,
    traverse_prompt_node,
)
from .tools import (
    # Tool tracking helpers
    create_tool_call_message,
    create_tool_result_message,
    extract_openai_messages,
    extract_anthropic_messages,
    extract_gemini_messages,
    # Input normalizers (for manual use)
    normalize_openai_messages,
    normalize_anthropic_messages,
    normalize_gemini_contents,
    # Short aliases (recommended)
    tool_result,
    from_openai,
    from_anthropic,
    from_gemini,
)

__all__ = [
    # Client
    "Hone",
    "create_hone_client",
    # Providers
    "AIProvider",
    "AIProviderValue",
    "AI_PROVIDER_VALUES",
    "is_valid_provider",
    "get_provider_display_name",
    # Config
    "HoneConfig",
    # Entity Types
    "EntityType",
    # Agent Types
    "HoneClient",
    "HoneAgent",
    "GetAgentOptions",
    "AgentResult",
    "Hyperparameters",
    # Tool Types
    "HoneTool",
    "GetToolOptions",
    "ToolResult",
    # Text Prompt Types
    "HoneTextPrompt",
    "GetTextPromptOptions",
    "TextPromptResult",
    # Tracking Types
    "HoneTrack",
    "Message",
    "ToolCall",
    "TrackConversationOptions",
    # Node Types
    "EntityNode",
    "AgentNode",
    "ToolNode",
    "TextPromptNode",
    # Request/Response Types
    "EntityRequest",
    "EntityResponse",
    "EntityRequestItem",
    "EntityResponseItem",
    # Params Types
    "Params",
    "ParamsValue",
    "SimpleParams",
    # Entity functions
    "get_agent_node",
    "get_tool_node",
    "get_text_prompt_node",
    "evaluate_agent",
    "evaluate_entity",
    "format_entity_request",
    "format_agent_request",
    "update_agent_nodes",
    "update_entity_nodes",
    "traverse_agent_node",
    "traverse_entity_node",
    "insert_params_into_prompt",
    # Tool tracking helpers
    "create_tool_call_message",
    "create_tool_result_message",
    "extract_openai_messages",
    "extract_anthropic_messages",
    "extract_gemini_messages",
    # Input normalizers
    "normalize_openai_messages",
    "normalize_anthropic_messages",
    "normalize_gemini_contents",
    # Short aliases
    "tool_result",
    "from_openai",
    "from_anthropic",
    "from_gemini",
    # Backwards compatibility
    "HonePrompt",
    "GetPromptOptions",
    "PromptNode",
    "PromptRequest",
    "PromptResponse",
    "AgentRequest",
    "AgentResponse",
    "get_prompt_node",
    "evaluate_prompt",
    "format_prompt_request",
    "update_prompt_nodes",
    "traverse_prompt_node",
]

__version__ = "0.1.0"
