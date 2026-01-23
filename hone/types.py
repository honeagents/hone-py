"""
Hone SDK Type Definitions.

Exact replica of TypeScript types.ts - defines all types used by the SDK.
"""

from typing import Any, Callable, Coroutine, Dict, List, Literal, Optional, Protocol, TypedDict, Union


class HoneConfig(TypedDict, total=False):
    """Configuration for the Hone client."""
    api_key: str  # Required
    base_url: str  # Optional
    timeout: int  # Optional (milliseconds)


# =============================================================================
# Entity Types
# =============================================================================

EntityType = Literal["agent", "tool", "prompt"]


# =============================================================================
# Params Types
# =============================================================================

ParamsValue = Union[str, "GetTextPromptOptions"]
Params = Dict[str, ParamsValue]
SimpleParams = Dict[str, str]


# =============================================================================
# Hyperparameters
# =============================================================================

class Hyperparameters(TypedDict, total=False):
    """
    Hyperparameters for LLM configuration.
    Used by agents to configure LLM behavior.
    """
    model: str  # LLM model identifier (e.g., "gpt-4", "claude-3-opus")
    provider: str  # LLM provider identifier (e.g., "openai", "anthropic")
    temperature: float  # Sampling temperature (0.00 to 2.00)
    max_tokens: int  # Maximum output tokens
    top_p: float  # Nucleus sampling parameter (0.00 to 1.00)
    frequency_penalty: float  # Repetition penalty (-2.00 to 2.00)
    presence_penalty: float  # Topic diversity penalty (-2.00 to 2.00)
    stop_sequences: List[str]  # Array of stop tokens
    tools: List[str]  # Array of tool IDs this agent can use


# =============================================================================
# Agent Types
# =============================================================================

class GetAgentOptions(Hyperparameters, total=False):
    """
    Options for fetching and evaluating an agent.

    Attributes:
        model: REQUIRED - LLM model identifier (e.g., "gpt-4", "claude-3-opus")
        provider: REQUIRED - LLM provider identifier (e.g., "openai", "anthropic")
        major_version: The major version of the agent. SDK controls this value.
                       When major_version changes, minor_version resets to 0.
                       If not specified, defaults to 1.
        name: Optional name for the agent for easier identification. Will fallback to id if not provided.
        params: Parameters to substitute into the prompt. You can also nest agent calls here.
        default_prompt: The default prompt to use if none is found in the database.
                       The use of variables should be in the form {{variableName}}.
        extra: Custom extra data to store with the agent. This data is stored in the
               database and returned in the AgentResult.
    """
    major_version: int
    name: str
    params: Dict[str, Union[str, "GetAgentOptions"]]
    default_prompt: str  # Required in practice
    extra: Dict[str, Any]  # Custom extra data


class AgentResult(TypedDict, total=False):
    """
    The result returned by hone.agent().
    Contains the evaluated system prompt and hyperparameters.
    """
    system_prompt: str  # The fully evaluated system prompt with all parameters substituted
    model: str  # LLM model identifier
    provider: str  # LLM provider
    temperature: Optional[float]  # Sampling temperature
    max_tokens: Optional[int]  # Maximum output tokens
    top_p: Optional[float]  # Nucleus sampling parameter
    frequency_penalty: Optional[float]  # Repetition penalty
    presence_penalty: Optional[float]  # Topic diversity penalty
    stop_sequences: List[str]  # Array of stop tokens
    tools: List[str]  # Array of allowed tool IDs


# =============================================================================
# Tool Types
# =============================================================================

class GetToolOptions(TypedDict, total=False):
    """
    Options for fetching a tool.
    Tools don't have hyperparameters - they're just versioned text templates.

    Attributes:
        major_version: The major version of the tool. SDK controls this value.
                       When majorVersion changes, minorVersion resets to 0.
                       If not specified, defaults to 1.
        name: Optional name for the tool for easier identification.
              Will fallback to id if not provided.
        params: Parameters to substitute into the prompt.
        default_prompt: The default prompt/description to use if none is found in the database.
                       The use of variables should be in the form {{variableName}}.
    """
    major_version: int
    name: str
    params: Dict[str, Union[str, "GetTextPromptOptions"]]
    default_prompt: str  # Required in practice


class ToolResult(TypedDict):
    """
    The result returned by hone.tool().
    Contains the evaluated prompt with parameters substituted.
    """
    prompt: str  # The fully evaluated prompt with all parameters substituted


# =============================================================================
# Text Prompt Types
# =============================================================================

class GetTextPromptOptions(TypedDict, total=False):
    """
    Options for fetching a text prompt.
    Text prompts are simple versioned text templates with no hyperparameters.
    They can be nested inside agents, tools, or other prompts.

    Attributes:
        major_version: The major version of the prompt. SDK controls this value.
                       When majorVersion changes, minorVersion resets to 0.
                       If not specified, defaults to 1.
        name: Optional name for the prompt for easier identification.
              Will fallback to id if not provided.
        params: Parameters to substitute into the prompt.
        default_prompt: The default text to use if none is found in the database.
                       The use of variables should be in the form {{variableName}}.
    """
    major_version: int
    name: str
    params: Dict[str, Union[str, "GetTextPromptOptions"]]
    default_prompt: str  # Required in practice


class TextPromptResult(TypedDict):
    """
    The result returned by hone.prompt().
    Contains the evaluated text with parameters substituted.
    """
    text: str  # The fully evaluated text with all parameters substituted


# =============================================================================
# Tracking Types
# =============================================================================

class ToolCall(TypedDict):
    """
    Represents a tool call made by the assistant.
    Compatible with OpenAI's function calling format.
    """
    id: str  # Unique identifier for this tool call
    name: str  # Name of the tool/function being called
    arguments: str  # JSON string of the arguments to pass to the tool


class Message(TypedDict, total=False):
    """
    A chat message with role and content.
    Supports tool calls and tool results.
    """
    role: Literal["user", "assistant", "system", "tool"]  # Required
    content: str  # Required
    tool_calls: List[ToolCall]  # Tool calls requested by the assistant (when role is "assistant")
    tool_call_id: str  # ID of the tool call this message is responding to (when role is "tool")


class TrackConversationOptions(TypedDict):
    """Options for tracking a conversation."""
    session_id: str


class TrackRequest(TypedDict):
    """The request payload sent to the /insert_runs endpoint."""
    id: str
    messages: List[Message]
    sessionId: str  # camelCase to match TypeScript API
    timestamp: str


# TrackResponse is void (None in Python)
TrackResponse = None


# =============================================================================
# Internal Node Types
# =============================================================================

class EntityNode(TypedDict, total=False):
    """Internal representation of an entity node in the tree."""
    id: str
    type: EntityType
    name: Optional[str]
    major_version: Optional[int]
    params: SimpleParams
    prompt: str
    children: List["EntityNode"]
    # Hyperparameters (for agents)
    model: Optional[str]
    provider: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    stop_sequences: Optional[List[str]]
    tools: Optional[List[str]]


# Type aliases for specific entity nodes
AgentNode = EntityNode  # Agent node has type="agent"
ToolNode = EntityNode  # Tool node has type="tool"
TextPromptNode = EntityNode  # Text prompt node has type="prompt"


# =============================================================================
# API Request/Response Types
# =============================================================================

class EntityRequestItem(TypedDict, total=False):
    """A single entity item in the API request."""
    id: str
    type: EntityType
    name: Optional[str]
    majorVersion: Optional[int]  # camelCase to match TypeScript API
    prompt: str
    paramKeys: List[str]  # camelCase to match TypeScript API
    childrenIds: List[str]  # camelCase to match TypeScript API
    childrenTypes: List[EntityType]
    # Hyperparameters
    model: Optional[str]
    provider: Optional[str]
    temperature: Optional[float]
    maxTokens: Optional[int]  # camelCase to match TypeScript API
    topP: Optional[float]  # camelCase to match TypeScript API
    frequencyPenalty: Optional[float]  # camelCase to match TypeScript API
    presencePenalty: Optional[float]  # camelCase to match TypeScript API
    stopSequences: Optional[List[str]]  # camelCase to match TypeScript API
    tools: Optional[List[str]]


class EntityRequestPayload(TypedDict):
    """The entities payload structure."""
    rootId: str  # camelCase to match TypeScript API
    rootType: EntityType
    map: Dict[str, EntityRequestItem]


class EntityRequest(TypedDict):
    """The request payload sent to the /sync_entities endpoint."""
    entities: EntityRequestPayload


class EntityResponseItem(TypedDict, total=False):
    """A single entity response item."""
    prompt: str
    type: EntityType
    # Agent-specific fields (null for tools)
    model: Optional[str]
    provider: Optional[str]
    temperature: Optional[float]
    maxTokens: Optional[int]
    topP: Optional[float]
    frequencyPenalty: Optional[float]
    presencePenalty: Optional[float]
    stopSequences: List[str]
    tools: List[str]
    # Custom extra data
    extra: Optional[Dict[str, Any]]


# The response received from the /sync_entities endpoint
# Key: entity ID, Value: the entity data
EntityResponse = Dict[str, EntityResponseItem]


# =============================================================================
# Function Type Aliases
# =============================================================================

HoneAgent = Callable[[str, GetAgentOptions], Coroutine[Any, Any, AgentResult]]
HoneTool = Callable[[str, GetToolOptions], Coroutine[Any, Any, ToolResult]]
HoneTextPrompt = Callable[[str, GetTextPromptOptions], Coroutine[Any, Any, TextPromptResult]]
HoneTrack = Callable[[str, List[Message], TrackConversationOptions], Coroutine[Any, Any, None]]


# =============================================================================
# Client Protocol
# =============================================================================

class HoneClient(Protocol):
    """
    Protocol for the Hone client interface.
    Matches the TypeScript HoneClient type.
    """

    async def agent(self, id: str, options: GetAgentOptions) -> AgentResult:
        """
        Fetches and evaluates an agent by its ID with the given options.

        Args:
            id: The unique identifier for the agent.
            options: Options for fetching and evaluating the agent. Model and provider are required.

        Returns:
            An AgentResult containing systemPrompt and hyperparameters.
        """
        ...

    async def tool(self, id: str, options: GetToolOptions) -> ToolResult:
        """
        Fetches and evaluates a tool by its ID with the given options.

        Args:
            id: The unique identifier for the tool.
            options: Options for fetching and evaluating the tool.

        Returns:
            A ToolResult containing the evaluated prompt.
        """
        ...

    async def prompt(self, id: str, options: GetTextPromptOptions) -> TextPromptResult:
        """
        Fetches and evaluates a text prompt by its ID with the given options.

        Args:
            id: The unique identifier for the prompt.
            options: Options for fetching and evaluating the prompt.

        Returns:
            A TextPromptResult containing the evaluated text.
        """
        ...

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
        ...


# =============================================================================
# Backwards Compatibility Aliases (deprecated)
# =============================================================================

# Old Agent request/response types for sync_agents endpoint compatibility
class AgentRequestItem(TypedDict, total=False):
    """A single agent item in the API request. DEPRECATED: Use EntityRequestItem."""
    id: str
    name: Optional[str]
    majorVersion: Optional[int]  # camelCase to match TypeScript API
    prompt: str
    paramKeys: List[str]  # camelCase to match TypeScript API
    childrenIds: List[str]  # camelCase to match TypeScript API
    # Hyperparameters
    model: Optional[str]
    temperature: Optional[float]
    maxTokens: Optional[int]  # camelCase to match TypeScript API
    topP: Optional[float]  # camelCase to match TypeScript API
    frequencyPenalty: Optional[float]  # camelCase to match TypeScript API
    presencePenalty: Optional[float]  # camelCase to match TypeScript API
    stopSequences: Optional[List[str]]  # camelCase to match TypeScript API


class AgentRequestPayload(TypedDict):
    """The agents payload structure. DEPRECATED: Use EntityRequestPayload."""
    rootId: str  # camelCase to match TypeScript API
    map: Dict[str, AgentRequestItem]


class AgentRequest(TypedDict):
    """The request payload sent to the /sync_agents endpoint. DEPRECATED: Use EntityRequest."""
    agents: AgentRequestPayload


class AgentResponseItem(TypedDict, total=False):
    """A single agent response item. DEPRECATED: Use EntityResponseItem."""
    prompt: str
    model: Optional[str]
    provider: Optional[str]
    temperature: Optional[float]
    maxTokens: Optional[int]
    topP: Optional[float]
    frequencyPenalty: Optional[float]
    presencePenalty: Optional[float]
    stopSequences: List[str]
    tools: List[str]


# The response received from the /sync_agents endpoint
AgentResponse = Dict[str, AgentResponseItem]


# Deprecated prompt aliases
GetPromptOptions = GetAgentOptions
PromptNode = AgentNode
PromptRequestItem = AgentRequestItem
PromptRequestPayload = AgentRequestPayload
PromptRequest = AgentRequest
PromptResponseItem = AgentResponseItem
PromptResponse = AgentResponse
HonePrompt = HoneAgent
