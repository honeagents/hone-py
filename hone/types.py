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


class Hyperparameters(TypedDict, total=False):
    """
    Hyperparameters for LLM configuration.
    """
    model: str  # LLM model identifier (e.g., "gpt-4", "claude-3-opus")
    temperature: float  # Sampling temperature (0.00 to 2.00)
    max_tokens: int  # Maximum output tokens
    top_p: float  # Nucleus sampling parameter (0.00 to 1.00)
    frequency_penalty: float  # Repetition penalty (-2.00 to 2.00)
    presence_penalty: float  # Topic diversity penalty (-2.00 to 2.00)
    stop_sequences: List[str]  # Array of stop tokens


class GetAgentOptions(Hyperparameters, total=False):
    """
    Options for fetching and evaluating an agent.

    Attributes:
        major_version: The major version of the agent. SDK controls this value.
                       When major_version changes, minor_version resets to 0.
                       If not specified, defaults to 1.
        name: Optional name for the agent for easier identification. Will fallback to id if not provided.
        params: Parameters to substitute into the prompt. You can also nest agent calls here.
        default_prompt: The default prompt to use if none is found in the database.
                       The use of variables should be in the form {{variableName}}.
    """
    major_version: int
    name: str
    params: Dict[str, Union[str, "GetAgentOptions"]]
    default_prompt: str  # Required in practice


# Type aliases
ParamsValue = Union[str, GetAgentOptions]
Params = Dict[str, ParamsValue]
SimpleParams = Dict[str, str]


class Message(TypedDict):
    """A chat message with role and content."""
    role: Literal["user", "assistant", "system"]
    content: str


class TrackConversationOptions(TypedDict):
    """Options for tracking a conversation."""
    session_id: str


class AgentNode(TypedDict, total=False):
    """Internal representation of an agent node in the tree."""
    id: str
    name: Optional[str]
    major_version: Optional[int]
    params: SimpleParams
    prompt: str
    children: List["AgentNode"]
    # Hyperparameters
    model: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    stop_sequences: Optional[List[str]]


class AgentRequestItem(TypedDict, total=False):
    """A single agent item in the API request."""
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
    """The agents payload structure."""
    rootId: str  # camelCase to match TypeScript API
    map: Dict[str, AgentRequestItem]


class AgentRequest(TypedDict):
    """The request payload sent to the /sync_agents endpoint."""
    agents: AgentRequestPayload


class AgentResponseItem(TypedDict):
    """A single agent response item."""
    prompt: str


# The response received from the /sync_agents endpoint
# Key: agent ID, Value: the newest prompt string
AgentResponse = Dict[str, AgentResponseItem]


class TrackRequest(TypedDict):
    """The request payload sent to the /insert_runs endpoint."""
    id: str
    messages: List[Message]
    sessionId: str  # camelCase to match TypeScript API
    timestamp: str


# TrackResponse is void (None in Python)
TrackResponse = None


# Type aliases for function signatures
HoneAgent = Callable[[str, GetAgentOptions], Coroutine[Any, Any, str]]
HoneTrack = Callable[[str, List[Message], TrackConversationOptions], Coroutine[Any, Any, None]]


class HoneClient(Protocol):
    """
    Protocol for the Hone client interface.
    Matches the TypeScript HoneClient type.
    """

    async def agent(self, id: str, options: GetAgentOptions) -> str:
        """
        Fetches and evaluates an agent by its ID with the given options.

        Args:
            id: The unique identifier for the agent.
            options: Options for fetching and evaluating the agent.

        Returns:
            The evaluated prompt string.
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


# ============================================================================
# Backwards Compatibility Aliases (deprecated)
# ============================================================================

# Deprecated: Use GetAgentOptions instead
GetPromptOptions = GetAgentOptions

# Deprecated: Use AgentNode instead
PromptNode = AgentNode

# Deprecated: Use AgentRequestItem instead
PromptRequestItem = AgentRequestItem

# Deprecated: Use AgentRequestPayload instead
PromptRequestPayload = AgentRequestPayload

# Deprecated: Use AgentRequest instead
PromptRequest = AgentRequest

# Deprecated: Use AgentResponseItem instead
PromptResponseItem = AgentResponseItem

# Deprecated: Use AgentResponse instead
PromptResponse = AgentResponse

# Deprecated: Use HoneAgent instead
HonePrompt = HoneAgent
