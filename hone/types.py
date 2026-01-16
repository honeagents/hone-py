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


class GetPromptOptions(TypedDict, total=False):
    """
    Options for fetching and evaluating a prompt.

    Attributes:
        version: The version of the prompt to retrieve. If not specified, the latest version is used.
                 Note: params and default_prompt are not updated remotely without version changes.
        name: Optional name for the prompt for easier identification. Will fallback to id if not provided.
        params: Parameters to substitute into the prompt. You can also nest prompt calls here.
        default_prompt: The default prompt to use if none is found in the database.
                       The use of variables should be in the form {{variableName}}.
    """
    version: str
    name: str
    params: Dict[str, Union[str, "GetPromptOptions"]]
    default_prompt: str  # Required in practice


# Type aliases
ParamsValue = Union[str, GetPromptOptions]
Params = Dict[str, ParamsValue]
SimpleParams = Dict[str, str]


class Message(TypedDict):
    """A chat message with role and content."""
    role: Literal["user", "assistant", "system"]
    content: str


class TrackConversationOptions(TypedDict):
    """Options for tracking a conversation."""
    session_id: str


class PromptNode(TypedDict):
    """Internal representation of a prompt node in the tree."""
    id: str
    name: Optional[str]
    version: Optional[str]
    params: SimpleParams
    prompt: str
    children: List["PromptNode"]


class PromptRequestItem(TypedDict):
    """A single prompt item in the API request."""
    id: str
    name: Optional[str]
    version: Optional[str]
    prompt: str
    paramKeys: List[str]  # camelCase to match TypeScript API
    childrenIds: List[str]  # camelCase to match TypeScript API


class PromptRequestPayload(TypedDict):
    """The prompts payload structure."""
    rootId: str  # camelCase to match TypeScript API
    map: Dict[str, PromptRequestItem]


class PromptRequest(TypedDict):
    """The request payload sent to the /sync_prompts endpoint."""
    prompts: PromptRequestPayload


class PromptResponseItem(TypedDict):
    """A single prompt response item."""
    prompt: str


# The response received from the /sync_prompts endpoint
# Key: prompt ID, Value: the newest prompt string
PromptResponse = Dict[str, PromptResponseItem]


class TrackRequest(TypedDict):
    """The request payload sent to the /insert_runs endpoint."""
    id: str
    messages: List[Message]
    sessionId: str  # camelCase to match TypeScript API
    timestamp: str


# TrackResponse is void (None in Python)
TrackResponse = None


# Type aliases for function signatures
HonePrompt = Callable[[str, GetPromptOptions], Coroutine[Any, Any, str]]
HoneTrack = Callable[[str, List[Message], TrackConversationOptions], Coroutine[Any, Any, None]]


class HoneClient(Protocol):
    """
    Protocol for the Hone client interface.
    Matches the TypeScript HoneClient type.
    """

    async def prompt(self, id: str, options: GetPromptOptions) -> str:
        """
        Fetches and evaluates a prompt by its ID with the given options.

        Args:
            id: The unique identifier for the prompt.
            options: Options for fetching and evaluating the prompt.

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
