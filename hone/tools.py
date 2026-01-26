"""
Tool tracking helpers for the Hone SDK.

These utilities help format tool calls and results for tracking conversations
that include function calling / tool use.

Exact replica of TypeScript tools.ts.
"""

import json
from typing import Any, Dict, List

from .types import Message, ToolCall


def create_tool_call_message(
    tool_calls: List[ToolCall],
    content: str = "",
) -> Message:
    """
    Creates an assistant message containing tool calls.

    Args:
        tool_calls: Array of tool calls the assistant is requesting
        content: Optional text content alongside tool calls (usually empty)

    Returns:
        A Message object formatted for tool call requests

    Example:
        >>> message = create_tool_call_message([
        ...     {"id": "call_abc123", "name": "get_weather", "arguments": '{"location":"SF"}'}
        ... ])
        >>> # {"role": "assistant", "content": "", "tool_calls": [...]}
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }


def create_tool_result_message(
    tool_call_id: str,
    result: Any,
) -> Message:
    """
    Creates a tool result message responding to a specific tool call.

    Args:
        tool_call_id: The ID of the tool call this result responds to
        result: The result from executing the tool (will be JSON stringified if not a string)

    Returns:
        A Message object formatted as a tool response

    Example:
        >>> message = create_tool_result_message("call_abc123", {"temp": 72, "unit": "F"})
        >>> # {"role": "tool", "content": '{"temp":72,"unit":"F"}', "tool_call_id": "call_abc123"}
    """
    content = result if isinstance(result, str) else json.dumps(result)
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
    }


def extract_openai_messages(response: Dict[str, Any]) -> List[Message]:
    """
    Extracts messages from an OpenAI chat completion response.

    Handles both regular assistant messages and messages with tool calls.

    Args:
        response: The OpenAI chat completion response object

    Returns:
        Array of Message objects ready to be tracked

    Example:
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> response = await client.chat.completions.create(
        ...     model="gpt-4o",
        ...     messages=[...],
        ...     tools=[...]
        ... )
        >>> messages = extract_openai_messages(response.model_dump())
        >>> await hone.track("conversation", [...existing_messages, *messages], {"session_id": session_id})
    """
    messages: List[Message] = []

    choices = response.get("choices", [])
    for choice in choices:
        msg = choice.get("message", {})
        message: Message = {
            "role": msg.get("role", "assistant"),
            "content": msg.get("content") or "",
        }

        tool_calls = msg.get("tool_calls")
        if tool_calls and len(tool_calls) > 0:
            message["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                }
                for tc in tool_calls
            ]

        messages.append(message)

    return messages


def extract_anthropic_messages(response: Dict[str, Any]) -> List[Message]:
    """
    Extracts messages from an Anthropic Claude response.

    Handles both text responses and tool use blocks.

    Args:
        response: The Anthropic message response object

    Returns:
        Array of Message objects ready to be tracked

    Example:
        >>> from anthropic import Anthropic
        >>> client = Anthropic()
        >>> response = await client.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     messages=[...],
        ...     tools=[...]
        ... )
        >>> messages = extract_anthropic_messages(response.model_dump())
        >>> await hone.track("conversation", [...existing_messages, *messages], {"session_id": session_id})
    """
    messages: List[Message] = []

    content_blocks = response.get("content", [])

    text_blocks = [block for block in content_blocks if block.get("type") == "text"]
    tool_use_blocks = [block for block in content_blocks if block.get("type") == "tool_use"]

    text_content = "\n".join(block.get("text", "") for block in text_blocks)

    if tool_use_blocks:
        tool_calls: List[ToolCall] = [
            {
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {})),
            }
            for block in tool_use_blocks
        ]

        messages.append({
            "role": "assistant",
            "content": text_content,
            "tool_calls": tool_calls,
        })
    else:
        messages.append({
            "role": response.get("role", "assistant"),
            "content": text_content,
        })

    return messages


def extract_gemini_messages(response: Dict[str, Any]) -> List[Message]:
    """
    Extracts messages from a Google Gemini response.

    Handles both text responses and function call parts.
    Note: Gemini doesn't provide unique IDs for function calls, so we generate
    them using the format `gemini_{functionName}_{index}`.

    Args:
        response: The Gemini GenerateContentResponse object

    Returns:
        Array of Message objects ready to be tracked

    Example:
        >>> from google.generativeai import GenerativeModel
        >>> model = GenerativeModel("gemini-pro")
        >>> response = await model.generate_content(
        ...     contents=[...],
        ...     tools=[{"function_declarations": [...]}]
        ... )
        >>> messages = extract_gemini_messages(response.to_dict())
        >>> await hone.track("conversation", [...existing_messages, *messages], {"session_id": session_id})
    """
    import time

    messages: List[Message] = []

    candidates = response.get("candidates", [])
    if not candidates:
        return messages

    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        if not parts:
            continue

        text_parts: List[str] = []
        function_calls: List[Dict[str, Any]] = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                function_calls.append(part["functionCall"])

        text_content = "\n".join(text_parts)

        if function_calls:
            # Gemini doesn't provide tool call IDs, so we generate them
            tool_calls: List[ToolCall] = [
                {
                    "id": f"gemini_{fc.get('name', 'unknown')}_{i}_{int(time.time() * 1000)}",
                    "name": fc.get("name", ""),
                    "arguments": json.dumps(fc.get("args", {})),
                }
                for i, fc in enumerate(function_calls)
            ]

            messages.append({
                "role": "assistant",
                "content": text_content,
                "tool_calls": tool_calls,
            })
        elif text_content:
            role = content.get("role", "model")
            messages.append({
                "role": "assistant" if role == "model" else role,
                "content": text_content,
            })

    return messages


# =============================================================================
# Input Message Normalizers (for zero-friction tracking)
# =============================================================================


def normalize_openai_messages(messages: List[Dict[str, Any]]) -> List[Message]:
    """
    Normalizes OpenAI input messages to Hone's Message format.

    Args:
        messages: Array of OpenAI ChatCompletionMessageParam

    Returns:
        Array of normalized Message objects

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"},
        ... ]
        >>> normalized = normalize_openai_messages(messages)
    """
    result: List[Message] = []

    for m in messages:
        role = m.get("role", "")

        # Handle system, user, assistant messages
        if role in ("system", "user", "assistant"):
            content = m.get("content", "")

            # Handle content that might be an array of content blocks
            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text_parts.append(c.get("text", ""))
                content = "\n".join(text_parts)
            elif content is None:
                content = ""

            message: Message = {
                "role": role,
                "content": content,
            }

            # Handle tool calls on assistant messages
            if role == "assistant" and "tool_calls" in m and m["tool_calls"]:
                tool_calls = m["tool_calls"]
                message["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", "{}"),
                    }
                    for tc in tool_calls
                    if tc.get("type") == "function"
                ]

            result.append(message)

        elif role == "tool":
            content = m.get("content", "")
            if not isinstance(content, str):
                content = ""
            result.append({
                "role": "tool",
                "content": content,
                "tool_call_id": m.get("tool_call_id", ""),
            })

    return result


def normalize_anthropic_messages(messages: List[Dict[str, Any]]) -> List[Message]:
    """
    Normalizes Anthropic input messages to Hone's Message format.
    Note: System prompt should be passed separately to track().

    Args:
        messages: Array of Anthropic MessageParam

    Returns:
        Array of normalized Message objects

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ... ]
        >>> normalized = normalize_anthropic_messages(messages)
    """
    result: List[Message] = []

    for m in messages:
        content = m.get("content", "")
        role = m.get("role", "user")

        if isinstance(content, str):
            result.append({
                "role": role,
                "content": content,
            })
        elif isinstance(content, list):
            # Handle content blocks
            text_parts: List[str] = []
            tool_calls: List[ToolCall] = []
            tool_results: List[Dict[str, str]] = []

            for block in content:
                block_type = block.get("type", "")

                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    })
                elif block_type == "tool_result":
                    block_content = block.get("content", "")
                    if isinstance(block_content, list):
                        # Extract text from content blocks
                        text_content = "\n".join(
                            c.get("text", "")
                            for c in block_content
                            if isinstance(c, dict) and c.get("type") == "text"
                        )
                        block_content = text_content
                    elif not isinstance(block_content, str):
                        block_content = ""
                    tool_results.append({
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": block_content,
                    })

            # Add text/tool_calls as assistant or user message
            if text_parts or tool_calls:
                message: Message = {
                    "role": role,
                    "content": "\n".join(text_parts),
                }
                if tool_calls:
                    message["tool_calls"] = tool_calls
                result.append(message)

            # Add tool results as separate tool messages
            for tr in tool_results:
                result.append({
                    "role": "tool",
                    "content": tr["content"],
                    "tool_call_id": tr["tool_use_id"],
                })

    return result


def normalize_gemini_contents(contents: List[Dict[str, Any]]) -> List[Message]:
    """
    Normalizes Gemini input contents to Hone's Message format.
    Note: System instruction should be passed separately to track().

    Args:
        contents: Array of Gemini Content

    Returns:
        Array of normalized Message objects

    Example:
        >>> contents = [
        ...     {"role": "user", "parts": [{"text": "Hello!"}]},
        ...     {"role": "model", "parts": [{"text": "Hi there!"}]},
        ... ]
        >>> normalized = normalize_gemini_contents(contents)
    """
    import time

    result: List[Message] = []

    for c in contents:
        parts = c.get("parts", [])
        role = c.get("role", "user")

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        tool_results: List[Dict[str, str]] = []

        for part in parts:
            if "text" in part and part["text"]:
                text_parts.append(part["text"])
            elif "functionCall" in part and part["functionCall"]:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": f"gemini_{fc.get('name', 'unknown')}_{int(time.time() * 1000)}",
                    "name": fc.get("name", ""),
                    "arguments": json.dumps(fc.get("args", {})),
                })
            elif "functionResponse" in part and part["functionResponse"]:
                fr = part["functionResponse"]
                tool_results.append({
                    "name": fr.get("name", ""),
                    "content": json.dumps(fr.get("response", {})),
                })

        # Map Gemini's "model" role to "assistant"
        mapped_role = "assistant" if role == "model" else "user"

        if text_parts or tool_calls:
            message: Message = {
                "role": mapped_role,
                "content": "\n".join(text_parts),
            }
            if tool_calls:
                message["tool_calls"] = tool_calls
            result.append(message)

        # Add function responses as tool messages
        for tr in tool_results:
            result.append({
                "role": "tool",
                "content": tr["content"],
                "tool_call_id": tr["name"],  # Gemini uses function name, not ID
            })

    return result


# =============================================================================
# Short Aliases (Recommended)
# =============================================================================

# Short alias for create_tool_result_message
tool_result = create_tool_result_message

# Short alias for extract_openai_messages
from_openai = extract_openai_messages

# Short alias for extract_anthropic_messages
from_anthropic = extract_anthropic_messages

# Short alias for extract_gemini_messages
from_gemini = extract_gemini_messages
