"""
Tool tracking helpers for the Hone SDK.

These utilities help format tool calls and results for tracking conversations
that include function calling / tool use.

Exact replica of TypeScript tools.ts.
"""

import json
from typing import Any, Dict, List, Optional, Union

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
