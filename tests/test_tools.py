"""
Unit tests for Hone SDK tools module.

Exact replica of TypeScript tools.test.ts - tests tool tracking helpers and normalizers.
"""

import json
import pytest

from hone.tools import (
    create_tool_call_message,
    create_tool_result_message,
    extract_openai_messages,
    extract_anthropic_messages,
    extract_gemini_messages,
    normalize_openai_messages,
    normalize_anthropic_messages,
    normalize_gemini_contents,
    tool_result,
    from_openai,
    from_anthropic,
    from_gemini,
)


class TestCreateToolCallMessage:
    """Tests for create_tool_call_message function."""

    def test_should_create_assistant_message_with_tool_calls(self):
        """Should create an assistant message with tool calls."""
        tool_calls = [
            {"id": "call_123", "name": "get_weather", "arguments": '{"location":"SF"}'},
        ]

        message = create_tool_call_message(tool_calls)

        assert message == {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls,
        }

    def test_should_include_content_when_provided(self):
        """Should include content when provided."""
        tool_calls = [
            {"id": "call_123", "name": "get_weather", "arguments": '{"location":"SF"}'},
        ]

        message = create_tool_call_message(tool_calls, "Let me check the weather")

        assert message == {
            "role": "assistant",
            "content": "Let me check the weather",
            "tool_calls": tool_calls,
        }

    def test_should_handle_multiple_tool_calls(self):
        """Should handle multiple tool calls."""
        tool_calls = [
            {"id": "call_1", "name": "get_weather", "arguments": '{"location":"SF"}'},
            {"id": "call_2", "name": "get_time", "arguments": '{"timezone":"PST"}'},
        ]

        message = create_tool_call_message(tool_calls)

        assert len(message["tool_calls"]) == 2
        assert message["tool_calls"][0]["name"] == "get_weather"
        assert message["tool_calls"][1]["name"] == "get_time"


class TestCreateToolResultMessage:
    """Tests for create_tool_result_message function."""

    def test_should_create_tool_result_with_string_content(self):
        """Should create a tool result message with string content."""
        message = create_tool_result_message("call_123", "72F and sunny")

        assert message == {
            "role": "tool",
            "content": "72F and sunny",
            "tool_call_id": "call_123",
        }

    def test_should_stringify_object_results(self):
        """Should stringify object results."""
        message = create_tool_result_message("call_123", {"temp": 72, "unit": "F"})

        assert message == {
            "role": "tool",
            "content": '{"temp": 72, "unit": "F"}',
            "tool_call_id": "call_123",
        }

    def test_should_stringify_array_results(self):
        """Should stringify array results."""
        message = create_tool_result_message("call_123", [1, 2, 3])

        assert message == {
            "role": "tool",
            "content": "[1, 2, 3]",
            "tool_call_id": "call_123",
        }

    def test_should_handle_null(self):
        """Should handle None/null."""
        message = create_tool_result_message("call_123", None)
        assert message["content"] == "null"


class TestToolResultAlias:
    """Tests for tool_result alias."""

    def test_should_be_alias_for_create_tool_result_message(self):
        """tool_result should be an alias for create_tool_result_message."""
        assert tool_result is create_tool_result_message


class TestExtractOpenAIMessages:
    """Tests for extract_openai_messages / from_openai."""

    def test_should_extract_simple_assistant_message(self):
        """Should extract a simple assistant message."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    },
                },
            ],
        }

        messages = extract_openai_messages(response)

        assert len(messages) == 1
        assert messages[0] == {
            "role": "assistant",
            "content": "Hello!",
        }

    def test_should_extract_message_with_tool_calls(self):
        """Should extract message with tool calls."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"San Francisco"}',
                                },
                            },
                        ],
                    },
                },
            ],
        }

        messages = extract_openai_messages(response)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == ""
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[0]["tool_calls"][0] == {
            "id": "call_abc123",
            "name": "get_weather",
            "arguments": '{"location":"San Francisco"}',
        }

    def test_should_handle_multiple_tool_calls(self):
        """Should handle multiple tool calls."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'll check both",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"SF"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "get_time",
                                    "arguments": '{"timezone":"PST"}',
                                },
                            },
                        ],
                    },
                },
            ],
        }

        messages = extract_openai_messages(response)

        assert len(messages[0]["tool_calls"]) == 2
        assert messages[0]["tool_calls"][0]["name"] == "get_weather"
        assert messages[0]["tool_calls"][1]["name"] == "get_time"

    def test_should_handle_empty_tool_calls_array(self):
        """Should handle empty tool_calls array."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "No tools needed",
                        "tool_calls": [],
                    },
                },
            ],
        }

        messages = extract_openai_messages(response)

        assert "tool_calls" not in messages[0] or messages[0].get("tool_calls") is None

    def test_from_openai_should_be_alias(self):
        """from_openai should be an alias."""
        assert from_openai is extract_openai_messages


class TestExtractAnthropicMessages:
    """Tests for extract_anthropic_messages / from_anthropic."""

    def test_should_extract_simple_text_message(self):
        """Should extract a simple text message."""
        response = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
        }

        messages = extract_anthropic_messages(response)

        assert len(messages) == 1
        assert messages[0] == {
            "role": "assistant",
            "content": "Hello!",
        }

    def test_should_extract_tool_use_blocks(self):
        """Should extract tool use blocks."""
        response = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"},
                },
            ],
        }

        messages = extract_anthropic_messages(response)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[0]["tool_calls"][0] == {
            "id": "toolu_123",
            "name": "get_weather",
            "arguments": '{"location": "San Francisco"}',
        }

    def test_should_combine_text_and_tool_use_blocks(self):
        """Should combine text and tool use blocks."""
        response = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"location": "SF"},
                },
            ],
        }

        messages = extract_anthropic_messages(response)

        assert len(messages) == 1
        assert messages[0]["content"] == "Let me check the weather."
        assert len(messages[0]["tool_calls"]) == 1

    def test_should_handle_multiple_text_blocks(self):
        """Should handle multiple text blocks."""
        response = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "First part."},
                {"type": "text", "text": "Second part."},
            ],
        }

        messages = extract_anthropic_messages(response)

        assert messages[0]["content"] == "First part.\nSecond part."

    def test_should_handle_multiple_tool_use_blocks(self):
        """Should handle multiple tool use blocks."""
        response = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"location": "SF"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "get_time",
                    "input": {"timezone": "PST"},
                },
            ],
        }

        messages = extract_anthropic_messages(response)

        assert len(messages[0]["tool_calls"]) == 2
        assert messages[0]["tool_calls"][0]["name"] == "get_weather"
        assert messages[0]["tool_calls"][1]["name"] == "get_time"

    def test_from_anthropic_should_be_alias(self):
        """from_anthropic should be an alias."""
        assert from_anthropic is extract_anthropic_messages


class TestExtractGeminiMessages:
    """Tests for extract_gemini_messages / from_gemini."""

    def test_should_extract_simple_text_message(self):
        """Should extract a simple text message."""
        response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello!"}],
                    },
                },
            ],
        }

        messages = extract_gemini_messages(response)

        assert len(messages) == 1
        assert messages[0] == {
            "role": "assistant",
            "content": "Hello!",
        }

    def test_should_extract_function_calls(self):
        """Should extract function calls."""
        response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "San Francisco"},
                                },
                            },
                        ],
                    },
                },
            ],
        }

        messages = extract_gemini_messages(response)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[0]["tool_calls"][0]["name"] == "get_weather"
        assert messages[0]["tool_calls"][0]["arguments"] == '{"location": "San Francisco"}'
        # Gemini IDs are generated
        assert messages[0]["tool_calls"][0]["id"].startswith("gemini_get_weather_")

    def test_should_combine_text_and_function_calls(self):
        """Should combine text and function calls."""
        response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Let me check."},
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "SF"},
                                },
                            },
                        ],
                    },
                },
            ],
        }

        messages = extract_gemini_messages(response)

        assert len(messages) == 1
        assert messages[0]["content"] == "Let me check."
        assert len(messages[0]["tool_calls"]) == 1

    def test_should_handle_empty_candidates(self):
        """Should handle empty candidates."""
        response = {"candidates": []}
        messages = extract_gemini_messages(response)
        assert len(messages) == 0

    def test_should_handle_missing_candidates(self):
        """Should handle undefined candidates."""
        response = {}
        messages = extract_gemini_messages(response)
        assert len(messages) == 0

    def test_should_handle_multiple_function_calls(self):
        """Should handle multiple function calls."""
        response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "SF"},
                                },
                            },
                            {
                                "functionCall": {
                                    "name": "get_time",
                                    "args": {"timezone": "PST"},
                                },
                            },
                        ],
                    },
                },
            ],
        }

        messages = extract_gemini_messages(response)

        assert len(messages[0]["tool_calls"]) == 2
        assert messages[0]["tool_calls"][0]["name"] == "get_weather"
        assert messages[0]["tool_calls"][1]["name"] == "get_time"

    def test_from_gemini_should_be_alias(self):
        """from_gemini should be an alias."""
        assert from_gemini is extract_gemini_messages


class TestNormalizeOpenAIMessages:
    """Tests for normalize_openai_messages function."""

    def test_should_normalize_simple_messages(self):
        """Should normalize simple text messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        normalized = normalize_openai_messages(messages)

        assert len(normalized) == 3
        assert normalized[0] == {"role": "system", "content": "You are helpful."}
        assert normalized[1] == {"role": "user", "content": "Hello!"}
        assert normalized[2] == {"role": "assistant", "content": "Hi there!"}

    def test_should_normalize_assistant_with_tool_calls(self):
        """Should normalize assistant messages with tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"SF"}',
                        },
                    },
                ],
            },
        ]

        normalized = normalize_openai_messages(messages)

        assert len(normalized) == 1
        assert normalized[0]["role"] == "assistant"
        assert len(normalized[0]["tool_calls"]) == 1
        assert normalized[0]["tool_calls"][0]["name"] == "get_weather"

    def test_should_normalize_tool_messages(self):
        """Should normalize tool result messages."""
        messages = [
            {
                "role": "tool",
                "content": '{"temp": 72}',
                "tool_call_id": "call_123",
            },
        ]

        normalized = normalize_openai_messages(messages)

        assert len(normalized) == 1
        assert normalized[0] == {
            "role": "tool",
            "content": '{"temp": 72}',
            "tool_call_id": "call_123",
        }

    def test_should_handle_content_array(self):
        """Should handle content as an array of blocks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                ],
            },
        ]

        normalized = normalize_openai_messages(messages)

        assert normalized[0]["content"] == "First part.\nSecond part."


class TestNormalizeAnthropicMessages:
    """Tests for normalize_anthropic_messages function."""

    def test_should_normalize_string_content(self):
        """Should normalize messages with string content."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        normalized = normalize_anthropic_messages(messages)

        assert len(normalized) == 2
        assert normalized[0] == {"role": "user", "content": "Hello!"}
        assert normalized[1] == {"role": "assistant", "content": "Hi there!"}

    def test_should_normalize_content_blocks(self):
        """Should normalize content block arrays."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"location": "SF"},
                    },
                ],
            },
        ]

        normalized = normalize_anthropic_messages(messages)

        assert len(normalized) == 1
        assert normalized[0]["content"] == "Let me check."
        assert len(normalized[0]["tool_calls"]) == 1
        assert normalized[0]["tool_calls"][0]["name"] == "get_weather"

    def test_should_normalize_tool_results(self):
        """Should normalize tool result blocks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": '{"temp": 72}',
                    },
                ],
            },
        ]

        normalized = normalize_anthropic_messages(messages)

        assert len(normalized) == 1
        assert normalized[0] == {
            "role": "tool",
            "content": '{"temp": 72}',
            "tool_call_id": "toolu_123",
        }


class TestNormalizeGeminiContents:
    """Tests for normalize_gemini_contents function."""

    def test_should_normalize_simple_text(self):
        """Should normalize simple text contents."""
        contents = [
            {"role": "user", "parts": [{"text": "Hello!"}]},
            {"role": "model", "parts": [{"text": "Hi there!"}]},
        ]

        normalized = normalize_gemini_contents(contents)

        assert len(normalized) == 2
        assert normalized[0] == {"role": "user", "content": "Hello!"}
        assert normalized[1] == {"role": "assistant", "content": "Hi there!"}

    def test_should_normalize_function_calls(self):
        """Should normalize function call parts."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "SF"},
                        },
                    },
                ],
            },
        ]

        normalized = normalize_gemini_contents(contents)

        assert len(normalized) == 1
        assert normalized[0]["role"] == "assistant"
        assert len(normalized[0]["tool_calls"]) == 1
        assert normalized[0]["tool_calls"][0]["name"] == "get_weather"

    def test_should_normalize_function_responses(self):
        """Should normalize function response parts."""
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"temp": 72},
                        },
                    },
                ],
            },
        ]

        normalized = normalize_gemini_contents(contents)

        assert len(normalized) == 1
        assert normalized[0]["role"] == "tool"
        assert normalized[0]["content"] == '{"temp": 72}'
        assert normalized[0]["tool_call_id"] == "get_weather"


class TestToolImports:
    """Tests for tool module imports."""

    def test_should_be_importable_from_hone_package(self):
        """Should be importable from the main hone package."""
        from hone import (
            create_tool_call_message,
            create_tool_result_message,
            extract_openai_messages,
            extract_anthropic_messages,
            extract_gemini_messages,
            normalize_openai_messages,
            normalize_anthropic_messages,
            normalize_gemini_contents,
            tool_result,
            from_openai,
            from_anthropic,
            from_gemini,
        )

        assert create_tool_call_message is not None
        assert create_tool_result_message is not None
        assert extract_openai_messages is not None
        assert extract_anthropic_messages is not None
        assert extract_gemini_messages is not None
        assert normalize_openai_messages is not None
        assert normalize_anthropic_messages is not None
        assert normalize_gemini_contents is not None
        assert tool_result is not None
        assert from_openai is not None
        assert from_anthropic is not None
        assert from_gemini is not None

    def test_normalizers_should_be_in_all_exports(self):
        """Normalizers should be included in __all__."""
        import hone

        assert "normalize_openai_messages" in hone.__all__
        assert "normalize_anthropic_messages" in hone.__all__
        assert "normalize_gemini_contents" in hone.__all__
