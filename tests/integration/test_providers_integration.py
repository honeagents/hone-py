"""
Integration tests for the Hone SDK providers and normalizers.

These tests verify that the providers, extractors, and normalizers work
correctly together in realistic scenarios.
"""

from hone import (
    # Provider constants
    AIProvider,
    AI_PROVIDER_VALUES,
    is_valid_provider,
    get_provider_display_name,
    # Message extraction
    from_openai,
    from_anthropic,
    from_gemini,
    # Message normalization
    normalize_openai_messages,
    normalize_anthropic_messages,
    normalize_gemini_contents,
    # Tool helpers
    create_tool_call_message,
    tool_result,
)


class TestProvidersIntegration:
    """Integration tests for provider constants."""

    def test_all_providers_have_display_names(self):
        """Every provider should have a valid display name."""
        for provider_value in AI_PROVIDER_VALUES:
            display_name = get_provider_display_name(provider_value)
            # Display name should not be the same as the value (except for some)
            assert display_name is not None
            assert len(display_name) > 0

    def test_provider_validation_works_for_all_enum_values(self):
        """Validation should work for all enum values."""
        for provider in AIProvider:
            assert is_valid_provider(provider.value) is True

    def test_providers_can_be_used_in_agent_config(self):
        """Providers should work in agent configuration dicts."""
        # Simulate building an agent config
        config = {
            "provider": AIProvider.OPENAI,
            "model": "gpt-4o",
            "default_prompt": "You are helpful.",
        }

        # The provider should be usable as a string
        assert config["provider"] == "openai"
        assert is_valid_provider(config["provider"])


class TestOpenAIIntegration:
    """Integration tests for OpenAI message handling."""

    def test_full_conversation_flow_with_tool_calls(self):
        """Test a complete OpenAI conversation with tool calls."""
        # Simulate user messages going into OpenAI
        input_messages = [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "What's the weather in San Francisco?"},
        ]

        # Normalize input for tracking
        normalized_input = normalize_openai_messages(input_messages)
        assert len(normalized_input) == 2
        assert normalized_input[0]["role"] == "system"
        assert normalized_input[1]["role"] == "user"

        # Simulate OpenAI response with tool call
        openai_response = {
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
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            },
                        ],
                    },
                },
            ],
        }

        # Extract assistant message from response
        assistant_messages = from_openai(openai_response)
        assert len(assistant_messages) == 1
        assert assistant_messages[0]["role"] == "assistant"
        assert len(assistant_messages[0]["tool_calls"]) == 1
        assert assistant_messages[0]["tool_calls"][0]["name"] == "get_weather"

        # Create tool result
        tool_result_msg = tool_result("call_abc123", {"temperature": 72, "conditions": "sunny"})
        assert tool_result_msg["role"] == "tool"
        assert tool_result_msg["tool_call_id"] == "call_abc123"
        assert "72" in tool_result_msg["content"]

        # Final response after tool execution
        final_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The weather in San Francisco is 72°F and sunny!",
                    },
                },
            ],
        }

        final_messages = from_openai(final_response)
        assert len(final_messages) == 1
        assert "72" in final_messages[0]["content"]

    def test_normalize_and_extract_roundtrip(self):
        """Test that normalized messages can be used in a conversation."""
        # Build conversation
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        normalized = normalize_openai_messages(messages)

        # Verify we can build on normalized messages
        all_messages = normalized + [{"role": "assistant", "content": "Hi there!"}]
        assert len(all_messages) == 3


class TestAnthropicIntegration:
    """Integration tests for Anthropic message handling."""

    def test_full_conversation_flow_with_tool_use(self):
        """Test a complete Anthropic conversation with tool use."""
        # Simulate user messages going into Anthropic
        input_messages = [
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ]

        # Normalize input
        normalized_input = normalize_anthropic_messages(input_messages)
        assert len(normalized_input) == 1
        assert normalized_input[0]["role"] == "user"

        # Simulate Anthropic response with tool use
        anthropic_response = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll check the weather for you."},
                {
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "get_weather",
                    "input": {"location": "Tokyo"},
                },
            ],
        }

        # Extract assistant message
        assistant_messages = from_anthropic(anthropic_response)
        assert len(assistant_messages) == 1
        assert assistant_messages[0]["content"] == "I'll check the weather for you."
        assert len(assistant_messages[0]["tool_calls"]) == 1
        assert assistant_messages[0]["tool_calls"][0]["name"] == "get_weather"

        # Create tool result
        tool_result_msg = tool_result("toolu_abc123", {"temperature": 25, "conditions": "cloudy"})
        assert tool_result_msg["role"] == "tool"
        assert tool_result_msg["tool_call_id"] == "toolu_abc123"

    def test_normalize_complex_content_blocks(self):
        """Test normalizing messages with complex content blocks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's my question:"},
                    {"type": "text", "text": "What is 2 + 2?"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The answer is 4."},
                ],
            },
        ]

        normalized = normalize_anthropic_messages(messages)
        assert len(normalized) == 2
        assert "question" in normalized[0]["content"]
        assert "2 + 2" in normalized[0]["content"]
        assert normalized[1]["content"] == "The answer is 4."


class TestGeminiIntegration:
    """Integration tests for Gemini message handling."""

    def test_full_conversation_flow_with_function_call(self):
        """Test a complete Gemini conversation with function calls."""
        # Simulate user contents going into Gemini
        input_contents = [
            {"role": "user", "parts": [{"text": "What time is it in London?"}]},
        ]

        # Normalize input
        normalized_input = normalize_gemini_contents(input_contents)
        assert len(normalized_input) == 1
        assert normalized_input[0]["role"] == "user"

        # Simulate Gemini response with function call
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Let me check the time."},
                            {
                                "functionCall": {
                                    "name": "get_time",
                                    "args": {"timezone": "Europe/London"},
                                },
                            },
                        ],
                    },
                },
            ],
        }

        # Extract assistant message
        assistant_messages = from_gemini(gemini_response)
        assert len(assistant_messages) == 1
        assert assistant_messages[0]["role"] == "assistant"
        assert "check the time" in assistant_messages[0]["content"]
        assert len(assistant_messages[0]["tool_calls"]) == 1
        assert assistant_messages[0]["tool_calls"][0]["name"] == "get_time"

    def test_normalize_with_function_response(self):
        """Test normalizing contents with function responses."""
        contents = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "greet",
                            "response": {"greeting": "Hello there!"},
                        },
                    },
                ],
            },
        ]

        normalized = normalize_gemini_contents(contents)
        # Should have user message and tool response
        assert any(m["role"] == "user" for m in normalized)
        assert any(m["role"] == "tool" for m in normalized)


class TestCrossProviderIntegration:
    """Integration tests for cross-provider scenarios."""

    def test_messages_are_compatible_across_providers(self):
        """Messages from different providers should have compatible structure."""
        # OpenAI response
        openai_msgs = from_openai({
            "choices": [{"message": {"role": "assistant", "content": "Hello from OpenAI"}}]
        })

        # Anthropic response
        anthropic_msgs = from_anthropic({
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello from Anthropic"}]
        })

        # Gemini response
        gemini_msgs = from_gemini({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello from Gemini"}]
                }
            }]
        })

        # All should have the same structure
        for msgs in [openai_msgs, anthropic_msgs, gemini_msgs]:
            assert len(msgs) == 1
            assert msgs[0]["role"] == "assistant"
            assert "content" in msgs[0]
            assert isinstance(msgs[0]["content"], str)

    def test_tool_calls_have_consistent_structure(self):
        """Tool calls from all providers should have consistent structure."""
        # OpenAI with tool call
        openai_msgs = from_openai({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "test_func", "arguments": "{}"}
                    }]
                }
            }]
        })

        # Anthropic with tool use
        anthropic_msgs = from_anthropic({
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "test_func",
                "input": {}
            }]
        })

        # Gemini with function call
        gemini_msgs = from_gemini({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "test_func", "args": {}}}]
                }
            }]
        })

        # All should have tool_calls with consistent structure
        for msgs in [openai_msgs, anthropic_msgs, gemini_msgs]:
            assert len(msgs) == 1
            assert "tool_calls" in msgs[0]
            assert len(msgs[0]["tool_calls"]) == 1
            tc = msgs[0]["tool_calls"][0]
            assert "id" in tc
            assert "name" in tc
            assert tc["name"] == "test_func"
            assert "arguments" in tc


class TestToolHelpersIntegration:
    """Integration tests for tool helper functions."""

    def test_create_and_use_tool_messages(self):
        """Test creating tool call and result messages together."""
        # Create a tool call message
        tool_calls = [
            {"id": "call_1", "name": "search", "arguments": '{"query": "python"}'},
            {"id": "call_2", "name": "calculate", "arguments": '{"expr": "2+2"}'},
        ]
        tool_call_msg = create_tool_call_message(tool_calls, "Let me help you with that.")

        assert tool_call_msg["role"] == "assistant"
        assert tool_call_msg["content"] == "Let me help you with that."
        assert len(tool_call_msg["tool_calls"]) == 2

        # Create tool results
        result1 = tool_result("call_1", ["result1", "result2"])
        result2 = tool_result("call_2", 4)

        assert result1["role"] == "tool"
        assert result1["tool_call_id"] == "call_1"
        assert "result1" in result1["content"]

        assert result2["role"] == "tool"
        assert result2["tool_call_id"] == "call_2"
        assert result2["content"] == "4"

    def test_tool_result_handles_various_types(self):
        """Test that tool_result handles various Python types."""
        # String
        assert tool_result("id1", "hello")["content"] == "hello"

        # Dict
        result = tool_result("id2", {"key": "value"})
        assert '"key"' in result["content"]
        assert '"value"' in result["content"]

        # List
        result = tool_result("id3", [1, 2, 3])
        assert result["content"] == "[1, 2, 3]"

        # None
        assert tool_result("id4", None)["content"] == "null"

        # Boolean
        assert tool_result("id5", True)["content"] == "true"

        # Number
        assert tool_result("id6", 42)["content"] == "42"
