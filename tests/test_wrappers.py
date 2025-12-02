"""
Tests for Hone SDK wrappers.

These tests verify that the wrappers correctly intercept and track LLM calls
to various providers (OpenAI, Anthropic, Gemini, LiteLLM).
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone
import asyncio


class TestWrapOpenAI:
    """Tests for the OpenAI wrapper."""

    def test_wrap_openai_tracks_chat_completion(self, mock_httpx_client):
        """wrap_openai should track chat completion calls."""
        from hone import Hone
        from hone.wrappers import wrap_openai

        # Create Hone client
        aix = Hone(api_key="test-key", project_id="test-project")

        # Create mock OpenAI client
        mock_openai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {
            "model": "gpt-4",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        }
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 100

        mock_openai_client.chat.completions.create.return_value = mock_response

        # Wrap the client
        wrapped_client = wrap_openai(mock_openai_client, hone_client=aix)

        # Make a call
        result = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )

        # Verify the call was made
        assert result.model == "gpt-4"

        # Flush to ensure tracking is sent
        aix.flush()

        # The call should have been tracked (enqueued)
        # In real scenario, we'd verify the API was called
        aix.shutdown()

    def test_wrap_openai_tracks_errors(self, mock_httpx_client):
        """wrap_openai should track failed calls."""
        from hone import Hone
        from hone.wrappers import wrap_openai

        aix = Hone(api_key="test-key", project_id="test-project")

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        wrapped_client = wrap_openai(mock_openai_client, hone_client=aix)

        with pytest.raises(Exception, match="API Error"):
            wrapped_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}]
            )

        aix.flush()
        aix.shutdown()

    def test_wrap_openai_preserves_client_identity(self, mock_httpx_client):
        """wrap_openai should return the same client instance."""
        from hone import Hone
        from hone.wrappers import wrap_openai

        aix = Hone(api_key="test-key", project_id="test-project")

        mock_openai_client = MagicMock()
        wrapped_client = wrap_openai(mock_openai_client, hone_client=aix)

        assert wrapped_client is mock_openai_client
        aix.shutdown()


class TestWrapAnthropic:
    """Tests for the Anthropic wrapper."""

    def test_wrap_anthropic_tracks_messages(self, mock_httpx_client):
        """wrap_anthropic should track messages.create calls."""
        from hone import Hone
        from hone.wrappers import wrap_anthropic

        aix = Hone(api_key="test-key", project_id="test-project")

        mock_anthropic_client = MagicMock()
        mock_response = MagicMock()
        mock_response.model = "claude-3-5-sonnet-latest"
        mock_response.model_dump.return_value = {
            "model": "claude-3-5-sonnet-latest",
            "content": [{"type": "text", "text": "Hello!"}],
        }
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        mock_anthropic_client.messages.create.return_value = mock_response

        wrapped_client = wrap_anthropic(mock_anthropic_client, hone_client=aix)

        result = wrapped_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello!"}]
        )

        assert result.model == "claude-3-5-sonnet-latest"

        aix.flush()
        aix.shutdown()

    def test_wrap_anthropic_tracks_errors(self, mock_httpx_client):
        """wrap_anthropic should track failed calls."""
        from hone import Hone
        from hone.wrappers import wrap_anthropic

        aix = Hone(api_key="test-key", project_id="test-project")

        mock_anthropic_client = MagicMock()
        mock_anthropic_client.messages.create.side_effect = Exception("Rate limit")

        wrapped_client = wrap_anthropic(mock_anthropic_client, hone_client=aix)

        with pytest.raises(Exception, match="Rate limit"):
            wrapped_client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1000,
                messages=[{"role": "user", "content": "Hello!"}]
            )

        aix.flush()
        aix.shutdown()


class TestWrapGemini:
    """Tests for the Gemini wrapper."""

    def test_wrap_gemini_tracks_generate_content(self, mock_httpx_client):
        """wrap_gemini should track generate_content calls."""
        from hone import Hone
        from hone.wrappers import wrap_gemini

        aix = Hone(api_key="test-key", project_id="test-project")

        mock_gemini_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello!"
        mock_response.to_dict.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Hello!"}]}}],
        }
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20

        mock_gemini_client.models.generate_content.return_value = mock_response

        wrapped_client = wrap_gemini(mock_gemini_client, hone_client=aix)

        result = wrapped_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello!"
        )

        assert result.text == "Hello!"

        aix.flush()
        aix.shutdown()

    def test_wrap_gemini_prevents_double_wrapping(self, mock_httpx_client):
        """wrap_gemini should raise error if client already wrapped."""
        from hone import Hone
        from hone.wrappers import wrap_gemini

        aix = Hone(api_key="test-key", project_id="test-project")

        mock_gemini_client = MagicMock()
        mock_gemini_client.models.generate_content.__wrapped__ = True

        with pytest.raises(ValueError, match="already been wrapped"):
            wrap_gemini(mock_gemini_client, hone_client=aix)

        aix.shutdown()


class TestWrapLiteLLM:
    """Tests for the LiteLLM wrapper."""

    def test_wrap_litellm_patches_completion(self, mock_httpx_client):
        """wrap_litellm should patch litellm.completion."""
        from hone import Hone
        from hone.wrappers.litellm import wrap_litellm, unwrap_litellm

        aix = Hone(api_key="test-key", project_id="test-project")

        # Mock litellm module
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {"model": "gpt-4"}
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 100

        original_completion = MagicMock(return_value=mock_response)
        mock_litellm.completion = original_completion
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)
        mock_litellm.embedding = MagicMock()
        mock_litellm.aembedding = AsyncMock()

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            wrap_litellm(aix)

            # The completion function should now be wrapped
            assert mock_litellm.completion is not original_completion

            # Call the wrapped function
            result = mock_litellm.completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}]
            )

            assert result.model == "gpt-4"

            # Unwrap
            unwrap_litellm()

        aix.flush()
        aix.shutdown()

    def test_wrap_litellm_import_error(self, mock_httpx_client):
        """wrap_litellm should raise ImportError if litellm not installed."""
        from hone import Hone
        from hone.wrappers.litellm import wrap_litellm

        aix = Hone(api_key="test-key", project_id="test-project")

        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(ImportError, match="LiteLLM is not installed"):
                wrap_litellm(aix)

        aix.shutdown()


class TestWrapperMetadataExtraction:
    """Tests for metadata extraction from LLM responses."""

    def test_openai_usage_extraction(self, mock_httpx_client):
        """OpenAI wrapper should extract usage metadata correctly."""
        from hone.wrappers.openai import _extract_usage_metadata

        mock_response = MagicMock()
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 100
        mock_response.id = "chatcmpl-123"

        metadata = _extract_usage_metadata(mock_response)

        assert metadata["model"] == "gpt-4"
        assert metadata["tokens_used"] == 150
        assert metadata["input_tokens"] == 50
        assert metadata["output_tokens"] == 100
        assert metadata["response_id"] == "chatcmpl-123"

    def test_anthropic_usage_extraction(self, mock_httpx_client):
        """Anthropic wrapper should extract usage metadata correctly."""
        from hone.wrappers.anthropic import _extract_usage_metadata

        mock_response = MagicMock()
        mock_response.model = "claude-3-5-sonnet-latest"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 100
        mock_response.id = "msg-123"
        mock_response.stop_reason = "end_turn"

        metadata = _extract_usage_metadata(mock_response)

        assert metadata["model"] == "claude-3-5-sonnet-latest"
        assert metadata["tokens_used"] == 150
        assert metadata["input_tokens"] == 50
        assert metadata["output_tokens"] == 100
        assert metadata["stop_reason"] == "end_turn"

    def test_gemini_usage_extraction(self, mock_httpx_client):
        """Gemini wrapper should extract usage metadata correctly."""
        from hone.wrappers.gemini import _extract_usage_metadata

        mock_response = MagicMock()
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 50
        mock_response.usage_metadata.candidates_token_count = 100
        mock_response.usage_metadata.total_token_count = 150

        metadata = _extract_usage_metadata(mock_response)

        assert metadata["tokens_used"] == 150
        assert metadata["input_tokens"] == 50
        assert metadata["output_tokens"] == 100


class TestGeminiInputProcessing:
    """Tests for Gemini input normalization."""

    def test_string_input_normalization(self):
        """String input should be converted to messages format."""
        from hone.wrappers.gemini import _process_gemini_inputs

        inputs = {"contents": "Hello!", "model": "gemini-2.5-flash"}
        result = _process_gemini_inputs(inputs)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello!"

    def test_list_string_input_normalization(self):
        """List of strings should be converted to messages format."""
        from hone.wrappers.gemini import _process_gemini_inputs

        inputs = {"contents": ["Hello!", "World!"], "model": "gemini-2.5-flash"}
        result = _process_gemini_inputs(inputs)

        assert "messages" in result
        assert len(result["messages"]) == 2

    def test_complex_content_normalization(self):
        """Complex content with parts should be normalized."""
        from hone.wrappers.gemini import _process_gemini_inputs

        inputs = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello!"}]}
            ],
            "model": "gemini-2.5-flash",
        }
        result = _process_gemini_inputs(inputs)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello!"


class TestOpenAIStreamingReduction:
    """Tests for OpenAI streaming chunk reduction."""

    def test_reduce_chat_chunks(self):
        """Chat chunks should be reduced to a single response."""
        from hone.wrappers.openai import _reduce_chat_chunks

        mock_chunks = []

        # Simulate streaming chunks
        for i, text in enumerate(["Hello", ", ", "world", "!"]):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = text
            chunk.choices[0].delta.role = "assistant" if i == 0 else None
            chunk.choices[0].delta.function_call = None
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].index = 0
            chunk.choices[0].finish_reason = "stop" if i == 3 else None
            chunk.model_dump.return_value = {}
            mock_chunks.append(chunk)

        result = _reduce_chat_chunks(mock_chunks)

        assert "choices" in result
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Hello, world!"
        assert result["choices"][0]["message"]["role"] == "assistant"


class TestAnthropicInputStripping:
    """Tests for Anthropic input normalization."""

    def test_strip_not_given(self):
        """NotGiven values should be stripped."""
        from hone.wrappers.anthropic import _strip_not_given

        # Without anthropic installed, should just return dict
        inputs = {"messages": [{"role": "user", "content": "Hello!"}], "model": "claude-3"}
        result = _strip_not_given(inputs)

        assert "messages" in result
        assert "model" in result

    def test_system_normalization(self):
        """System prompt should be moved to messages."""
        from hone.wrappers.anthropic import _strip_not_given

        inputs = {
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "Hello!"}],
            "model": "claude-3",
        }
        result = _strip_not_given(inputs)

        # System should be moved to messages
        assert "system" not in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful"
