"""
Unit tests for Hone SDK client.

Matches TypeScript client.test.ts - tests the Hone client class with V2 API.
"""

import json
import os

import httpx
import pytest
import respx

from hone.client import Hone, create_hone_client, DEFAULT_BASE_URL
from hone.types import HoneConfig, Message, EntityV2Response


class TestHoneConstructor:
    """Tests for Hone constructor."""

    def test_should_initialize_with_required_config(self):
        """Should initialize with required config."""
        config: HoneConfig = {
            "api_key": "my-key",
        }

        client = Hone(config)
        assert isinstance(client, Hone)

    def test_should_use_default_base_url_when_not_provided(self):
        """Should use default base URL when not provided."""
        client = Hone({"api_key": "key"})
        assert isinstance(client, Hone)

    def test_should_use_custom_base_url_when_provided(self):
        """Should use custom base URL when provided."""
        client = Hone({
            "api_key": "key",
            "base_url": "https://custom.api.com",
        })
        assert isinstance(client, Hone)

    def test_should_use_custom_timeout_when_provided(self):
        """Should use custom timeout when provided."""
        client = Hone({
            "api_key": "key",
            "timeout": 5000,
        })
        assert isinstance(client, Hone)

    def test_should_prioritize_hone_api_url_env_var_over_config_base_url(self):
        """Should prioritize HONE_API_URL env var over config baseUrl."""
        original_env = os.environ.get("HONE_API_URL")
        os.environ["HONE_API_URL"] = "https://env.api.com"

        try:
            client = Hone({
                "api_key": "key",
                "base_url": "https://config.api.com",
            })
            assert isinstance(client, Hone)
        finally:
            if original_env is not None:
                os.environ["HONE_API_URL"] = original_env
            else:
                del os.environ["HONE_API_URL"]


class TestHoneAgent:
    """Tests for Hone.agent method using V2 API."""

    @pytest.fixture
    def mock_api_key(self):
        return "test-api-key"

    @pytest.fixture
    def client(self, mock_api_key):
        return Hone({"api_key": mock_api_key})

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_fetch_agent_successfully_and_return_evaluated_result(self, client):
        """Should fetch agent successfully and return evaluated result with hyperparameters."""
        mock_response: EntityV2Response = {
            "evaluatedPrompt": "Hello, Alice! Welcome.",
            "template": "Hello, {{userName}}! Welcome.",
            "type": "agent",
            "data": {
                "model": "gpt-4",
                "provider": "openai",
                "temperature": 0.7,
                "maxTokens": 1000,
                "topP": 0.9,
                "frequencyPenalty": 0.1,
                "presencePenalty": 0.2,
                "stopSequences": ["END"],
                "tools": [],
            },
        }

        respx.post(f"{DEFAULT_BASE_URL}/v2/entities").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = await client.agent("greeting", {
            "model": "gpt-4",
            "provider": "openai",
            "default_prompt": "Hi, {{userName}}!",
            "params": {
                "userName": "Alice",
            },
        })

        assert result["system_prompt"] == "Hello, Alice! Welcome."
        assert result["model"] == "gpt-4"
        assert result["provider"] == "openai"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_throw_error_when_api_call_fails_no_fallback(self, client):
        """Should throw error when API call fails (no fallback)."""
        respx.post(f"{DEFAULT_BASE_URL}/v2/entities").mock(
            side_effect=httpx.RequestError("Network error")
        )

        with pytest.raises(Exception, match="Network error"):
            await client.agent("greeting", {
                "model": "gpt-4",
                "provider": "openai",
                "default_prompt": "Hi, {{userName}}!",
                "params": {
                    "userName": "Bob",
                },
            })

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_handle_nested_agents_v2_evaluates_server_side(self, client):
        """Should handle nested agents (V2 evaluates server-side)."""
        # V2 API returns already evaluated prompt - nesting is handled server-side
        mock_response: EntityV2Response = {
            "evaluatedPrompt": "Welcome: Hello, Charlie!",
            "template": "Welcome: {{intro}}",
            "type": "agent",
            "data": {
                "model": "gpt-4",
                "provider": "openai",
                "temperature": None,
                "maxTokens": None,
                "topP": None,
                "frequencyPenalty": None,
                "presencePenalty": None,
                "stopSequences": [],
                "tools": [],
            },
        }

        respx.post(f"{DEFAULT_BASE_URL}/v2/entities").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = await client.agent("main", {
            "model": "gpt-4",
            "provider": "openai",
            "default_prompt": "Fallback: {{intro}}",
            "params": {
                "intro": {
                    "default_prompt": "Hi, {{userName}}!",
                    "params": {
                        "userName": "Charlie",
                    },
                },
            },
        })

        assert result["system_prompt"] == "Welcome: Hello, Charlie!"

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_handle_agent_with_no_parameters(self, client):
        """Should handle agent with no parameters."""
        mock_response: EntityV2Response = {
            "evaluatedPrompt": "This is a static prompt",
            "template": "This is a static prompt",
            "type": "agent",
            "data": {
                "model": "gpt-4",
                "provider": "openai",
                "temperature": None,
                "maxTokens": None,
                "topP": None,
                "frequencyPenalty": None,
                "presencePenalty": None,
                "stopSequences": [],
                "tools": [],
            },
        }

        respx.post(f"{DEFAULT_BASE_URL}/v2/entities").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = await client.agent("static", {
            "model": "gpt-4",
            "provider": "openai",
            "default_prompt": "Fallback static",
        })

        assert result["system_prompt"] == "This is a static prompt"

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_throw_error_when_api_returns_error_status_no_fallback(self, client):
        """Should throw error when API returns error status (no fallback)."""
        respx.post(f"{DEFAULT_BASE_URL}/v2/entities").mock(
            return_value=httpx.Response(404, json={"error": "Agent not found"})
        )

        with pytest.raises(Exception, match="Hone API error \\(404\\)"):
            await client.agent("missing", {
                "model": "gpt-4",
                "provider": "openai",
                "default_prompt": "Fallback prompt",
            })

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_handle_major_version_and_name_in_agent_options(self, client):
        """Should handle majorVersion and name in agent options."""
        mock_response: EntityV2Response = {
            "evaluatedPrompt": "Hello v2!",
            "template": "Hello v2!",
            "type": "agent",
            "data": {
                "model": None,
                "provider": None,
                "temperature": None,
                "maxTokens": None,
                "topP": None,
                "frequencyPenalty": None,
                "presencePenalty": None,
                "stopSequences": [],
                "tools": [],
            },
        }

        route = respx.post(f"{DEFAULT_BASE_URL}/v2/entities").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        await client.agent("greeting-v2", {
            "major_version": 2,
            "name": "greeting",
            "default_prompt": "Hello v1!",
        })

        # Verify V2 request format
        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["id"] == "greeting-v2"
        assert body["majorVersion"] == 2
        assert body["name"] == "greeting"

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_send_correct_request_format_to_api(self, client):
        """Should send correct V2 request format to API."""
        mock_response: EntityV2Response = {
            "evaluatedPrompt": "Test value1",
            "template": "Test {{param1}}",
            "type": "agent",
            "data": {
                "model": None,
                "provider": None,
                "temperature": None,
                "maxTokens": None,
                "topP": None,
                "frequencyPenalty": None,
                "presencePenalty": None,
                "stopSequences": [],
                "tools": [],
            },
        }

        route = respx.post(f"{DEFAULT_BASE_URL}/v2/entities").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        await client.agent("test", {
            "default_prompt": "Test {{param1}}",
            "params": {
                "param1": "value1",
            },
        })

        request = route.calls.last.request
        body = json.loads(request.content)

        # V2 uses flat structure with params containing values
        assert body["id"] == "test"
        assert body["type"] == "agent"
        assert body["prompt"] == "Test {{param1}}"
        assert body["params"] == {"param1": "value1"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_send_hyperparameters_in_request(self, client):
        """Should send hyperparameters in V2 request data field."""
        mock_response: EntityV2Response = {
            "evaluatedPrompt": "Test prompt",
            "template": "Test prompt",
            "type": "agent",
            "data": {
                "model": "gpt-4",
                "provider": "openai",
                "temperature": 0.7,
                "maxTokens": 1000,
                "topP": 0.9,
                "frequencyPenalty": 0.5,
                "presencePenalty": 0.3,
                "stopSequences": ["END"],
                "tools": [],
            },
        }

        route = respx.post(f"{DEFAULT_BASE_URL}/v2/entities").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        await client.agent("test", {
            "default_prompt": "Test prompt",
            "model": "gpt-4",
            "provider": "openai",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "stop_sequences": ["END"],
        })

        request = route.calls.last.request
        body = json.loads(request.content)

        # V2 puts hyperparameters in data field
        assert body["data"]["model"] == "gpt-4"
        assert body["data"]["provider"] == "openai"
        assert body["data"]["temperature"] == 0.7
        assert body["data"]["maxTokens"] == 1000
        assert body["data"]["topP"] == 0.9
        assert body["data"]["frequencyPenalty"] == 0.5
        assert body["data"]["presencePenalty"] == 0.3
        assert body["data"]["stopSequences"] == ["END"]


class TestHoneTrack:
    """Tests for Hone.track method."""

    @pytest.fixture
    def mock_api_key(self):
        return "test-api-key"

    @pytest.fixture
    def client(self, mock_api_key):
        return Hone({"api_key": mock_api_key})

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_track_conversation_successfully(self, client):
        """Should track conversation successfully."""
        route = respx.post(f"{DEFAULT_BASE_URL}/insert_runs").mock(
            return_value=httpx.Response(200, json={})
        )

        messages: list[Message] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        await client.track("test-conversation", messages, {"session_id": "session-xyz"})

        assert route.called
        request = route.calls.last.request
        body = json.loads(request.content)

        assert body["id"] == "test-conversation"
        assert body["messages"] == messages
        assert "timestamp" in body

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_track_with_session_id_when_provided(self, client):
        """Should track with session ID when provided."""
        route = respx.post(f"{DEFAULT_BASE_URL}/insert_runs").mock(
            return_value=httpx.Response(200, json={})
        )

        messages: list[Message] = [{"role": "user", "content": "Hello"}]

        await client.track("test", messages, {"session_id": "session-123"})

        request = route.calls.last.request
        body = json.loads(request.content)

        assert body["sessionId"] == "session-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_track_with_empty_messages_array(self, client):
        """Should track with empty messages array."""
        route = respx.post(f"{DEFAULT_BASE_URL}/insert_runs").mock(
            return_value=httpx.Response(200, json={})
        )

        await client.track("test", [], {"session_id": "session-empty"})

        assert route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_track_with_multiple_message_types(self, client):
        """Should track with multiple message types."""
        route = respx.post(f"{DEFAULT_BASE_URL}/insert_runs").mock(
            return_value=httpx.Response(200, json={})
        )

        messages: list[Message] = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I'll check that for you."},
            {"role": "user", "content": "Thanks!"},
        ]

        await client.track("multi-turn", messages, {"session_id": "session-multi"})

        request = route.calls.last.request
        body = json.loads(request.content)

        assert body["messages"] == messages

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_throw_error_when_track_api_call_fails(self, client):
        """Should throw error when track API call fails."""
        respx.post(f"{DEFAULT_BASE_URL}/insert_runs").mock(
            return_value=httpx.Response(500, json={"message": "Server error"})
        )

        messages: list[Message] = [{"role": "user", "content": "Test"}]

        with pytest.raises(Exception, match="Hone API error \\(500\\): Server error"):
            await client.track("test", messages, {"session_id": "session-123"})


class TestHoneErrorHandling:
    """Tests for Hone error handling."""

    @pytest.fixture
    def mock_api_key(self):
        return "test-api-key"

    @pytest.fixture
    def client(self, mock_api_key):
        return Hone({"api_key": mock_api_key})

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_throw_error_with_message_from_api_response(self, client):
        """Should throw error with message from API response."""
        respx.post(f"{DEFAULT_BASE_URL}/insert_runs").mock(
            return_value=httpx.Response(401, json={"message": "Invalid API key"})
        )

        with pytest.raises(Exception, match="Hone API error \\(401\\): Invalid API key"):
            await client.track("test", [{"role": "user", "content": "Hi"}], {"session_id": "session-123"})

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_use_status_text_when_error_message_not_in_response(self, client):
        """Should use status text when error message not in response."""
        respx.post(f"{DEFAULT_BASE_URL}/insert_runs").mock(
            return_value=httpx.Response(403, json={})
        )

        with pytest.raises(Exception, match="Hone API error \\(403\\)"):
            await client.track("test", [{"role": "user", "content": "Hi"}], {"session_id": "session-123"})

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_handle_json_parse_error_in_error_response(self, client):
        """Should handle JSON parse error in error response."""
        respx.post(f"{DEFAULT_BASE_URL}/insert_runs").mock(
            return_value=httpx.Response(500, content=b"Not valid JSON")
        )

        with pytest.raises(Exception, match="Hone API error \\(500\\)"):
            await client.track("test", [{"role": "user", "content": "Hi"}], {"session_id": "session-123"})

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_include_user_agent_header_in_requests(self, client):
        """Should include User-Agent header in requests."""
        route = respx.post(f"{DEFAULT_BASE_URL}/insert_runs").mock(
            return_value=httpx.Response(200, json={})
        )

        await client.track("test", [{"role": "user", "content": "Test"}], {"session_id": "session-123"})

        request = route.calls.last.request
        assert request.headers["User-Agent"] == "hone-sdk-python/0.1.0"


class TestCreateHoneClientFactory:
    """Tests for create_hone_client factory function."""

    def test_should_create_hone_client_instance(self):
        """Should create a Hone client instance."""
        config: HoneConfig = {
            "api_key": "test-key",
        }

        client = create_hone_client(config)

        assert isinstance(client, Hone)
        assert hasattr(client, "agent")
        assert hasattr(client, "track")

    def test_should_create_client_with_custom_config(self):
        """Should create client with custom config."""
        config: HoneConfig = {
            "api_key": "test-key",
            "base_url": "https://custom.com",
            "timeout": 5000,
        }

        client = create_hone_client(config)

        assert isinstance(client, Hone)


class TestHoneRequestHeaders:
    """Tests for request headers."""

    @pytest.fixture
    def mock_api_key(self):
        return "test-api-key"

    @pytest.fixture
    def client(self, mock_api_key):
        return Hone({"api_key": mock_api_key})

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_include_all_required_headers(self, client, mock_api_key):
        """Should include all required headers."""
        route = respx.post(f"{DEFAULT_BASE_URL}/insert_runs").mock(
            return_value=httpx.Response(200, json={})
        )

        await client.track("test", [], {"session_id": "session-123"})

        request = route.calls.last.request
        headers = request.headers

        assert headers["Content-Type"] == "application/json"
        assert headers["x-api-key"] == mock_api_key
        assert headers["User-Agent"] == "hone-sdk-python/0.1.0"


class TestHoneBaseUrlHandling:
    """Tests for base URL handling."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_construct_correct_url_with_default_base_url(self):
        """Should construct correct URL with default base URL."""
        client = Hone({"api_key": "key"})

        route = respx.post(f"{DEFAULT_BASE_URL}/insert_runs").mock(
            return_value=httpx.Response(200, json={})
        )

        await client.track("test", [], {"session_id": "session-123"})

        assert route.called
        request = route.calls.last.request
        assert str(request.url) == f"{DEFAULT_BASE_URL}/insert_runs"

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_construct_correct_url_with_custom_base_url(self):
        """Should construct correct URL with custom base URL."""
        custom_client = Hone({
            "api_key": "key",
            "base_url": "https://custom.api.com",
        })

        route = respx.post("https://custom.api.com/insert_runs").mock(
            return_value=httpx.Response(200, json={})
        )

        await custom_client.track("test", [], {"session_id": "session-123"})

        assert route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_handle_base_url_without_trailing_slash(self):
        """Should handle base URL without trailing slash."""
        custom_client = Hone({
            "api_key": "key",
            "base_url": "https://api.example.com/v1",
        })

        route = respx.post("https://api.example.com/v1/insert_runs").mock(
            return_value=httpx.Response(200, json={})
        )

        await custom_client.track("test", [], {"session_id": "session-123"})

        assert route.called
