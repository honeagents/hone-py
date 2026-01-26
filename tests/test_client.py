"""
Unit tests for Hone SDK client.

Exact replica of TypeScript client.test.ts - tests the Hone client class.
"""

import json
import os

import httpx
import pytest
import respx

from hone.client import Hone, create_hone_client, DEFAULT_BASE_URL, SUPABASE_ANON_KEY
from hone.types import HoneConfig, Message, AgentResponse


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
    """Tests for Hone.agent method."""

    @pytest.fixture
    def mock_api_key(self):
        return "test-api-key"

    @pytest.fixture
    def client(self, mock_api_key):
        return Hone({"api_key": mock_api_key})

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_fetch_agent_successfully_and_return_evaluated_result(self, client):
        """Should fetch agent successfully and return evaluated result."""
        mock_response: AgentResponse = {
            "greeting": {"prompt": "Hello, {{userName}}! Welcome."},
        }

        respx.post(f"{DEFAULT_BASE_URL}/sync_agents").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = await client.agent("greeting", {
            "default_prompt": "Hi, {{userName}}!",
            "params": {
                "userName": "Alice",
            },
        })

        assert result == "Hello, Alice! Welcome."

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_use_fallback_agent_when_api_call_fails(self, client, capsys):
        """Should use fallback agent when API call fails."""
        respx.post(f"{DEFAULT_BASE_URL}/sync_agents").mock(
            side_effect=httpx.RequestError("Network error")
        )

        result = await client.agent("greeting", {
            "default_prompt": "Hi, {{userName}}!",
            "params": {
                "userName": "Bob",
            },
        })

        assert result == "Hi, Bob!"
        captured = capsys.readouterr()
        assert "Error fetching agent, using fallback:" in captured.out

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_handle_nested_agents(self, client):
        """Should handle nested agents."""
        mock_response: AgentResponse = {
            "main": {"prompt": "Welcome: {{intro}}"},
            "intro": {"prompt": "Hello, {{userName}}!"},
        }

        respx.post(f"{DEFAULT_BASE_URL}/sync_agents").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = await client.agent("main", {
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

        assert result == "Welcome: Hello, Charlie!"

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_handle_agent_with_no_parameters(self, client):
        """Should handle agent with no parameters."""
        mock_response: AgentResponse = {
            "static": {"prompt": "This is a static prompt"},
        }

        respx.post(f"{DEFAULT_BASE_URL}/sync_agents").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        result = await client.agent("static", {
            "default_prompt": "Fallback static",
        })

        assert result == "This is a static prompt"

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_use_fallback_when_api_returns_error_status(self, client, capsys):
        """Should use fallback when API returns error status."""
        respx.post(f"{DEFAULT_BASE_URL}/sync_agents").mock(
            return_value=httpx.Response(404, json={"message": "Agent not found"})
        )

        result = await client.agent("missing", {
            "default_prompt": "Fallback prompt",
        })

        assert result == "Fallback prompt"

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_handle_major_version_and_name_in_agent_options(self, client):
        """Should handle majorVersion and name in agent options."""
        mock_response = {
            "greeting-v2": {"prompt": "Hello v2!"},
        }

        route = respx.post(f"{DEFAULT_BASE_URL}/sync_agents").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        await client.agent("greeting-v2", {
            "major_version": 2,
            "name": "greeting",
            "default_prompt": "Hello v1!",
        })

        # Verify request body
        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["agents"]["map"]["greeting-v2"]["majorVersion"] == 2
        assert body["agents"]["map"]["greeting-v2"]["name"] == "greeting"

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_send_correct_request_format_to_api(self, client):
        """Should send correct request format to API."""
        route = respx.post(f"{DEFAULT_BASE_URL}/sync_agents").mock(
            return_value=httpx.Response(200, json={})
        )

        await client.agent("test", {
            "default_prompt": "Test {{param1}}",
            "params": {
                "param1": "value1",
            },
        })

        request = route.calls.last.request
        body = json.loads(request.content)

        assert body["agents"]["rootId"] == "test"
        assert "map" in body["agents"]
        assert body["agents"]["map"]["test"] == {
            "id": "test",
            "name": None,
            "majorVersion": None,
            "prompt": "Test {{param1}}",
            "paramKeys": ["param1"],
            "childrenIds": [],
            "model": None,
            "temperature": None,
            "maxTokens": None,
            "topP": None,
            "frequencyPenalty": None,
            "presencePenalty": None,
            "stopSequences": None,
        }

    @respx.mock
    @pytest.mark.asyncio
    async def test_should_send_hyperparameters_in_request(self, client):
        """Should send hyperparameters in request."""
        route = respx.post(f"{DEFAULT_BASE_URL}/sync_agents").mock(
            return_value=httpx.Response(200, json={})
        )

        await client.agent("test", {
            "default_prompt": "Test prompt",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "stop_sequences": ["END"],
        })

        request = route.calls.last.request
        body = json.loads(request.content)

        assert body["agents"]["map"]["test"]["model"] == "gpt-4"
        assert body["agents"]["map"]["test"]["temperature"] == 0.7
        assert body["agents"]["map"]["test"]["maxTokens"] == 1000
        assert body["agents"]["map"]["test"]["topP"] == 0.9
        assert body["agents"]["map"]["test"]["frequencyPenalty"] == 0.5
        assert body["agents"]["map"]["test"]["presencePenalty"] == 0.3
        assert body["agents"]["map"]["test"]["stopSequences"] == ["END"]


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
        assert headers["Authorization"] == f"Bearer {SUPABASE_ANON_KEY}"
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
