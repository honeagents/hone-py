"""
Unit tests for Hone SDK agent utilities.

Matches TypeScript agent.test.ts - tests agent utility functions.
Note: Parameter validation and evaluation is handled server-side.
"""

import pytest

from hone.agent import (
    get_agent_node,
    format_entity_v2_request,
    update_agent_nodes,
)
from hone.types import GetAgentOptions, AgentNode, EntityNode


class TestGetAgentNode:
    """Tests for get_agent_node function."""

    def test_should_create_simple_agent_node_with_no_parameters(self):
        """Should create a simple agent node with no parameters."""
        options: GetAgentOptions = {
            "default_prompt": "Hello, World!",
        }

        node = get_agent_node("greeting", options)

        assert node["id"] == "greeting"
        assert node["type"] == "agent"
        assert node["params"] == {}
        assert node["prompt"] == "Hello, World!"
        assert node["children"] == []

    def test_should_create_agent_node_with_simple_string_parameters(self):
        """Should create an agent node with simple string parameters."""
        options: GetAgentOptions = {
            "default_prompt": "Hello, {{userName}}!",
            "params": {
                "userName": "Alice",
            },
        }

        node = get_agent_node("greeting", options)

        assert node["params"] == {"userName": "Alice"}
        assert node["prompt"] == "Hello, {{userName}}!"

    def test_should_create_agent_node_with_major_version_and_name(self):
        """Should create an agent node with majorVersion and name."""
        options: GetAgentOptions = {
            "major_version": 1,
            "name": "greeting-agent",
            "default_prompt": "Hello!",
        }

        node = get_agent_node("greeting", options)

        assert node["major_version"] == 1
        assert node["name"] == "greeting-agent"

    def test_should_create_agent_node_with_hyperparameters(self):
        """Should create an agent node with hyperparameters."""
        options: GetAgentOptions = {
            "default_prompt": "Hello!",
            "model": "gpt-4",
            "provider": "openai",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "stop_sequences": ["END", "STOP"],
        }

        node = get_agent_node("greeting", options)

        assert node["model"] == "gpt-4"
        assert node["provider"] == "openai"
        assert node["temperature"] == 0.7
        assert node["max_tokens"] == 1000
        assert node["top_p"] == 0.9
        assert node["frequency_penalty"] == 0.5
        assert node["presence_penalty"] == 0.3
        assert node["stop_sequences"] == ["END", "STOP"]

    def test_should_create_nested_agent_nodes_from_nested_options(self):
        """Should create nested agent nodes from nested options."""
        options: GetAgentOptions = {
            "default_prompt": "Intro: {{introduction}}",
            "params": {
                "introduction": {
                    "default_prompt": "Hello, {{userName}}!",
                    "params": {
                        "userName": "Bob",
                    },
                },
            },
        }

        node = get_agent_node("main", options)

        assert node["id"] == "main"
        assert len(node["children"]) == 1
        assert node["children"][0]["id"] == "introduction"
        assert node["children"][0]["params"] == {"userName": "Bob"}

    def test_should_handle_multiple_nested_agents(self):
        """Should handle multiple nested agents."""
        options: GetAgentOptions = {
            "default_prompt": "{{header}} Content: {{body}} {{footer}}",
            "params": {
                "header": {
                    "default_prompt": "Header text",
                },
                "body": {
                    "default_prompt": "Body with {{detail}}",
                    "params": {
                        "detail": "important info",
                    },
                },
                "footer": {
                    "default_prompt": "Footer",
                },
            },
        }

        node = get_agent_node("document", options)

        assert len(node["children"]) == 3
        assert [c["id"] for c in node["children"]] == ["header", "body", "footer"]

    def test_should_throw_error_for_self_referencing_agents(self):
        """Should throw an error for self-referencing agents."""
        options: GetAgentOptions = {
            "default_prompt": "This is an agent that references {{system-agent}}",
            "params": {
                "system-agent": {
                    "default_prompt": "This should cause an error",
                },
            },
        }

        with pytest.raises(ValueError):
            get_agent_node("system-agent", options)

    def test_should_throw_error_for_circular_agent_references(self):
        """Should throw an error for circular agent references."""
        options: GetAgentOptions = {
            "default_prompt": "A references {{b}}",
            "params": {
                "b": {
                    "default_prompt": "B references {{a}}",
                    "params": {
                        "a": {
                            "default_prompt": "A references {{b}} (circular)",
                        },
                    },
                },
            },
        }

        with pytest.raises(ValueError):
            get_agent_node("a", options)


# Note: Parameter validation and evaluation is handled server-side
# The following functions were removed: insert_params_into_prompt, evaluate_agent, traverse_agent_node, format_entity_request


class TestUpdateAgentNodes:
    """Tests for update_agent_nodes function."""

    def test_should_update_single_node(self):
        """Should update a single node."""
        node: AgentNode = {
            "id": "greeting",
            "type": "agent",
            "params": {},
            "prompt": "Old prompt",
            "children": [],
        }

        updated = update_agent_nodes(node, lambda n: {**n, "prompt": "New prompt"})

        assert updated["prompt"] == "New prompt"
        assert updated["id"] == "greeting"

    def test_should_update_all_nodes_in_nested_structure(self):
        """Should update all nodes in a nested structure."""
        node: AgentNode = {
            "id": "root",
            "type": "agent",
            "params": {},
            "prompt": "root",
            "children": [
                {
                    "id": "child1",
                    "type": "agent",
                    "params": {},
                    "prompt": "child1",
                    "children": [],
                },
                {
                    "id": "child2",
                    "type": "agent",
                    "params": {},
                    "prompt": "child2",
                    "children": [],
                },
            ],
        }

        updated = update_agent_nodes(node, lambda n: {**n, "prompt": f"updated-{n['id']}"})

        assert updated["prompt"] == "updated-root"
        assert updated["children"][0]["prompt"] == "updated-child1"
        assert updated["children"][1]["prompt"] == "updated-child2"

    def test_should_update_deeply_nested_nodes(self):
        """Should update deeply nested nodes."""
        node: AgentNode = {
            "id": "level1",
            "type": "agent",
            "params": {},
            "prompt": "level1",
            "children": [
                {
                    "id": "level2",
                    "type": "agent",
                    "params": {},
                    "prompt": "level2",
                    "children": [
                        {
                            "id": "level3",
                            "type": "agent",
                            "params": {},
                            "prompt": "level3",
                            "children": [],
                        },
                    ],
                },
            ],
        }

        updated = update_agent_nodes(node, lambda n: {**n, "prompt": f"{n['prompt']}-updated"})

        assert updated["prompt"] == "level1-updated"
        assert updated["children"][0]["prompt"] == "level2-updated"
        assert updated["children"][0]["children"][0]["prompt"] == "level3-updated"

    def test_should_preserve_node_structure_while_updating(self):
        """Should preserve node structure while updating."""
        node: AgentNode = {
            "id": "root",
            "type": "agent",
            "name": "root-name",
            "major_version": 1,
            "params": {"key": "value"},
            "prompt": "original",
            "children": [
                {
                    "id": "child",
                    "type": "agent",
                    "params": {},
                    "prompt": "child-original",
                    "children": [],
                },
            ],
        }

        updated = update_agent_nodes(node, lambda n: {**n, "prompt": n["prompt"].upper()})

        assert updated["id"] == "root"
        assert updated["name"] == "root-name"
        assert updated["major_version"] == 1
        assert updated["params"] == {"key": "value"}
        assert updated["prompt"] == "ORIGINAL"
        assert updated["children"][0]["prompt"] == "CHILD-ORIGINAL"

    def test_should_allow_conditional_updates(self):
        """Should allow conditional updates."""
        node: AgentNode = {
            "id": "root",
            "type": "agent",
            "params": {},
            "prompt": "root",
            "children": [
                {
                    "id": "update-me",
                    "type": "agent",
                    "params": {},
                    "prompt": "old",
                    "children": [],
                },
                {
                    "id": "leave-me",
                    "type": "agent",
                    "params": {},
                    "prompt": "unchanged",
                    "children": [],
                },
            ],
        }

        def conditional_update(n):
            if n["id"] == "update-me":
                return {**n, "prompt": "new"}
            return n

        updated = update_agent_nodes(node, conditional_update)

        assert updated["children"][0]["prompt"] == "new"
        assert updated["children"][1]["prompt"] == "unchanged"


class TestFormatEntityV2Request:
    """Tests for format_entity_v2_request function."""

    def test_should_format_simple_agent_node(self):
        """Should format a simple agent node."""
        node: AgentNode = {
            "id": "greeting",
            "type": "agent",
            "name": "greeting-agent",
            "major_version": 1,
            "params": {"userName": "Alice"},
            "prompt": "Hello, {{userName}}!",
            "children": [],
            "model": "gpt-4",
            "provider": "openai",
        }

        request = format_entity_v2_request(node)

        assert request["id"] == "greeting"
        assert request["type"] == "agent"
        assert request["name"] == "greeting-agent"
        assert request["majorVersion"] == 1
        assert request["prompt"] == "Hello, {{userName}}!"
        assert request["params"] == {"userName": "Alice"}
        assert request["data"]["model"] == "gpt-4"
        assert request["data"]["provider"] == "openai"

    def test_should_format_nested_agent_nodes_with_param_values(self):
        """Should format nested agent nodes with param values."""
        node: AgentNode = {
            "id": "main",
            "type": "agent",
            "params": {},
            "prompt": "Intro: {{introduction}}",
            "children": [
                {
                    "id": "introduction",
                    "type": "prompt",
                    "params": {"userName": "Bob"},
                    "prompt": "Hello, {{userName}}!",
                    "children": [],
                },
            ],
            "model": "gpt-4",
            "provider": "openai",
        }

        request = format_entity_v2_request(node)

        assert request["id"] == "main"
        assert request["type"] == "agent"
        assert request["params"] is not None
        assert "introduction" in request["params"]

        # The nested entity should be a full EntityV2Request object
        nested_intro = request["params"]["introduction"]
        assert nested_intro["id"] == "introduction"
        assert nested_intro["type"] == "prompt"
        assert nested_intro["prompt"] == "Hello, {{userName}}!"
        assert nested_intro["params"] == {"userName": "Bob"}

    def test_should_format_agent_node_with_all_hyperparameters_in_data(self):
        """Should format agent node with all hyperparameters in data."""
        node: AgentNode = {
            "id": "greeting",
            "type": "agent",
            "params": {},
            "prompt": "Hello!",
            "children": [],
            "model": "gpt-4",
            "provider": "openai",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "stop_sequences": ["END"],
            "tools": ["search", "calculator"],
        }

        request = format_entity_v2_request(node)

        assert request["data"] is not None
        assert request["data"]["model"] == "gpt-4"
        assert request["data"]["provider"] == "openai"
        assert request["data"]["temperature"] == 0.7
        assert request["data"]["maxTokens"] == 1000
        assert request["data"]["topP"] == 0.9
        assert request["data"]["frequencyPenalty"] == 0.5
        assert request["data"]["presencePenalty"] == 0.3
        assert request["data"]["stopSequences"] == ["END"]
        assert request["data"]["tools"] == ["search", "calculator"]

    def test_should_not_include_data_for_non_agent_types(self):
        """Should not include data for non-agent types."""
        node: EntityNode = {
            "id": "my-prompt",
            "type": "prompt",
            "params": {"value": "test"},
            "prompt": "Value: {{value}}",
            "children": [],
        }

        request = format_entity_v2_request(node)

        assert request["id"] == "my-prompt"
        assert request["type"] == "prompt"
        assert request.get("data") is None

    def test_should_format_deeply_nested_structure(self):
        """Should format deeply nested structure."""
        node: AgentNode = {
            "id": "doc",
            "type": "agent",
            "params": {},
            "prompt": "{{section}}",
            "children": [
                {
                    "id": "section",
                    "type": "prompt",
                    "params": {},
                    "prompt": "{{paragraph}}",
                    "children": [
                        {
                            "id": "paragraph",
                            "type": "prompt",
                            "params": {"text": "content"},
                            "prompt": "{{text}}",
                            "children": [],
                        },
                    ],
                },
            ],
            "model": "gpt-4",
            "provider": "openai",
        }

        request = format_entity_v2_request(node)

        assert request["id"] == "doc"
        section = request["params"]["section"]
        assert section["id"] == "section"
        paragraph = section["params"]["paragraph"]
        assert paragraph["id"] == "paragraph"
        assert paragraph["params"] == {"text": "content"}

    def test_should_mix_string_params_and_nested_entities(self):
        """Should mix string params and nested entities."""
        node: AgentNode = {
            "id": "document",
            "type": "agent",
            "params": {"title": "My Document"},
            "prompt": "{{title}}: {{body}}",
            "children": [
                {
                    "id": "body",
                    "type": "prompt",
                    "params": {"content": "Hello World"},
                    "prompt": "Content: {{content}}",
                    "children": [],
                },
            ],
            "model": "gpt-4",
            "provider": "openai",
        }

        request = format_entity_v2_request(node)

        assert request["params"]["title"] == "My Document"  # String param
        assert isinstance(request["params"]["body"], dict)  # Nested entity
        body = request["params"]["body"]
        assert body["id"] == "body"

    def test_should_omit_params_when_empty(self):
        """Should omit params when empty."""
        node: EntityNode = {
            "id": "simple",
            "type": "prompt",
            "params": {},
            "prompt": "Static text",
            "children": [],
        }

        request = format_entity_v2_request(node)

        assert request.get("params") is None

    def test_should_handle_multiple_children(self):
        """Should handle multiple children."""
        node: AgentNode = {
            "id": "document",
            "type": "agent",
            "params": {},
            "prompt": "{{header}} {{body}} {{footer}}",
            "children": [
                {
                    "id": "header",
                    "type": "prompt",
                    "params": {},
                    "prompt": "HEADER",
                    "children": [],
                },
                {
                    "id": "body",
                    "type": "prompt",
                    "params": {"content": "text"},
                    "prompt": "{{content}}",
                    "children": [],
                },
                {
                    "id": "footer",
                    "type": "prompt",
                    "params": {},
                    "prompt": "FOOTER",
                    "children": [],
                },
            ],
            "model": "gpt-4",
            "provider": "openai",
        }

        request = format_entity_v2_request(node)

        params = request.get("params", {})
        assert len(params) == 3
        assert "header" in params
        assert "body" in params
        assert "footer" in params
