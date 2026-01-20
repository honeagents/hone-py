"""
Unit tests for Hone SDK agent utilities.

Exact replica of TypeScript agent.test.ts - tests all agent utility functions.
"""

import pytest

from hone.agent import (
    get_agent_node,
    evaluate_agent,
    insert_params_into_prompt,
    traverse_agent_node,
    format_agent_request,
    update_agent_nodes,
)
from hone.types import GetAgentOptions, AgentNode


class TestGetAgentNode:
    """Tests for get_agent_node function."""

    def test_should_create_simple_agent_node_with_no_parameters(self):
        """Should create a simple agent node with no parameters."""
        options: GetAgentOptions = {
            "default_prompt": "Hello, World!",
        }

        node = get_agent_node("greeting", options)

        assert node == {
            "id": "greeting",
            "major_version": None,
            "name": None,
            "params": {},
            "prompt": "Hello, World!",
            "children": [],
            "model": None,
            "temperature": None,
            "max_tokens": None,
            "top_p": None,
            "frequency_penalty": None,
            "presence_penalty": None,
            "stop_sequences": None,
        }

    def test_should_create_agent_node_with_simple_string_parameters(self):
        """Should create an agent node with simple string parameters."""
        options: GetAgentOptions = {
            "default_prompt": "Hello, {{userName}}!",
            "params": {
                "userName": "Alice",
            },
        }

        node = get_agent_node("greeting", options)

        assert node == {
            "id": "greeting",
            "major_version": None,
            "name": None,
            "params": {
                "userName": "Alice",
            },
            "prompt": "Hello, {{userName}}!",
            "children": [],
            "model": None,
            "temperature": None,
            "max_tokens": None,
            "top_p": None,
            "frequency_penalty": None,
            "presence_penalty": None,
            "stop_sequences": None,
        }

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
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "stop_sequences": ["END", "STOP"],
        }

        node = get_agent_node("greeting", options)

        assert node["model"] == "gpt-4"
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

    def test_should_throw_error_when_agent_has_placeholders_without_matching_parameters(self):
        """Should throw an error when agent has placeholders without matching parameters."""
        options: GetAgentOptions = {
            "default_prompt": "Hello {{name}}, your role is {{role}}",
            "params": {
                "name": "Alice",
                # 'role' is missing
            },
        }

        node = get_agent_node("greeting", options)

        # Should throw when evaluating because 'role' placeholder has no value
        with pytest.raises(ValueError, match=r"(?i)missing parameter.*role"):
            evaluate_agent(node)

    def test_should_throw_error_listing_all_missing_parameters(self):
        """Should throw an error listing all missing parameters."""
        options: GetAgentOptions = {
            "default_prompt": "{{greeting}} {{name}}, you are {{role}} in {{location}}",
            "params": {
                "name": "Bob",
                # Missing: greeting, role, location
            },
        }

        node = get_agent_node("test", options)

        with pytest.raises(ValueError, match=r"(?i)missing parameter"):
            evaluate_agent(node)


class TestInsertParamsIntoPrompt:
    """Tests for insert_params_into_prompt function."""

    def test_should_replace_single_placeholder(self):
        """Should replace single placeholder."""
        result = insert_params_into_prompt("Hello, {{name}}!", {"name": "Alice"})
        assert result == "Hello, Alice!"

    def test_should_replace_multiple_placeholders(self):
        """Should replace multiple placeholders."""
        result = insert_params_into_prompt(
            "{{greeting}} {{name}}, {{action}}!",
            {
                "greeting": "Hello",
                "name": "Bob",
                "action": "welcome",
            },
        )
        assert result == "Hello Bob, welcome!"

    def test_should_replace_multiple_occurrences_of_same_placeholder(self):
        """Should replace multiple occurrences of the same placeholder."""
        result = insert_params_into_prompt(
            "{{name}} said: 'Hello {{name}}'",
            {"name": "Charlie"},
        )
        assert result == "Charlie said: 'Hello Charlie'"

    def test_should_return_original_prompt_when_no_params_provided(self):
        """Should return original prompt when no params provided."""
        prompt = "Hello, {{name}}!"
        result = insert_params_into_prompt(prompt)
        assert result == prompt

    def test_should_handle_empty_params_object(self):
        """Should handle empty params object."""
        prompt = "Hello, {{name}}!"
        result = insert_params_into_prompt(prompt, {})
        assert result == prompt

    def test_should_not_replace_placeholders_with_no_matching_params(self):
        """Should not replace placeholders with no matching params."""
        result = insert_params_into_prompt("Hello, {{name}}!", {"greeting": "Hi"})
        assert result == "Hello, {{name}}!"

    def test_should_handle_prompts_with_no_placeholders(self):
        """Should handle prompts with no placeholders."""
        result = insert_params_into_prompt("Hello, World!", {"name": "Alice"})
        assert result == "Hello, World!"

    def test_should_handle_special_characters_in_values(self):
        """Should handle special characters in values."""
        result = insert_params_into_prompt(
            "Message: {{text}}",
            {"text": "Special chars: $, *, (, )"},
        )
        assert result == "Message: Special chars: $, *, (, )"


class TestEvaluateAgent:
    """Tests for evaluate_agent function."""

    def test_should_evaluate_simple_agent_with_params(self):
        """Should evaluate a simple agent with params."""
        node: AgentNode = {
            "id": "greeting",
            "name": None,
            "major_version": None,
            "params": {"userName": "Alice"},
            "prompt": "Hello, {{userName}}!",
            "children": [],
        }

        result = evaluate_agent(node)
        assert result == "Hello, Alice!"

    def test_should_evaluate_nested_agents_depth_first(self):
        """Should evaluate nested agents depth-first."""
        node: AgentNode = {
            "id": "main",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "Intro: {{introduction}}",
            "children": [
                {
                    "id": "introduction",
                    "name": None,
                    "major_version": None,
                    "params": {"userName": "Bob"},
                    "prompt": "Hello, {{userName}}!",
                    "children": [],
                },
            ],
        }

        result = evaluate_agent(node)
        assert result == "Intro: Hello, Bob!"

    def test_should_evaluate_multiple_levels_of_nesting(self):
        """Should evaluate multiple levels of nesting."""
        node: AgentNode = {
            "id": "main",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "Doc: {{section}}",
            "children": [
                {
                    "id": "section",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "Section: {{paragraph}}",
                    "children": [
                        {
                            "id": "paragraph",
                            "name": None,
                            "major_version": None,
                            "params": {"text": "content"},
                            "prompt": "Para: {{text}}",
                            "children": [],
                        },
                    ],
                },
            ],
        }

        result = evaluate_agent(node)
        assert result == "Doc: Section: Para: content"

    def test_should_handle_multiple_children(self):
        """Should handle multiple children."""
        node: AgentNode = {
            "id": "document",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "{{header}}\n{{body}}\n{{footer}}",
            "children": [
                {
                    "id": "header",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "HEADER",
                    "children": [],
                },
                {
                    "id": "body",
                    "name": None,
                    "major_version": None,
                    "params": {"content": "text"},
                    "prompt": "Body: {{content}}",
                    "children": [],
                },
                {
                    "id": "footer",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "FOOTER",
                    "children": [],
                },
            ],
        }

        result = evaluate_agent(node)
        assert result == "HEADER\nBody: text\nFOOTER"

    def test_should_handle_agent_with_no_children_or_params(self):
        """Should handle agent with no children or params."""
        node: AgentNode = {
            "id": "simple",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "Static text",
            "children": [],
        }

        result = evaluate_agent(node)
        assert result == "Static text"

    def test_should_cache_evaluated_nodes(self):
        """Should cache evaluated nodes to avoid recomputation."""
        shared_child: AgentNode = {
            "id": "shared",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "Shared",
            "children": [],
        }

        node: AgentNode = {
            "id": "main",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "{{shared}}",
            "children": [shared_child],
        }

        result = evaluate_agent(node)
        assert result == "Shared"


class TestTraverseAgentNode:
    """Tests for traverse_agent_node function."""

    def test_should_visit_single_node(self):
        """Should visit single node."""
        node: AgentNode = {
            "id": "root",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "test",
            "children": [],
        }

        visited = []
        traverse_agent_node(
            node,
            lambda n, parent_id: visited.append({"id": n["id"], "parent_id": parent_id}),
        )

        assert visited == [{"id": "root", "parent_id": None}]

    def test_should_visit_nodes_in_depth_first_order(self):
        """Should visit nodes in depth-first order."""
        node: AgentNode = {
            "id": "root",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "test",
            "children": [
                {
                    "id": "child1",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "test",
                    "children": [
                        {
                            "id": "grandchild1",
                            "name": None,
                            "major_version": None,
                            "params": {},
                            "prompt": "test",
                            "children": [],
                        },
                    ],
                },
                {
                    "id": "child2",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "test",
                    "children": [],
                },
            ],
        }

        visited = []
        traverse_agent_node(node, lambda n, _: visited.append(n["id"]))

        assert visited == ["root", "child1", "grandchild1", "child2"]

    def test_should_pass_correct_parent_id_to_callback(self):
        """Should pass correct parent ID to callback."""
        node: AgentNode = {
            "id": "root",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "test",
            "children": [
                {
                    "id": "child1",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "test",
                    "children": [
                        {
                            "id": "grandchild1",
                            "name": None,
                            "major_version": None,
                            "params": {},
                            "prompt": "test",
                            "children": [],
                        },
                    ],
                },
                {
                    "id": "child2",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "test",
                    "children": [],
                },
            ],
        }

        relationships = []
        traverse_agent_node(
            node,
            lambda n, parent_id: relationships.append({"id": n["id"], "parent_id": parent_id}),
        )

        assert relationships == [
            {"id": "root", "parent_id": None},
            {"id": "child1", "parent_id": "root"},
            {"id": "grandchild1", "parent_id": "child1"},
            {"id": "child2", "parent_id": "root"},
        ]


class TestFormatAgentRequest:
    """Tests for format_agent_request function."""

    def test_should_format_simple_agent_node(self):
        """Should format a simple agent node."""
        node: AgentNode = {
            "id": "greeting",
            "name": "greeting-agent",
            "major_version": 1,
            "params": {"userName": "Alice"},
            "prompt": "Hello, {{userName}}!",
            "children": [],
        }

        request = format_agent_request(node)

        assert request["agents"]["rootId"] == "greeting"
        assert request["agents"]["map"]["greeting"] == {
            "id": "greeting",
            "name": "greeting-agent",
            "majorVersion": 1,
            "prompt": "Hello, {{userName}}!",
            "paramKeys": ["userName"],
            "childrenIds": [],
            "model": None,
            "temperature": None,
            "maxTokens": None,
            "topP": None,
            "frequencyPenalty": None,
            "presencePenalty": None,
            "stopSequences": None,
        }

    def test_should_format_nested_agent_nodes(self):
        """Should format nested agent nodes."""
        node: AgentNode = {
            "id": "main",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "Intro: {{introduction}}",
            "children": [
                {
                    "id": "introduction",
                    "name": None,
                    "major_version": None,
                    "params": {"userName": "Bob"},
                    "prompt": "Hello, {{userName}}!",
                    "children": [],
                },
            ],
        }

        request = format_agent_request(node)

        assert request["agents"]["rootId"] == "main"
        assert request["agents"]["map"]["main"] == {
            "id": "main",
            "name": None,
            "majorVersion": None,
            "prompt": "Intro: {{introduction}}",
            "paramKeys": ["introduction"],
            "childrenIds": ["introduction"],
            "model": None,
            "temperature": None,
            "maxTokens": None,
            "topP": None,
            "frequencyPenalty": None,
            "presencePenalty": None,
            "stopSequences": None,
        }
        assert request["agents"]["map"]["introduction"] == {
            "id": "introduction",
            "name": None,
            "majorVersion": None,
            "prompt": "Hello, {{userName}}!",
            "paramKeys": ["userName"],
            "childrenIds": [],
            "model": None,
            "temperature": None,
            "maxTokens": None,
            "topP": None,
            "frequencyPenalty": None,
            "presencePenalty": None,
            "stopSequences": None,
        }

    def test_should_format_agent_node_with_hyperparameters(self):
        """Should format agent node with hyperparameters."""
        node: AgentNode = {
            "id": "greeting",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "Hello!",
            "children": [],
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "stop_sequences": ["END"],
        }

        request = format_agent_request(node)

        assert request["agents"]["map"]["greeting"]["model"] == "gpt-4"
        assert request["agents"]["map"]["greeting"]["temperature"] == 0.7
        assert request["agents"]["map"]["greeting"]["maxTokens"] == 1000
        assert request["agents"]["map"]["greeting"]["topP"] == 0.9
        assert request["agents"]["map"]["greeting"]["frequencyPenalty"] == 0.5
        assert request["agents"]["map"]["greeting"]["presencePenalty"] == 0.3
        assert request["agents"]["map"]["greeting"]["stopSequences"] == ["END"]

    def test_should_format_deeply_nested_structure(self):
        """Should format deeply nested structure."""
        node: AgentNode = {
            "id": "doc",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "{{section}}",
            "children": [
                {
                    "id": "section",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "{{paragraph}}",
                    "children": [
                        {
                            "id": "paragraph",
                            "name": None,
                            "major_version": None,
                            "params": {"text": "content"},
                            "prompt": "{{text}}",
                            "children": [],
                        },
                    ],
                },
            ],
        }

        request = format_agent_request(node)

        assert request["agents"]["rootId"] == "doc"
        assert len(request["agents"]["map"]) == 3
        assert request["agents"]["map"]["doc"]["childrenIds"] == ["section"]
        assert request["agents"]["map"]["section"]["childrenIds"] == ["paragraph"]
        assert request["agents"]["map"]["paragraph"]["paramKeys"] == ["text"]

    def test_should_handle_multiple_children(self):
        """Should handle multiple children."""
        node: AgentNode = {
            "id": "document",
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "{{header}} {{body}} {{footer}}",
            "children": [
                {
                    "id": "header",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "HEADER",
                    "children": [],
                },
                {
                    "id": "body",
                    "name": None,
                    "major_version": None,
                    "params": {"content": "text"},
                    "prompt": "{{content}}",
                    "children": [],
                },
                {
                    "id": "footer",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "FOOTER",
                    "children": [],
                },
            ],
        }

        request = format_agent_request(node)

        assert request["agents"]["map"]["document"]["childrenIds"] == [
            "header",
            "body",
            "footer",
        ]
        assert len(request["agents"]["map"]) == 4


class TestUpdateAgentNodes:
    """Tests for update_agent_nodes function."""

    def test_should_update_single_node(self):
        """Should update a single node."""
        node: AgentNode = {
            "id": "greeting",
            "name": None,
            "major_version": None,
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
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "root",
            "children": [
                {
                    "id": "child1",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "child1",
                    "children": [],
                },
                {
                    "id": "child2",
                    "name": None,
                    "major_version": None,
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
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "level1",
            "children": [
                {
                    "id": "level2",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "level2",
                    "children": [
                        {
                            "id": "level3",
                            "name": None,
                            "major_version": None,
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
            "name": "root-name",
            "major_version": 1,
            "params": {"key": "value"},
            "prompt": "original",
            "children": [
                {
                    "id": "child",
                    "name": None,
                    "major_version": None,
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
            "name": None,
            "major_version": None,
            "params": {},
            "prompt": "root",
            "children": [
                {
                    "id": "update-me",
                    "name": None,
                    "major_version": None,
                    "params": {},
                    "prompt": "old",
                    "children": [],
                },
                {
                    "id": "leave-me",
                    "name": None,
                    "major_version": None,
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
