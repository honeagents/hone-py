"""
Unit tests for Hone SDK prompt utilities.

Exact replica of TypeScript prompt.test.ts - tests all prompt utility functions.
"""

import pytest

from hone.prompt import (
    get_prompt_node,
    evaluate_prompt,
    insert_params_into_prompt,
    traverse_prompt_node,
    format_prompt_request,
    update_prompt_nodes,
)
from hone.types import GetPromptOptions, PromptNode


class TestGetPromptNode:
    """Tests for get_prompt_node function."""

    def test_should_create_simple_prompt_node_with_no_parameters(self):
        """Should create a simple prompt node with no parameters."""
        options: GetPromptOptions = {
            "default_prompt": "Hello, World!",
        }

        node = get_prompt_node("greeting", options)

        assert node == {
            "id": "greeting",
            "version": None,
            "name": None,
            "params": {},
            "prompt": "Hello, World!",
            "children": [],
        }

    def test_should_create_prompt_node_with_simple_string_parameters(self):
        """Should create a prompt node with simple string parameters."""
        options: GetPromptOptions = {
            "default_prompt": "Hello, {{userName}}!",
            "params": {
                "userName": "Alice",
            },
        }

        node = get_prompt_node("greeting", options)

        assert node == {
            "id": "greeting",
            "version": None,
            "name": None,
            "params": {
                "userName": "Alice",
            },
            "prompt": "Hello, {{userName}}!",
            "children": [],
        }

    def test_should_create_prompt_node_with_version_and_name(self):
        """Should create a prompt node with version and name."""
        options: GetPromptOptions = {
            "version": "v1",
            "name": "greeting-prompt",
            "default_prompt": "Hello!",
        }

        node = get_prompt_node("greeting", options)

        assert node["version"] == "v1"
        assert node["name"] == "greeting-prompt"

    def test_should_create_nested_prompt_nodes_from_nested_options(self):
        """Should create nested prompt nodes from nested options."""
        options: GetPromptOptions = {
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

        node = get_prompt_node("main", options)

        assert node["id"] == "main"
        assert len(node["children"]) == 1
        assert node["children"][0]["id"] == "introduction"
        assert node["children"][0]["params"] == {"userName": "Bob"}

    def test_should_handle_multiple_nested_prompts(self):
        """Should handle multiple nested prompts."""
        options: GetPromptOptions = {
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

        node = get_prompt_node("document", options)

        assert len(node["children"]) == 3
        assert [c["id"] for c in node["children"]] == ["header", "body", "footer"]

    def test_should_throw_error_for_self_referencing_prompts(self):
        """Should throw an error for self-referencing prompts."""
        options: GetPromptOptions = {
            "default_prompt": "This is a prompt that references {{system-prompt}}",
            "params": {
                "system-prompt": {
                    "default_prompt": "This should cause an error",
                },
            },
        }

        with pytest.raises(ValueError):
            get_prompt_node("system-prompt", options)

    def test_should_throw_error_for_circular_prompt_references(self):
        """Should throw an error for circular prompt references."""
        options: GetPromptOptions = {
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
            get_prompt_node("a", options)

    def test_should_throw_error_when_prompt_has_placeholders_without_matching_parameters(self):
        """Should throw an error when prompt has placeholders without matching parameters."""
        options: GetPromptOptions = {
            "default_prompt": "Hello {{name}}, your role is {{role}}",
            "params": {
                "name": "Alice",
                # 'role' is missing
            },
        }

        node = get_prompt_node("greeting", options)

        # Should throw when evaluating because 'role' placeholder has no value
        with pytest.raises(ValueError, match=r"(?i)missing parameter.*role"):
            evaluate_prompt(node)

    def test_should_throw_error_listing_all_missing_parameters(self):
        """Should throw an error listing all missing parameters."""
        options: GetPromptOptions = {
            "default_prompt": "{{greeting}} {{name}}, you are {{role}} in {{location}}",
            "params": {
                "name": "Bob",
                # Missing: greeting, role, location
            },
        }

        node = get_prompt_node("test", options)

        with pytest.raises(ValueError, match=r"(?i)missing parameter"):
            evaluate_prompt(node)


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


class TestEvaluatePrompt:
    """Tests for evaluate_prompt function."""

    def test_should_evaluate_simple_prompt_with_params(self):
        """Should evaluate a simple prompt with params."""
        node: PromptNode = {
            "id": "greeting",
            "name": None,
            "version": None,
            "params": {"userName": "Alice"},
            "prompt": "Hello, {{userName}}!",
            "children": [],
        }

        result = evaluate_prompt(node)
        assert result == "Hello, Alice!"

    def test_should_evaluate_nested_prompts_depth_first(self):
        """Should evaluate nested prompts depth-first."""
        node: PromptNode = {
            "id": "main",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "Intro: {{introduction}}",
            "children": [
                {
                    "id": "introduction",
                    "name": None,
                    "version": None,
                    "params": {"userName": "Bob"},
                    "prompt": "Hello, {{userName}}!",
                    "children": [],
                },
            ],
        }

        result = evaluate_prompt(node)
        assert result == "Intro: Hello, Bob!"

    def test_should_evaluate_multiple_levels_of_nesting(self):
        """Should evaluate multiple levels of nesting."""
        node: PromptNode = {
            "id": "main",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "Doc: {{section}}",
            "children": [
                {
                    "id": "section",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "Section: {{paragraph}}",
                    "children": [
                        {
                            "id": "paragraph",
                            "name": None,
                            "version": None,
                            "params": {"text": "content"},
                            "prompt": "Para: {{text}}",
                            "children": [],
                        },
                    ],
                },
            ],
        }

        result = evaluate_prompt(node)
        assert result == "Doc: Section: Para: content"

    def test_should_handle_multiple_children(self):
        """Should handle multiple children."""
        node: PromptNode = {
            "id": "document",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "{{header}}\n{{body}}\n{{footer}}",
            "children": [
                {
                    "id": "header",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "HEADER",
                    "children": [],
                },
                {
                    "id": "body",
                    "name": None,
                    "version": None,
                    "params": {"content": "text"},
                    "prompt": "Body: {{content}}",
                    "children": [],
                },
                {
                    "id": "footer",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "FOOTER",
                    "children": [],
                },
            ],
        }

        result = evaluate_prompt(node)
        assert result == "HEADER\nBody: text\nFOOTER"

    def test_should_handle_prompt_with_no_children_or_params(self):
        """Should handle prompt with no children or params."""
        node: PromptNode = {
            "id": "simple",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "Static text",
            "children": [],
        }

        result = evaluate_prompt(node)
        assert result == "Static text"

    def test_should_cache_evaluated_nodes(self):
        """Should cache evaluated nodes to avoid recomputation."""
        shared_child: PromptNode = {
            "id": "shared",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "Shared",
            "children": [],
        }

        node: PromptNode = {
            "id": "main",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "{{shared}}",
            "children": [shared_child],
        }

        result = evaluate_prompt(node)
        assert result == "Shared"


class TestTraversePromptNode:
    """Tests for traverse_prompt_node function."""

    def test_should_visit_single_node(self):
        """Should visit single node."""
        node: PromptNode = {
            "id": "root",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "test",
            "children": [],
        }

        visited = []
        traverse_prompt_node(
            node,
            lambda n, parent_id: visited.append({"id": n["id"], "parent_id": parent_id}),
        )

        assert visited == [{"id": "root", "parent_id": None}]

    def test_should_visit_nodes_in_depth_first_order(self):
        """Should visit nodes in depth-first order."""
        node: PromptNode = {
            "id": "root",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "test",
            "children": [
                {
                    "id": "child1",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "test",
                    "children": [
                        {
                            "id": "grandchild1",
                            "name": None,
                            "version": None,
                            "params": {},
                            "prompt": "test",
                            "children": [],
                        },
                    ],
                },
                {
                    "id": "child2",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "test",
                    "children": [],
                },
            ],
        }

        visited = []
        traverse_prompt_node(node, lambda n, _: visited.append(n["id"]))

        assert visited == ["root", "child1", "grandchild1", "child2"]

    def test_should_pass_correct_parent_id_to_callback(self):
        """Should pass correct parent ID to callback."""
        node: PromptNode = {
            "id": "root",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "test",
            "children": [
                {
                    "id": "child1",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "test",
                    "children": [
                        {
                            "id": "grandchild1",
                            "name": None,
                            "version": None,
                            "params": {},
                            "prompt": "test",
                            "children": [],
                        },
                    ],
                },
                {
                    "id": "child2",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "test",
                    "children": [],
                },
            ],
        }

        relationships = []
        traverse_prompt_node(
            node,
            lambda n, parent_id: relationships.append({"id": n["id"], "parent_id": parent_id}),
        )

        assert relationships == [
            {"id": "root", "parent_id": None},
            {"id": "child1", "parent_id": "root"},
            {"id": "grandchild1", "parent_id": "child1"},
            {"id": "child2", "parent_id": "root"},
        ]


class TestFormatPromptRequest:
    """Tests for format_prompt_request function."""

    def test_should_format_simple_prompt_node(self):
        """Should format a simple prompt node."""
        node: PromptNode = {
            "id": "greeting",
            "name": "greeting-prompt",
            "version": "v1",
            "params": {"userName": "Alice"},
            "prompt": "Hello, {{userName}}!",
            "children": [],
        }

        request = format_prompt_request(node)

        assert request["prompts"]["rootId"] == "greeting"
        assert request["prompts"]["map"]["greeting"] == {
            "id": "greeting",
            "name": "greeting-prompt",
            "version": "v1",
            "prompt": "Hello, {{userName}}!",
            "paramKeys": ["userName"],
            "childrenIds": [],
        }

    def test_should_format_nested_prompt_nodes(self):
        """Should format nested prompt nodes."""
        node: PromptNode = {
            "id": "main",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "Intro: {{introduction}}",
            "children": [
                {
                    "id": "introduction",
                    "name": None,
                    "version": None,
                    "params": {"userName": "Bob"},
                    "prompt": "Hello, {{userName}}!",
                    "children": [],
                },
            ],
        }

        request = format_prompt_request(node)

        assert request["prompts"]["rootId"] == "main"
        assert request["prompts"]["map"]["main"] == {
            "id": "main",
            "name": None,
            "version": None,
            "prompt": "Intro: {{introduction}}",
            "paramKeys": ["introduction"],
            "childrenIds": ["introduction"],
        }
        assert request["prompts"]["map"]["introduction"] == {
            "id": "introduction",
            "name": None,
            "version": None,
            "prompt": "Hello, {{userName}}!",
            "paramKeys": ["userName"],
            "childrenIds": [],
        }

    def test_should_format_deeply_nested_structure(self):
        """Should format deeply nested structure."""
        node: PromptNode = {
            "id": "doc",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "{{section}}",
            "children": [
                {
                    "id": "section",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "{{paragraph}}",
                    "children": [
                        {
                            "id": "paragraph",
                            "name": None,
                            "version": None,
                            "params": {"text": "content"},
                            "prompt": "{{text}}",
                            "children": [],
                        },
                    ],
                },
            ],
        }

        request = format_prompt_request(node)

        assert request["prompts"]["rootId"] == "doc"
        assert len(request["prompts"]["map"]) == 3
        assert request["prompts"]["map"]["doc"]["childrenIds"] == ["section"]
        assert request["prompts"]["map"]["section"]["childrenIds"] == ["paragraph"]
        assert request["prompts"]["map"]["paragraph"]["paramKeys"] == ["text"]

    def test_should_handle_multiple_children(self):
        """Should handle multiple children."""
        node: PromptNode = {
            "id": "document",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "{{header}} {{body}} {{footer}}",
            "children": [
                {
                    "id": "header",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "HEADER",
                    "children": [],
                },
                {
                    "id": "body",
                    "name": None,
                    "version": None,
                    "params": {"content": "text"},
                    "prompt": "{{content}}",
                    "children": [],
                },
                {
                    "id": "footer",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "FOOTER",
                    "children": [],
                },
            ],
        }

        request = format_prompt_request(node)

        assert request["prompts"]["map"]["document"]["childrenIds"] == [
            "header",
            "body",
            "footer",
        ]
        assert len(request["prompts"]["map"]) == 4


class TestUpdatePromptNodes:
    """Tests for update_prompt_nodes function."""

    def test_should_update_single_node(self):
        """Should update a single node."""
        node: PromptNode = {
            "id": "greeting",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "Old prompt",
            "children": [],
        }

        updated = update_prompt_nodes(node, lambda n: {**n, "prompt": "New prompt"})

        assert updated["prompt"] == "New prompt"
        assert updated["id"] == "greeting"

    def test_should_update_all_nodes_in_nested_structure(self):
        """Should update all nodes in a nested structure."""
        node: PromptNode = {
            "id": "root",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "root",
            "children": [
                {
                    "id": "child1",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "child1",
                    "children": [],
                },
                {
                    "id": "child2",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "child2",
                    "children": [],
                },
            ],
        }

        updated = update_prompt_nodes(node, lambda n: {**n, "prompt": f"updated-{n['id']}"})

        assert updated["prompt"] == "updated-root"
        assert updated["children"][0]["prompt"] == "updated-child1"
        assert updated["children"][1]["prompt"] == "updated-child2"

    def test_should_update_deeply_nested_nodes(self):
        """Should update deeply nested nodes."""
        node: PromptNode = {
            "id": "level1",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "level1",
            "children": [
                {
                    "id": "level2",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "level2",
                    "children": [
                        {
                            "id": "level3",
                            "name": None,
                            "version": None,
                            "params": {},
                            "prompt": "level3",
                            "children": [],
                        },
                    ],
                },
            ],
        }

        updated = update_prompt_nodes(node, lambda n: {**n, "prompt": f"{n['prompt']}-updated"})

        assert updated["prompt"] == "level1-updated"
        assert updated["children"][0]["prompt"] == "level2-updated"
        assert updated["children"][0]["children"][0]["prompt"] == "level3-updated"

    def test_should_preserve_node_structure_while_updating(self):
        """Should preserve node structure while updating."""
        node: PromptNode = {
            "id": "root",
            "name": "root-name",
            "version": "v1",
            "params": {"key": "value"},
            "prompt": "original",
            "children": [
                {
                    "id": "child",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "child-original",
                    "children": [],
                },
            ],
        }

        updated = update_prompt_nodes(node, lambda n: {**n, "prompt": n["prompt"].upper()})

        assert updated["id"] == "root"
        assert updated["name"] == "root-name"
        assert updated["version"] == "v1"
        assert updated["params"] == {"key": "value"}
        assert updated["prompt"] == "ORIGINAL"
        assert updated["children"][0]["prompt"] == "CHILD-ORIGINAL"

    def test_should_allow_conditional_updates(self):
        """Should allow conditional updates."""
        node: PromptNode = {
            "id": "root",
            "name": None,
            "version": None,
            "params": {},
            "prompt": "root",
            "children": [
                {
                    "id": "update-me",
                    "name": None,
                    "version": None,
                    "params": {},
                    "prompt": "old",
                    "children": [],
                },
                {
                    "id": "leave-me",
                    "name": None,
                    "version": None,
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

        updated = update_prompt_nodes(node, conditional_update)

        assert updated["children"][0]["prompt"] == "new"
        assert updated["children"][1]["prompt"] == "unchanged"
