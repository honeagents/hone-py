"""
Integration tests for the Hone SDK client.

Exact replica of TypeScript sdk-client.integration.test.ts - tests verify
the full SDK client works correctly with the sync_prompts RPC.
"""

import pytest

from hone.client import Hone

from .helpers import (
    TestDatabase,
    TestFixture,
    unique_prompt_id,
)

# Point the SDK to local Supabase
SUPABASE_RPC_URL = "http://127.0.0.1:54321/rest/v1/rpc"


class TestHonePromptMethod:
    """Tests for Hone.prompt method with real database."""

    @pytest.mark.asyncio
    async def test_should_create_prompt_and_return_evaluated_result(
        self, db: TestDatabase, fixture: TestFixture, clean_project
    ):
        """Should create prompt and return evaluated result."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("basic")

        result = await client.prompt(prompt_id, {
            "default_prompt": "Hello, {{name}}!",
            "params": {
                "name": "World",
            },
        })

        # Should return the evaluated prompt
        assert result == "Hello, World!"

        # Verify prompt was created in database
        prompt = await db.get_prompt(fixture.project.id, prompt_id)
        assert prompt is not None
        assert prompt.text == "Hello, {{name}}!"

    @pytest.mark.asyncio
    async def test_should_return_updated_prompt_text_from_database(
        self, db: TestDatabase, fixture: TestFixture, clean_project
    ):
        """Should return the updated prompt text from the database."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("update-text")

        # First call creates the prompt
        await client.prompt(prompt_id, {
            "params": {"name": "Gabe"},
            "default_prompt": "{{name}}'s first prompt",
        })

        # Should have created the prompt
        prompt = await db.get_prompt(fixture.project.id, prompt_id)
        assert prompt is not None
        assert prompt.text == "{{name}}'s first prompt"

        # Directly update the prompt text in database (simulating UI change)
        await db.patch(
            "prompts",
            {"text": "{{name}}'s UPDATED prompt"},
            {"project_id": fixture.project.id, "id": prompt_id},
        )

        # Second call should get the updated prompt
        result = await client.prompt(prompt_id, {
            "params": {"name": "Gabe"},
            "default_prompt": "{{name}}'s first prompt",
        })

        # Should return the evaluated updated prompt
        assert result == "Gabe's UPDATED prompt"

    @pytest.mark.asyncio
    async def test_should_default_name_to_prompt_id(
        self, db: TestDatabase, fixture: TestFixture, clean_project
    ):
        """Should default name to prompt id when name is not provided."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("noname")

        await client.prompt(prompt_id, {
            "default_prompt": "A prompt without an explicit name",
        })

        # Verify name defaults to the prompt ID in the database
        prompt = await db.get_prompt(fixture.project.id, prompt_id)
        assert prompt is not None
        assert prompt.name == prompt_id

    @pytest.mark.asyncio
    async def test_should_use_fallback_when_api_call_fails(
        self, fixture: TestFixture, clean_project
    ):
        """Should use fallback when API call fails."""
        # Create client with invalid API key
        bad_client = Hone({
            "api_key": "invalid_key",
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("fallback")

        # Should use fallback prompt instead of throwing
        result = await bad_client.prompt(prompt_id, {
            "default_prompt": "Fallback: {{name}}",
            "params": {
                "name": "User",
            },
        })

        assert result == "Fallback: User"

    @pytest.mark.asyncio
    async def test_should_handle_prompt_without_parameters(
        self, fixture: TestFixture, clean_project
    ):
        """Should handle prompt without parameters."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("noparam")

        result = await client.prompt(prompt_id, {
            "default_prompt": "Static prompt with no params",
        })

        assert result == "Static prompt with no params"

    @pytest.mark.asyncio
    async def test_should_handle_prompt_with_version_and_name(
        self, db: TestDatabase, fixture: TestFixture, clean_project
    ):
        """Should handle prompt with version and name."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("versioned")

        result = await client.prompt(prompt_id, {
            "version": "2.0.0",
            "name": "My Named Prompt",
            "default_prompt": "Hello, {{user}}!",
            "params": {"user": "Developer"},
        })

        assert result == "Hello, Developer!"

        # Verify metadata was stored
        prompt = await db.get_prompt(fixture.project.id, prompt_id)
        assert prompt is not None
        assert prompt.version == "2.0.0"
        assert prompt.name == "My Named Prompt"


class TestNestedPrompts:
    """Tests for nested prompt handling."""

    @pytest.mark.asyncio
    async def test_should_handle_nested_prompt_calls(
        self, db: TestDatabase, fixture: TestFixture, clean_project
    ):
        """Should handle nested prompt calls."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        main_id = unique_prompt_id("main")
        intro_id = "intro"

        result = await client.prompt(main_id, {
            "default_prompt": "Welcome! {{intro}}",
            "params": {
                intro_id: {
                    "default_prompt": "Hello, {{name}}!",
                    "params": {"name": "Nested User"},
                },
            },
        })

        assert result == "Welcome! Hello, Nested User!"

        # Verify both prompts were created
        main_prompt = await db.get_prompt(fixture.project.id, main_id)
        intro_prompt = await db.get_prompt(fixture.project.id, intro_id)

        assert main_prompt is not None
        assert intro_prompt is not None

        # Verify hierarchy was created
        hierarchy = await db.get_hierarchy(fixture.project.id)
        assert len(hierarchy) == 1
        assert hierarchy[0].parent_prompt_id == main_id
        assert hierarchy[0].child_prompt_id == intro_id

    @pytest.mark.asyncio
    async def test_should_return_updated_nested_prompt_text(
        self, db: TestDatabase, fixture: TestFixture, clean_project
    ):
        """Should return the updated prompt text from the database for nested prompts."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        parent_id = unique_prompt_id("parent")
        child_id = "child"

        # First call creates both prompts
        await client.prompt(parent_id, {
            "default_prompt": "Parent says: {{child}}",
            "params": {
                child_id: {
                    "default_prompt": "Child v1",
                },
            },
        })

        # Update child prompt text directly in database
        await db.patch(
            "prompts",
            {"text": "Child v2"},
            {"project_id": fixture.project.id, "id": child_id},
        )

        # Second call should get updated child prompt
        result = await client.prompt(parent_id, {
            "default_prompt": "Parent says: {{child}}",
            "params": {
                child_id: {
                    "default_prompt": "Child v1",
                },
            },
        })

        assert result == "Parent says: Child v2"

    @pytest.mark.asyncio
    async def test_should_handle_deeply_nested_prompts(
        self, db: TestDatabase, fixture: TestFixture, clean_project
    ):
        """Should handle deeply nested prompts."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        l0_id = unique_prompt_id("l0")
        l1_key = "l1"
        l2_key = "l2"

        result = await client.prompt(l0_id, {
            "default_prompt": "L0 -> {{l1}}",
            "params": {
                l1_key: {
                    "default_prompt": "L1 -> {{l2}}",
                    "params": {
                        l2_key: {
                            "default_prompt": "L2 (deepest)",
                        },
                    },
                },
            },
        })

        assert result == "L0 -> L1 -> L2 (deepest)"

        # Verify hierarchy depth
        hierarchy = await db.get_hierarchy(fixture.project.id)
        assert len(hierarchy) == 2

    @pytest.mark.asyncio
    async def test_should_handle_mixed_nested_and_string_params(
        self, fixture: TestFixture, clean_project
    ):
        """Should handle mixed nested and string params."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        root_id = unique_prompt_id("mixroot")
        greeting_key = "greeting"

        result = await client.prompt(root_id, {
            "default_prompt": "Hello {{userName}}, {{greeting}}",
            "params": {
                "userName": "Alice",
                greeting_key: {
                    "default_prompt": "welcome to {{place}}!",
                    "params": {"place": "the app"},
                },
            },
        })

        assert result == "Hello Alice, welcome to the app!"


class TestParameterSubstitution:
    """Tests for parameter substitution."""

    @pytest.mark.asyncio
    async def test_should_substitute_multiple_occurrences_of_same_param(
        self, fixture: TestFixture, clean_project
    ):
        """Should substitute multiple occurrences of same param."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("multi")

        result = await client.prompt(prompt_id, {
            "default_prompt": "{{name}} says: Hello {{name}}!",
            "params": {"name": "Bot"},
        })

        assert result == "Bot says: Hello Bot!"

    @pytest.mark.asyncio
    async def test_should_handle_params_with_special_regex_characters(
        self, fixture: TestFixture, clean_project
    ):
        """Should handle params with special regex characters."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("regex")

        result = await client.prompt(prompt_id, {
            "default_prompt": "Result: {{value}}",
            "params": {"value": "$100.00 (50% off!)"},
        })

        assert result == "Result: $100.00 (50% off!)"

    @pytest.mark.asyncio
    async def test_should_preserve_param_order(
        self, fixture: TestFixture, clean_project
    ):
        """Should preserve param order."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("order")

        result = await client.prompt(prompt_id, {
            "default_prompt": "{{a}} {{b}} {{c}}",
            "params": {"a": "1", "b": "2", "c": "3"},
        })

        assert result == "1 2 3"


class TestRemoteUpdates:
    """Tests for remote prompt updates."""

    @pytest.mark.asyncio
    async def test_should_use_updated_prompt_when_version_changes(
        self, db: TestDatabase, fixture: TestFixture, clean_project
    ):
        """Should use updated prompt when version changes."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("update")

        # First call with v1
        result = await client.prompt(prompt_id, {
            "version": "1.0",
            "default_prompt": "Version 1: {{name}}",
            "params": {"name": "Test"},
        })
        assert result == "Version 1: Test"

        # Second call with v2 - should update
        result = await client.prompt(prompt_id, {
            "version": "2.0",
            "default_prompt": "Version 2: {{name}}",
            "params": {"name": "Test"},
        })
        assert result == "Version 2: Test"

        # Verify database has v2
        prompt = await db.get_prompt(fixture.project.id, prompt_id)
        assert prompt is not None
        assert prompt.version == "2.0"
        assert prompt.text == "Version 2: {{name}}"

    @pytest.mark.asyncio
    async def test_should_not_update_database_when_version_unchanged(
        self, db: TestDatabase, fixture: TestFixture, clean_project
    ):
        """Should not update database when version unchanged."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("cached")

        # First call
        await client.prompt(prompt_id, {
            "version": "1.0",
            "default_prompt": "Original prompt",
        })

        # Verify initial state
        prompt = await db.get_prompt(fixture.project.id, prompt_id)
        assert prompt is not None
        assert prompt.text == "Original prompt"

        # Second call with same version but different default
        result = await client.prompt(prompt_id, {
            "version": "1.0",
            "default_prompt": "Different default (should not save)",
        })

        # SDK returns the stored database prompt
        assert result == "Original prompt"

        # Database should still have original
        prompt = await db.get_prompt(fixture.project.id, prompt_id)
        assert prompt is not None
        assert prompt.text == "Original prompt"


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_should_gracefully_handle_network_errors(
        self, fixture: TestFixture, clean_project
    ):
        """Should gracefully handle network errors."""
        # Create client with unreachable URL
        bad_client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": "http://localhost:99999/bad",
            "timeout": 1000,
        })
        prompt_id = unique_prompt_id("network")

        # Should fall back to default prompt
        result = await bad_client.prompt(prompt_id, {
            "default_prompt": "Fallback prompt",
        })

        assert result == "Fallback prompt"

    @pytest.mark.asyncio
    async def test_should_throw_for_missing_required_parameters(
        self, fixture: TestFixture, clean_project
    ):
        """Should throw for missing required parameters."""
        client = Hone({
            "api_key": fixture.project.api_key,
            "base_url": SUPABASE_RPC_URL,
        })
        prompt_id = unique_prompt_id("missing")

        # When using a template with a placeholder but not providing the param,
        # the SDK should throw an error during evaluation
        with pytest.raises(ValueError, match="Missing parameter"):
            await client.prompt(prompt_id, {
                "default_prompt": "Hello {{missingParam}}!",
                # No params provided
            })
