"""
Unit tests for Hone SDK providers module.

Exact replica of TypeScript providers tests - tests AIProvider enum and helper functions.
"""

import pytest

from hone.providers import (
    AIProvider,
    AI_PROVIDER_VALUES,
    is_valid_provider,
    get_provider_display_name,
)


class TestAIProvider:
    """Tests for AIProvider enum."""

    def test_should_have_all_expected_providers(self):
        """Should have all expected provider values."""
        expected_providers = [
            "openai",
            "anthropic",
            "google",
            "google-vertex",
            "azure",
            "xai",
            "mistral",
            "cohere",
            "groq",
            "togetherai",
            "fireworks",
            "deepinfra",
            "deepseek",
            "cerebras",
            "perplexity",
            "amazon-bedrock",
            "baseten",
        ]

        for provider in expected_providers:
            assert provider in AI_PROVIDER_VALUES, f"Missing provider: {provider}"

    def test_should_have_correct_string_values(self):
        """Should have correct string values for each provider."""
        assert AIProvider.OPENAI.value == "openai"
        assert AIProvider.ANTHROPIC.value == "anthropic"
        assert AIProvider.GOOGLE.value == "google"
        assert AIProvider.GOOGLE_VERTEX.value == "google-vertex"
        assert AIProvider.AZURE.value == "azure"
        assert AIProvider.XAI.value == "xai"
        assert AIProvider.MISTRAL.value == "mistral"
        assert AIProvider.COHERE.value == "cohere"
        assert AIProvider.GROQ.value == "groq"
        assert AIProvider.TOGETHERAI.value == "togetherai"
        assert AIProvider.FIREWORKS.value == "fireworks"
        assert AIProvider.DEEPINFRA.value == "deepinfra"
        assert AIProvider.DEEPSEEK.value == "deepseek"
        assert AIProvider.CEREBRAS.value == "cerebras"
        assert AIProvider.PERPLEXITY.value == "perplexity"
        assert AIProvider.AMAZON_BEDROCK.value == "amazon-bedrock"
        assert AIProvider.BASETEN.value == "baseten"

    def test_should_be_usable_as_string(self):
        """Provider enum values should be usable as strings."""
        provider = AIProvider.OPENAI
        # Direct comparison works due to str inheritance
        assert provider == "openai"
        # .value gives the string value
        assert provider.value == "openai"

    def test_ai_provider_values_should_match_enum(self):
        """AI_PROVIDER_VALUES should match all enum values."""
        enum_values = [p.value for p in AIProvider]
        assert set(AI_PROVIDER_VALUES) == set(enum_values)
        assert len(AI_PROVIDER_VALUES) == len(AIProvider)


class TestIsValidProvider:
    """Tests for is_valid_provider function."""

    def test_should_return_true_for_valid_providers(self):
        """Should return True for valid provider strings."""
        assert is_valid_provider("openai") is True
        assert is_valid_provider("anthropic") is True
        assert is_valid_provider("google") is True
        assert is_valid_provider("amazon-bedrock") is True

    def test_should_return_false_for_invalid_providers(self):
        """Should return False for invalid provider strings."""
        assert is_valid_provider("invalid") is False
        assert is_valid_provider("OpenAI") is False  # Case sensitive
        assert is_valid_provider("") is False
        assert is_valid_provider("gpt-4") is False  # Model, not provider

    def test_should_work_with_all_enum_values(self):
        """Should return True for all AIProvider enum values."""
        for provider in AIProvider:
            assert is_valid_provider(provider.value) is True


class TestGetProviderDisplayName:
    """Tests for get_provider_display_name function."""

    def test_should_return_correct_display_names(self):
        """Should return correct display names for known providers."""
        assert get_provider_display_name("openai") == "OpenAI"
        assert get_provider_display_name("anthropic") == "Anthropic"
        assert get_provider_display_name("google") == "Google AI"
        assert get_provider_display_name("google-vertex") == "Google Vertex AI"
        assert get_provider_display_name("azure") == "Azure OpenAI"
        assert get_provider_display_name("xai") == "xAI"
        assert get_provider_display_name("mistral") == "Mistral AI"
        assert get_provider_display_name("cohere") == "Cohere"
        assert get_provider_display_name("groq") == "Groq"
        assert get_provider_display_name("togetherai") == "Together.ai"
        assert get_provider_display_name("fireworks") == "Fireworks"
        assert get_provider_display_name("deepinfra") == "DeepInfra"
        assert get_provider_display_name("deepseek") == "DeepSeek"
        assert get_provider_display_name("cerebras") == "Cerebras"
        assert get_provider_display_name("perplexity") == "Perplexity"
        assert get_provider_display_name("amazon-bedrock") == "Amazon Bedrock"
        assert get_provider_display_name("baseten") == "Baseten"

    def test_should_return_input_for_unknown_providers(self):
        """Should return the input string for unknown providers."""
        assert get_provider_display_name("unknown") == "unknown"
        assert get_provider_display_name("my-custom-provider") == "my-custom-provider"

    def test_should_work_with_enum_values(self):
        """Should work when passed AIProvider enum values."""
        assert get_provider_display_name(AIProvider.OPENAI.value) == "OpenAI"
        assert get_provider_display_name(AIProvider.AMAZON_BEDROCK.value) == "Amazon Bedrock"


class TestProviderImports:
    """Tests for provider module imports."""

    def test_should_be_importable_from_hone_package(self):
        """Should be importable from the main hone package."""
        from hone import (
            AIProvider,
            AI_PROVIDER_VALUES,
            is_valid_provider,
            get_provider_display_name,
        )

        assert AIProvider is not None
        assert AI_PROVIDER_VALUES is not None
        assert is_valid_provider is not None
        assert get_provider_display_name is not None

    def test_should_be_in_all_exports(self):
        """Should be included in __all__."""
        import hone

        assert "AIProvider" in hone.__all__
        assert "AI_PROVIDER_VALUES" in hone.__all__
        assert "is_valid_provider" in hone.__all__
        assert "get_provider_display_name" in hone.__all__
