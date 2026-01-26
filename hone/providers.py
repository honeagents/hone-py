"""
AI Provider definitions compatible with Vercel AI SDK.

This module provides type-safe provider identifiers that align with
the Vercel AI SDK's official provider packages (@ai-sdk/*).

See: https://ai-sdk.dev/providers/ai-sdk-providers

Example:
    >>> from hone import AIProvider
    >>>
    >>> agent = await hone.agent("my-agent", {
    ...     "provider": AIProvider.OPENAI,
    ...     "model": "gpt-4o",
    ...     "default_prompt": "You are a helpful assistant.",
    ... })
"""

from enum import Enum
from typing import Tuple


class AIProvider(str, Enum):
    """
    Supported AI providers from the Vercel AI SDK ecosystem.

    These correspond to the official @ai-sdk/* provider packages.
    Use these identifiers when specifying the `provider` field in agent options.
    """

    # =========================================================================
    # Major LLM Providers
    # =========================================================================

    # OpenAI - GPT models (gpt-4o, gpt-4, gpt-3.5-turbo, etc.)
    OPENAI = "openai"

    # Anthropic - Claude models (claude-3-opus, claude-3-sonnet, claude-3-haiku, etc.)
    ANTHROPIC = "anthropic"

    # Google Generative AI - Gemini models (gemini-pro, gemini-1.5-pro, etc.)
    GOOGLE = "google"

    # Google Vertex AI - Enterprise Gemini models
    GOOGLE_VERTEX = "google-vertex"

    # Azure OpenAI Service - Azure-hosted OpenAI models
    AZURE = "azure"

    # =========================================================================
    # Specialized Providers
    # =========================================================================

    # xAI - Grok models
    XAI = "xai"

    # Mistral AI - Mistral models (mistral-large, mistral-medium, etc.)
    MISTRAL = "mistral"

    # Cohere - Command models
    COHERE = "cohere"

    # =========================================================================
    # Inference Providers
    # =========================================================================

    # Groq - Fast inference for open models
    GROQ = "groq"

    # Together.ai - Open model hosting
    TOGETHERAI = "togetherai"

    # Fireworks - Fast inference platform
    FIREWORKS = "fireworks"

    # DeepInfra - Model inference
    DEEPINFRA = "deepinfra"

    # DeepSeek - DeepSeek models
    DEEPSEEK = "deepseek"

    # Cerebras - Fast inference
    CEREBRAS = "cerebras"

    # Perplexity - Perplexity models with web search
    PERPLEXITY = "perplexity"

    # =========================================================================
    # Cloud Providers
    # =========================================================================

    # Amazon Bedrock - AWS-hosted models
    AMAZON_BEDROCK = "amazon-bedrock"

    # Baseten - Model hosting platform
    BASETEN = "baseten"


# Type alias for provider values (strings)
AIProviderValue = str

# List of all valid provider values for runtime validation
AI_PROVIDER_VALUES: Tuple[str, ...] = tuple(provider.value for provider in AIProvider)


def is_valid_provider(value: str) -> bool:
    """
    Check if a string is a valid AI provider.

    Args:
        value: The string to check

    Returns:
        True if the value is a valid AIProvider

    Example:
        >>> if is_valid_provider(user_input):
        ...     # user_input is a valid provider
        ...     pass
    """
    return value in AI_PROVIDER_VALUES


# Display name mapping for providers
_PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google AI",
    "google-vertex": "Google Vertex AI",
    "azure": "Azure OpenAI",
    "xai": "xAI",
    "mistral": "Mistral AI",
    "cohere": "Cohere",
    "groq": "Groq",
    "togetherai": "Together.ai",
    "fireworks": "Fireworks",
    "deepinfra": "DeepInfra",
    "deepseek": "DeepSeek",
    "cerebras": "Cerebras",
    "perplexity": "Perplexity",
    "amazon-bedrock": "Amazon Bedrock",
    "baseten": "Baseten",
}


def get_provider_display_name(provider: str) -> str:
    """
    Get the display name for a provider.

    Args:
        provider: The provider identifier

    Returns:
        Human-readable provider name

    Example:
        >>> get_provider_display_name("openai")
        'OpenAI'
        >>> get_provider_display_name("amazon-bedrock")
        'Amazon Bedrock'
    """
    return _PROVIDER_DISPLAY_NAMES.get(provider, provider)
