"""
Hone Python SDK - Track and evaluate your LLM calls.

Portions of this code adapted from LangSmith SDK
Copyright (c) 2023 LangChain
Licensed under MIT License
https://github.com/langchain-ai/langsmith-sdk

Example Usage:
    ```python
    from hone import Hone
    from hone.wrappers import wrap_openai
    from openai import OpenAI

    hone = Hone(api_key="hone_xxx", project_id="my-project")

    # Option 1: Use the @hone.track() decorator
    @hone.track()
    def my_llm_call(message: str):
        return openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message}]
        )

    # Option 2: Use automatic wrappers for zero-code instrumentation
    client = wrap_openai(OpenAI(), hone_client=hone)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    hone.shutdown()
    ```
"""

from hone.client import Hone
from hone.models import TrackedCall
from hone.exceptions import (
    HoneError,
    HoneConnectionError,
    HoneValidationError,
    HoneAuthenticationError,
    HoneRateLimitError,
)

# Import wrappers for easy access
from hone.wrappers import (
    wrap_openai,
    wrap_anthropic,
    wrap_gemini,
    wrap_litellm,
)

__version__ = "0.1.0"

__all__ = [
    # Core client
    "Hone",
    "TrackedCall",
    # Exceptions
    "HoneError",
    "HoneConnectionError",
    "HoneValidationError",
    "HoneAuthenticationError",
    "HoneRateLimitError",
    # Wrappers
    "wrap_openai",
    "wrap_anthropic",
    "wrap_gemini",
    "wrap_litellm",
    # Version
    "__version__",
]
