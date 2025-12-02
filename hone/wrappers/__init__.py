"""
Hone SDK Wrappers - Automatic tracing for LLM providers.

Portions of this code adapted from LangSmith SDK
Copyright (c) 2023 LangChain
Licensed under MIT License
https://github.com/langchain-ai/langsmith-sdk

Modifications for Hone Platform:
- Replaced LangSmith client with Hone client
- Added Hone-specific metadata extraction
- Simplified tracing to work with Hone's tracking model

Example Usage:
    ```python
    from hone import Hone
    from hone.wrappers import wrap_openai, wrap_anthropic
    from openai import OpenAI
    from anthropic import Anthropic

    hone = Hone(api_key="hone_xxx", project_id="my-project")

    # Wrap OpenAI client
    openai_client = wrap_openai(OpenAI(), hone_client=hone)

    # Now all calls are automatically tracked
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    ```
"""

from hone.wrappers.openai import wrap_openai
from hone.wrappers.anthropic import wrap_anthropic
from hone.wrappers.gemini import wrap_gemini
from hone.wrappers.litellm import wrap_litellm

__all__ = [
    "wrap_openai",
    "wrap_anthropic",
    "wrap_gemini",
    "wrap_litellm",
]
