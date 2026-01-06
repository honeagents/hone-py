"""
Hone OpenAI Wrapper.

Provides automatic tracing for OpenAI API calls.
"""

try:
    from langsmith.wrappers import wrap_openai
except ImportError:
    def wrap_openai(*args, **kwargs):
        raise ImportError(
            "OpenAI wrapper requires 'openai' package. "
            "Install with: pip install hone-sdk[openai]"
        )

__all__ = ["wrap_openai"]
