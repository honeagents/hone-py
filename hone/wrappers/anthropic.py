"""
Hone Anthropic Wrapper.

Provides automatic tracing for Anthropic API calls.
"""

try:
    from langsmith.wrappers import wrap_anthropic
except ImportError:
    def wrap_anthropic(*args, **kwargs):
        raise ImportError(
            "Anthropic wrapper requires 'anthropic' package. "
            "Install with: pip install hone-sdk[anthropic]"
        )

__all__ = ["wrap_anthropic"]
