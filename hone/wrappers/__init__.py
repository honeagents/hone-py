"""
Hone Wrappers Module.

Re-exports LangSmith's auto-instrumentation wrappers for various LLM providers.
These wrappers automatically trace LLM calls without requiring decorators.
"""

# Note: We don't import wrappers at package level to avoid
# requiring optional dependencies. Users should import directly:
#
# from hone.wrappers.openai import wrap_openai
# from hone.wrappers.anthropic import wrap_anthropic

__all__ = []
