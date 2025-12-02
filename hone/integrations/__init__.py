"""
Hone SDK Integrations - Agent framework integrations.

This module provides integrations with popular agent frameworks:
- OpenAI Agents SDK
- Claude Agent SDK (Anthropic)
- LangChain

These integrations provide automatic tracing of agent workflows
without requiring code changes in your agent implementation.
"""

# Import integrations when available
__all__ = []

# OpenAI Agents SDK integration
try:
    from hone.integrations.openai_agents import OpenAIAgentsTracingProcessor
    __all__.append("OpenAIAgentsTracingProcessor")
except ImportError:
    pass

# Claude Agent SDK integration
try:
    from hone.integrations.claude_agents import configure_claude_agent_sdk
    __all__.append("configure_claude_agent_sdk")
except ImportError:
    pass
