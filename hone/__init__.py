"""
Hone SDK - AI Experience Engineering Platform.

Hone is an SDK-first evaluation platform that automatically tracks LLM calls,
generates test cases from production failures, and helps improve prompts.

This SDK wraps the LangSmith SDK to redirect all data to Hone's backend
while maintaining full compatibility with LangSmith's battle-tested APIs.

Quick Start:
    ```python
    import os
    from hone import traceable, Client

    # Set your API key
    os.environ["HONE_API_KEY"] = "hone_xxx"

    # Track functions with the decorator
    @traceable
    def my_agent(query: str) -> str:
        # Your LLM call here
        return "response"

    # Or use the client directly
    client = Client()
    ```

Environment Variables:
    HONE_API_KEY: Your Hone API key
    HONE_ENDPOINT: API endpoint (default: https://api.honeagents.ai)
    HONE_PROJECT: Project name for organizing traces
    HONE_TRACING: Enable/disable tracing ("true" or "false")

Migration from LangSmith:
    Simply change your imports from `langsmith` to `hone`.
    LANGSMITH_* environment variables are supported for backward compatibility.
"""

__version__ = "0.1.0"

# Apply patches FIRST before any other imports
# This ensures all subsequent langsmith imports use Hone configuration
from hone import _patch  # noqa: F401

# Re-export client classes
from hone.client import Client, AsyncClient

# Re-export tracing decorators and utilities
from hone.run_helpers import (
    traceable,
    trace,
    get_current_run_tree,
    get_tracing_context,
    RunTree,
)

# Re-export commonly used schemas
from hone.schemas import (
    Run,
    Feedback,
    Dataset,
    Example,
    EvaluationResult,
)

__all__ = [
    # Version
    "__version__",
    # Clients
    "Client",
    "AsyncClient",
    # Tracing
    "traceable",
    "trace",
    "get_current_run_tree",
    "get_tracing_context",
    "RunTree",
    # Schemas
    "Run",
    "Feedback",
    "Dataset",
    "Example",
    "EvaluationResult",
]
