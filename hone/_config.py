"""
Hone SDK Configuration Constants.

This module defines the default configuration values for the Hone SDK.
"""

# Default Hone API endpoint
HONE_DEFAULT_ENDPOINT = "https://api.honeagents.ai"

# Environment variable prefix for Hone
HONE_ENV_PREFIX = "HONE"

# Environment variable namespace search order
# HONE_* takes priority, then falls back to LANGSMITH_* for migration ease
ENV_NAMESPACES = ("HONE", "LANGSMITH", "LANGCHAIN")

# SDK version
VERSION = "0.1.0"
