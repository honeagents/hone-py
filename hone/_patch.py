"""
Hone SDK Monkeypatch Module.

This module applies patches to the LangSmith SDK to redirect
all API calls to the Hone backend. It is automatically imported
when the hone package is loaded.

The patches:
1. Override get_api_url() to return Hone endpoint by default
2. Override get_env_var() to check HONE_* env vars first
"""

import functools
import os
from typing import Optional

from hone._config import HONE_DEFAULT_ENDPOINT, ENV_NAMESPACES

# Import langsmith modules to patch
from langsmith import utils as ls_utils


def _get_hone_api_url(api_url: Optional[str] = None) -> str:
    """
    Get the Hone API URL.

    Priority:
    1. Explicit api_url parameter
    2. HONE_ENDPOINT environment variable
    3. LANGSMITH_ENDPOINT environment variable (for migration)
    4. LANGCHAIN_ENDPOINT environment variable
    5. Default Hone endpoint

    Args:
        api_url: Optional explicit API URL

    Returns:
        The API URL to use
    """
    if api_url:
        return api_url

    # Check environment variables in order
    for prefix in ENV_NAMESPACES:
        env_val = os.environ.get(f"{prefix}_ENDPOINT")
        if env_val and env_val.strip():
            return env_val.strip()

    return HONE_DEFAULT_ENDPOINT


# Store reference to original function for potential fallback
_original_get_env_var = None
if hasattr(ls_utils.get_env_var, '__wrapped__'):
    _original_get_env_var = ls_utils.get_env_var.__wrapped__
else:
    _original_get_env_var = ls_utils.get_env_var


@functools.lru_cache(maxsize=100)
def _get_hone_env_var(
    name: str,
    default: Optional[str] = None,
    *,
    namespaces: tuple = ENV_NAMESPACES,
) -> Optional[str]:
    """
    Get an environment variable with Hone namespace priority.

    Searches for environment variables in order:
    1. HONE_<name>
    2. LANGSMITH_<name>
    3. LANGCHAIN_<name>
    4. Returns default

    Args:
        name: The variable name suffix (e.g., "API_KEY")
        default: Default value if not found
        namespaces: Tuple of namespace prefixes to search

    Returns:
        The environment variable value or default
    """
    for namespace in namespaces:
        env_name = f"{namespace}_{name}"
        value = os.environ.get(env_name)
        if value is not None and value.strip() != "":
            return value
    return default


def apply_patches():
    """
    Apply all Hone patches to the LangSmith SDK.

    This function is called automatically on import.
    """
    # Patch get_api_url
    ls_utils.get_api_url = _get_hone_api_url

    # Patch get_env_var and preserve cache_clear method
    ls_utils.get_env_var = _get_hone_env_var

    # Clear any cached values from previous calls
    if hasattr(_get_hone_env_var, 'cache_clear'):
        _get_hone_env_var.cache_clear()


# Apply patches when this module is imported
apply_patches()
