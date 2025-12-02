"""
Custom exceptions for Hone SDK.

Portions of this code adapted from LangSmith SDK
Copyright (c) 2023 LangChain
Licensed under MIT License
https://github.com/langchain-ai/langsmith-sdk
"""

from __future__ import annotations


class HoneError(Exception):
    """Base exception class for Hone SDK errors."""

    pass


class HoneConnectionError(HoneError):
    """Exception raised when connection to Hone API fails."""

    pass


class HoneValidationError(HoneError):
    """Exception raised when input validation fails."""

    pass


class HoneAuthenticationError(HoneError):
    """Exception raised when authentication fails."""

    pass


class HoneRateLimitError(HoneError):
    """Exception raised when rate limit is exceeded."""

    pass
