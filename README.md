# Hone SDK (Python)

**Last Updated:** 2026-01-15

Python SDK for the Hone AI Experience Engineering Platform.

## Installation

```bash
pip install hone-sdk
```

## Quick Start

```python
from hone import Hone

# Initialize the client
client = Hone({"api_key": "your-api-key"})

# Fetch and evaluate a prompt
result = await client.prompt("greeting", {
    "default_prompt": "Hello, {{name}}!",
    "params": {"name": "World"},
})
print(result)  # "Hello, World!"

# Track a conversation
await client.track("conversation-id", [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
], {"session_id": "session-123"})
```

## Features

- Prompt management with nested prompt support
- Parameter substitution with `{{variableName}}` syntax
- Conversation tracking
- Graceful fallback to default prompts on API errors

## Configuration

```python
from hone import Hone

client = Hone({
    "api_key": "your-api-key",          # Required
    "base_url": "https://custom.api",   # Optional
    "timeout": 10000,                   # Optional (milliseconds)
})
```

## Environment Variables

- `HONE_API_URL`: Override the API base URL
