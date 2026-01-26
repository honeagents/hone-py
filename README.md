# Hone SDK (Python)

**Last Updated:** 2026-01-26

Python SDK for the Hone AI Experience Engineering Platform.

## Installation

```bash
pip install honeagents-hone
```

## Quick Start

```python
import asyncio
from hone import create_hone_client, AIProvider

async def main():
    # Initialize the client
    hone = create_hone_client({"api_key": "your-api-key"})

    # Fetch an agent with hyperparameters
    agent = await hone.agent("customer-support", {
        "model": "gpt-4o-mini",
        "provider": AIProvider.OPENAI,  # or just "openai"
        "temperature": 0.7,
        "default_prompt": "You are a helpful customer support agent.",
    })

    print(agent["system_prompt"])  # The evaluated prompt
    print(agent["model"])          # "gpt-4o-mini"
    print(agent["temperature"])    # 0.7

    # Track a conversation
    await hone.track("customer-support", [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
    ], {"session_id": "session-123"})

asyncio.run(main())
```

## Features

- **Agent management** with versioned prompts and hyperparameters
- **Zero-friction tracking** - pass native provider formats directly, no conversion needed
- **Tool management** with versioned tool descriptions
- **Parameter substitution** with `{{variableName}}` syntax
- **Nested entities** - agents can reference other agents, tools, or prompts
- **Conversation tracking** with tool call support
- **Tool tracking helpers** for OpenAI, Anthropic, and Google Gemini
- **Input normalizers** for converting provider message formats
- **Provider constants** with type-safe enum values (17 providers)
- **Graceful fallback** to defaults on API errors

## API Reference

### `hone.agent(id, options) -> AgentResult`

Fetches and evaluates an agent by its ID.

```python
agent = await hone.agent("my-agent", {
    # Required
    "model": "gpt-4o-mini",
    "provider": "openai",
    "default_prompt": "You are a {{tone}} assistant.",

    # Optional
    "major_version": 1,
    "name": "My Agent",
    "params": {"tone": "friendly"},
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop_sequences": [],
    "tools": ["get_weather", "search_web"],
    "extra": {"custom_field": "value"},
})

# Result includes:
# - system_prompt: The evaluated prompt string
# - model: LLM model identifier
# - provider: LLM provider
# - temperature, max_tokens, top_p, etc.: Hyperparameters
# - tools: List of allowed tool IDs
# - Any extra fields you provided
```

### `hone.tool(id, options) -> ToolResult`

Fetches and evaluates a tool description by its ID.

```python
tool = await hone.tool("get_weather", {
    "major_version": 1,
    "default_prompt": "Get the current weather for a location.",
})

print(tool["prompt"])  # The evaluated tool description
```

### `hone.prompt(id, options) -> TextPromptResult`

Fetches and evaluates a text prompt by its ID.

```python
prompt = await hone.prompt("tone-guidelines", {
    "default_prompt": "Always be friendly and professional.",
})

print(prompt["text"])  # The evaluated text
```

### `hone.track(id, messages, options)`

Tracks a conversation with tool call support.

```python
await hone.track("my-agent", [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_123", "name": "get_weather", "arguments": '{"location":"SF"}'}
        ]
    },
    {"role": "tool", "content": '{"temp": 72}', "tool_call_id": "call_123"},
    {"role": "assistant", "content": "It's 72°F in San Francisco."},
], {"session_id": "session-123"})
```

## Tool Tracking Helpers

The SDK provides helpers for extracting messages from LLM responses:

### OpenAI

```python
from hone import from_openai, tool_result

# Extract messages from OpenAI response
response = await openai.chat.completions.create(...)
messages.extend(from_openai(response.model_dump()))

# Create tool result messages
for tool_call in response.choices[0].message.tool_calls:
    result = execute_tool(tool_call.function.name, tool_call.function.arguments)
    messages.append(tool_result(tool_call.id, result))
```

### Anthropic

```python
from hone import from_anthropic, tool_result

# Extract messages from Anthropic response
response = await anthropic.messages.create(...)
messages.extend(from_anthropic(response.model_dump()))

# Create tool result messages
for block in response.content:
    if block.type == "tool_use":
        result = execute_tool(block.name, block.input)
        messages.append(tool_result(block.id, result))
```

### Google Gemini

```python
from hone import from_gemini, tool_result

# Extract messages from Gemini response
response = await model.generate_content(...)
messages.extend(from_gemini(response.to_dict()))
```

### Input Normalizers

For normalizing input messages (not responses), use the normalizer functions:

```python
from hone import (
    normalize_openai_messages,
    normalize_anthropic_messages,
    normalize_gemini_contents,
)

# Normalize OpenAI input messages
openai_messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
]
normalized = normalize_openai_messages(openai_messages)

# Normalize Anthropic input messages
anthropic_messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
]
normalized = normalize_anthropic_messages(anthropic_messages)

# Normalize Gemini contents
gemini_contents = [
    {"role": "user", "parts": [{"text": "Hello!"}]},
    {"role": "model", "parts": [{"text": "Hi there!"}]},
]
normalized = normalize_gemini_contents(gemini_contents)
```

## Provider Constants

Use type-safe provider constants:

```python
from hone import AIProvider, is_valid_provider, get_provider_display_name

# Use enum values
config = {
    "provider": AIProvider.OPENAI,  # "openai"
    "model": "gpt-4o",
}

# Validate provider strings
user_input = "openai"
if is_valid_provider(user_input):
    print(f"Valid provider: {user_input}")

# Get display names
get_provider_display_name("openai")  # "OpenAI"
get_provider_display_name("amazon-bedrock")  # "Amazon Bedrock"
```

### Supported Providers

| Provider           | Value             | Display Name        |
| ------------------ | ----------------- | ------------------- |
| OpenAI             | `openai`          | OpenAI              |
| Anthropic          | `anthropic`       | Anthropic           |
| Google AI          | `google`          | Google AI           |
| Google Vertex AI   | `google-vertex`   | Google Vertex AI    |
| Azure OpenAI       | `azure`           | Azure OpenAI        |
| xAI                | `xai`             | xAI                 |
| Mistral AI         | `mistral`         | Mistral AI          |
| Cohere             | `cohere`          | Cohere              |
| Groq               | `groq`            | Groq                |
| Together.ai        | `togetherai`      | Together.ai         |
| Fireworks          | `fireworks`       | Fireworks           |
| DeepInfra          | `deepinfra`       | DeepInfra           |
| DeepSeek           | `deepseek`        | DeepSeek            |
| Cerebras           | `cerebras`        | Cerebras            |
| Perplexity         | `perplexity`      | Perplexity          |
| Amazon Bedrock     | `amazon-bedrock`  | Amazon Bedrock      |
| Baseten            | `baseten`         | Baseten             |

## Nested Entities

Agents can reference other prompts or agents:

```python
agent = await hone.agent("main-agent", {
    "model": "gpt-4o-mini",
    "provider": "openai",
    "default_prompt": """You are a helpful assistant.

{{tone-guidelines}}

{{user-info}}""",
    "params": {
        # Simple string parameter
        "user-info": {
            "default_prompt": "Name: {{name}}\nEmail: {{email}}",
            "params": {
                "name": "John Doe",
                "email": "john@example.com",
            },
        },
        # Nested prompt (fetched from API)
        "tone-guidelines": {
            "major_version": 1,
            "default_prompt": "Always be friendly and professional.",
        },
    },
})
```

## Configuration

```python
from hone import create_hone_client

client = create_hone_client({
    "api_key": "your-api-key",          # Required
    "base_url": "https://custom.api",   # Optional
    "timeout": 10000,                   # Optional (milliseconds)
})
```

## Environment Variables

- `HONE_API_URL`: Override the API base URL (useful for local development)

## Type Definitions

```python
from hone import (
    # Client types
    HoneClient,
    HoneConfig,

    # Provider types
    AIProvider,
    AIProviderValue,
    AI_PROVIDER_VALUES,

    # Agent types
    GetAgentOptions,
    AgentResult,
    Hyperparameters,

    # Tool types
    GetToolOptions,
    ToolResult,

    # Prompt types
    GetTextPromptOptions,
    TextPromptResult,

    # Tracking types
    Message,
    ToolCall,
    TrackConversationOptions,
)
```

## Example: Complete Agent with Tools

```python
import asyncio
from openai import AsyncOpenAI
from hone import create_hone_client, from_openai, tool_result

async def main():
    hone = create_hone_client({"api_key": "your-hone-api-key"})
    openai = AsyncOpenAI()

    # Fetch agent config from Hone
    agent = await hone.agent("support-agent", {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "temperature": 0.7,
        "tools": ["get_weather", "search_knowledge"],
        "default_prompt": "You are a helpful support assistant.",
    })

    # Fetch tool descriptions from Hone
    weather_tool = await hone.tool("get_weather", {
        "default_prompt": "Get current weather for a location.",
    })

    # Build OpenAI tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": weather_tool["prompt"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Run conversation
    session_id = "session-123"
    messages = [
        {"role": "system", "content": agent["system_prompt"]},
        {"role": "user", "content": "What's the weather in San Francisco?"},
    ]

    # Agentic loop
    while True:
        response = await openai.chat.completions.create(
            model=agent["model"],
            temperature=agent["temperature"],
            messages=messages,
            tools=tools,
        )

        # Add assistant message
        messages.extend(from_openai(response.model_dump()))

        # Check for tool calls
        if not response.choices[0].message.tool_calls:
            break

        # Execute tools
        for tc in response.choices[0].message.tool_calls:
            result = {"temperature": 72, "conditions": "sunny"}  # Mock result
            messages.append(tool_result(tc.id, result))

    # Track the conversation
    await hone.track("support-agent", messages, {"session_id": session_id})

    print("Final response:", messages[-1]["content"])

asyncio.run(main())
```
