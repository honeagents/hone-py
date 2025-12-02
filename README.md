# AIX Python SDK

Track and evaluate your LLM calls with the AIX SDK.

## Installation

```bash
pip install aix-sdk
```

For development:

```bash
pip install aix-sdk[dev]
```

## Quick Start

```python
from aix import AIX

# Initialize the client
aix = AIX(
    api_key="aix_your_api_key",
    project_id="my-project"
)

# Track your LLM calls with the @hone.track() decorator
@hone.track()
def my_llm_function(message: str):
    return openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}]
    )

# Use your function normally - tracking happens automatically
result = my_llm_function("Hello, AI!")

# Don't forget to shutdown when done
aix.shutdown()
```

## Automatic Provider Wrappers

The easiest way to track all your LLM calls is using automatic wrappers. These require zero code changes to your existing logic.

### OpenAI

```python
from openai import OpenAI
from aix import AIX, wrap_openai

aix = AIX(api_key="aix_xxx", project_id="my-project")
client = wrap_openai(OpenAI(), aix_client=aix)

# All calls are now automatically tracked
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Streaming is also supported
for chunk in client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

### Anthropic

```python
from anthropic import Anthropic
from aix import AIX, wrap_anthropic

aix = AIX(api_key="aix_xxx", project_id="my-project")
client = wrap_anthropic(Anthropic(), aix_client=aix)

response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello!"}]
)

# Streaming with context manager
with client.messages.stream(
    model="claude-3-5-sonnet-latest",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello!"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Google Gemini

```python
from google import genai
from aix import AIX, wrap_gemini

aix = AIX(api_key="aix_xxx", project_id="my-project")
client = wrap_gemini(genai.Client(api_key="..."), aix_client=aix)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello!"
)
print(response.text)
```

### LiteLLM (Multi-Provider)

LiteLLM provides a unified interface to 100+ LLM providers. Wrapping LiteLLM tracks calls to all of them.

```python
import litellm
from aix import AIX, wrap_litellm

aix = AIX(api_key="aix_xxx", project_id="my-project")
wrap_litellm(aix)  # Patches litellm globally

# Now all litellm calls are tracked
response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Works with any provider
response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)

response = litellm.completion(
    model="bedrock/anthropic.claude-v2",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Features

- **Zero-latency tracking**: Background thread handles uploads without blocking your code
- **Automatic metadata extraction**: Captures model, tokens, and cost from OpenAI/Anthropic responses
- **Async support**: Works with both sync and async functions
- **Error tracking**: Captures and tracks exceptions
- **Batch uploads**: Efficiently batches calls to minimize API requests
- **Retry logic**: Automatic retries with exponential backoff
- **Automatic wrappers**: Zero-code instrumentation for OpenAI, Anthropic, Gemini, and LiteLLM
- **Streaming support**: Full support for streaming responses

## Configuration

```python
aix = AIX(
    api_key="aix_xxx",           # Required: Your AIX API key
    project_id="my-project",     # Required: Project identifier
    api_url="https://api.hone.ai", # Optional: API endpoint (default: localhost)
    batch_size=100,              # Optional: Calls per batch (default: 100)
    flush_interval=1.0,          # Optional: Seconds between flushes (default: 1.0)
    max_retries=3,               # Optional: Retry attempts (default: 3)
)
```

## Advanced Usage

### Custom Names and Metadata

```python
@hone.track(
    name="customer-support-bot",
    metadata={"version": "1.0", "environment": "production"}
)
def support_bot(query: str):
    # Your LLM logic here
    pass
```

### Async Functions

```python
@hone.track()
async def async_llm_call(prompt: str):
    return await openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
```

### Context Manager

```python
with AIX(api_key="aix_xxx", project_id="my-project") as aix:
    @hone.track()
    def my_function():
        pass

    my_function()
# Automatically cleaned up when exiting context
```

### Manual Flush

```python
# Force upload of all pending calls
aix.flush()

# With timeout
aix.flush(timeout=10.0)
```

## API Reference

### `AIX` Class

Main client class for tracking LLM calls.

#### Methods

- `track(name=None, metadata=None)` - Decorator to track function calls
- `flush(timeout=5.0)` - Force upload of pending calls
- `shutdown(timeout=5.0)` - Gracefully shutdown the client

### `TrackedCall` Model

Data model for tracked calls with the following fields:

- `function_name: str` - Name of the tracked function
- `input: Dict[str, Any]` - Input arguments
- `output: Any` - Function return value
- `duration_ms: int` - Execution time in milliseconds
- `started_at: datetime` - Start timestamp
- `metadata: Dict[str, Any]` - Custom metadata
- `model: str` - LLM model used (auto-extracted)
- `tokens_used: int` - Token count (auto-extracted)
- `error: str` - Error message if exception occurred

## Agent Framework Integrations

### OpenAI Agents SDK

```python
from agents import Agent, Runner, function_tool, set_trace_processors
from aix import AIX
from aix.integrations import OpenAIAgentsTracingProcessor

aix = AIX(api_key="aix_xxx", project_id="my-project")
set_trace_processors([OpenAIAgentsTracingProcessor(aix_client=aix)])

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Weather Agent",
    instructions="Help users with weather information",
    model="o3-mini",
    tools=[get_weather],
)

result = await Runner.run(agent, "What's the weather in San Francisco?")
```

### Claude Agent SDK

```python
from aix import AIX
from aix.integrations import configure_claude_agent_sdk

aix = AIX(api_key="aix_xxx", project_id="my-project")
configure_claude_agent_sdk(aix)

# Now use claude_agent_sdk as normal - tracing is automatic
from claude_agent_sdk import ClaudeSDKClient

client = ClaudeSDKClient()
async for event in client.conversation_stream(messages=[...]):
    print(event)
```

## Development

### Setup

```bash
cd sdk/python
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Run with Coverage

```bash
pytest tests/ --cov=aix --cov-report=html
```

## License

MIT License - See LICENSE file for details.

Portions of this code adapted from [LangSmith SDK](https://github.com/langchain-ai/langsmith-sdk) (MIT License).
