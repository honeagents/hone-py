# Hone SDK (Python)

**Last Updated:** 2026-01-09

**AI Experience Engineering Platform** - Track, evaluate, and improve your LLM applications.

Hone is an SDK-first evaluation platform that automatically tracks LLM calls, generates test cases from production failures, and helps non-technical users improve prompts.

## Installation

```bash
pip install hone-sdk
```

With optional provider integrations:

```bash
# OpenAI support
pip install hone-sdk[openai]

# Anthropic support
pip install hone-sdk[anthropic]

# All integrations
pip install hone-sdk[all]
```

## Quick Start

### 1. Set your API key

```bash
export HONE_API_KEY=hone_xxx
```

### 2. Track your LLM calls

```python
from hone import traceable

@traceable
def my_agent(query: str) -> str:
    # Your LLM call here
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

# Every call is now tracked automatically
result = my_agent("What is the capital of France?")
```

### 3. View your traces

Visit [https://honeagents.ai](https://honeagents.ai) to see your traced calls, detected agents, and evaluation results.

## Features

### Automatic Tracing

The `@traceable` decorator captures:
- Function inputs and outputs
- Execution time and latency
- Token usage and costs
- Nested call hierarchies
- Errors and exceptions

```python
from hone import traceable

@traceable(name="customer-support-agent")
def support_agent(user_message: str) -> str:
    # Nested calls are automatically traced
    context = retrieve_context(user_message)
    response = generate_response(user_message, context)
    return response

@traceable
def retrieve_context(query: str) -> str:
    # RAG retrieval
    return vector_db.search(query)

@traceable
def generate_response(query: str, context: str) -> str:
    # LLM generation
    return llm.generate(query, context)
```

### Auto-Instrumentation

Wrap your LLM clients for automatic tracing without decorators:

```python
from hone.wrappers.openai import wrap_openai
import openai

# Wrap the OpenAI client
client = wrap_openai(openai.OpenAI())

# All calls are now traced automatically
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Manual Client Usage

For more control, use the Client directly:

```python
from hone import Client

client = Client()

# Create a run manually
run = client.create_run(
    name="my-pipeline",
    run_type="chain",
    inputs={"query": "Hello"},
)

# ... do work ...

# End the run
client.update_run(
    run.id,
    outputs={"response": "Hi there!"},
    end_time=datetime.now(),
)
```

### Feedback & Evaluation

Record evaluation scores for your runs:

```python
from hone import Client

client = Client()

# Record feedback
client.create_feedback(
    run_id="...",
    key="user_satisfaction",
    score=0.9,
    comment="User seemed happy with the response",
)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HONE_API_KEY` | Your Hone API key | Required |
| `HONE_ENDPOINT` | API endpoint URL | `https://api.honeagents.ai` |
| `HONE_PROJECT` | Project name | `default` |
| `HONE_TRACING` | Enable tracing | `true` |

### Migration from LangSmith

Hone SDK is fully compatible with LangSmith environment variables for easy migration:

```bash
# These still work!
export LANGSMITH_API_KEY=ls_xxx
export LANGSMITH_PROJECT=my-project
```

Priority order: `HONE_*` > `LANGSMITH_*` > `LANGCHAIN_*`

## Advanced Usage

### Async Support

```python
from hone import AsyncClient, traceable
import asyncio

@traceable
async def async_agent(query: str) -> str:
    # Async functions work seamlessly
    response = await async_llm_call(query)
    return response

async def main():
    client = AsyncClient()
    result = await async_agent("Hello!")
```

### Custom Metadata

```python
@traceable(
    name="my-agent",
    metadata={"version": "1.0", "environment": "production"},
    tags=["customer-support", "production"],
)
def my_agent(query: str) -> str:
    return llm.generate(query)
```

### Run Trees (Manual Hierarchy)

```python
from hone import RunTree

with RunTree(name="parent-operation") as parent:
    # Child operations
    with parent.child(name="child-1"):
        do_something()

    with parent.child(name="child-2"):
        do_something_else()
```

## API Reference

### `@traceable`

Decorator to automatically trace function calls.

```python
@traceable(
    name: str = None,           # Custom name (default: function name)
    run_type: str = "chain",    # Run type: chain, llm, tool, etc.
    metadata: dict = None,      # Additional metadata
    tags: list[str] = None,     # Categorization tags
    client: Client = None,      # Custom client instance
    project_name: str = None,   # Override project name
)
```

### `Client`

Main client for Hone API.

```python
client = Client(
    api_url: str = None,    # API endpoint
    api_key: str = None,    # API key
    **kwargs                # Additional LangSmith client options
)

# Methods
client.create_run(...)      # Create a new run
client.update_run(...)      # Update/end a run
client.create_feedback(...) # Record evaluation feedback
client.create_dataset(...)  # Create a test dataset
client.create_example(...)  # Add example to dataset
```

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: [https://docs.honeagents.ai](https://docs.honeagents.ai)
- Issues: [https://github.com/stone-pebble/hone-sdk/issues](https://github.com/stone-pebble/hone-sdk/issues)
- Email: support@honeagents.ai
