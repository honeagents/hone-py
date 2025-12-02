"""
Basic Usage Example for AIX Python SDK.

This example demonstrates how to use the AIX SDK to track LLM calls.

Prerequisites:
    pip install aix-sdk openai

Usage:
    export AIX_API_KEY=your_aix_api_key
    export OPENAI_API_KEY=your_openai_api_key
    python basic_usage.py
"""

import os
import time
from typing import List, Dict, Any

# Note: In production, install via: pip install aix-sdk
# For local development, run from the sdk/python directory:
#   pip install -e .
from aix import AIX

# Optional: Import OpenAI for real LLM calls
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("OpenAI not installed. Using mock responses.")


def main():
    """Main example demonstrating AIX SDK usage."""

    # Initialize the AIX client
    aix = AIX(
        api_key=os.getenv("AIX_API_KEY", "aix_test_key"),
        project_id="example-project",
        api_url=os.getenv("AIX_API_URL", "http://localhost:8000"),
    )

    print("AIX SDK initialized!")
    print(f"  Project ID: {aix.project_id}")
    print(f"  API URL: {aix.api_url}")
    print()

    # Example 1: Track a simple function
    print("Example 1: Tracking a simple function")
    print("-" * 40)

    @hone.track()
    def simple_function(message: str) -> str:
        """A simple function that returns a greeting."""
        return f"Hello, {message}!"

    result = simple_function("World")
    print(f"Result: {result}")
    print()

    # Example 2: Track with custom name and metadata
    print("Example 2: Custom name and metadata")
    print("-" * 40)

    @hone.track(
        name="customer-support-bot",
        metadata={"version": "1.0", "environment": "development"}
    )
    def support_bot(query: str) -> Dict[str, Any]:
        """Simulated customer support bot."""
        # Simulate processing time
        time.sleep(0.1)
        return {
            "response": f"Thank you for your question about: {query}",
            "confidence": 0.95,
        }

    result = support_bot("refund policy")
    print(f"Result: {result}")
    print()

    # Example 3: Track OpenAI calls (if available)
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        print("Example 3: Tracking OpenAI calls")
        print("-" * 40)

        openai_client = OpenAI()

        @hone.track(name="openai-chat")
        def chat_with_gpt(user_message: str) -> str:
            """Chat with GPT-4."""
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=100,
            )
            return response.choices[0].message.content

        result = chat_with_gpt("What is the capital of France?")
        print(f"OpenAI Response: {result}")
        print()
    else:
        print("Example 3: Mock LLM call (OpenAI not configured)")
        print("-" * 40)

        @hone.track(name="mock-llm")
        def mock_llm_call(messages: List[Dict[str, str]]) -> Dict[str, Any]:
            """Simulated LLM call with mock response."""
            # Simulate a mock LLM response object
            class MockChoice:
                def __init__(self):
                    self.message = type("Message", (), {"content": "Paris is the capital of France."})()

            class MockUsage:
                prompt_tokens = 15
                completion_tokens = 8
                total_tokens = 23

            class MockResponse:
                def __init__(self):
                    self.choices = [MockChoice()]
                    self.model = "gpt-4o-mini"
                    self.usage = MockUsage()

            time.sleep(0.05)  # Simulate API latency
            return MockResponse()

        result = mock_llm_call([
            {"role": "user", "content": "What is the capital of France?"}
        ])
        print(f"Mock Response: {result.choices[0].message.content}")
        print()

    # Example 4: Async function tracking
    print("Example 4: Async function tracking")
    print("-" * 40)

    import asyncio

    @hone.track(name="async-processor")
    async def async_process(data: str) -> str:
        """Async processing function."""
        await asyncio.sleep(0.05)  # Simulate async I/O
        return f"Processed: {data.upper()}"

    async def run_async():
        result = await async_process("hello async world")
        print(f"Async Result: {result}")

    asyncio.run(run_async())
    print()

    # Example 5: Error handling
    print("Example 5: Error tracking")
    print("-" * 40)

    @hone.track(name="error-prone-function")
    def risky_function(value: int) -> int:
        """Function that might raise an error."""
        if value < 0:
            raise ValueError("Value must be non-negative")
        return value * 2

    try:
        result = risky_function(5)
        print(f"Success: {result}")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        result = risky_function(-1)
    except ValueError as e:
        print(f"Error tracked and re-raised: {e}")
    print()

    # Flush pending calls and shutdown
    print("Flushing pending calls...")
    aix.flush()

    print("Shutting down AIX client...")
    aix.shutdown()

    print("Done! All calls have been tracked.")


if __name__ == "__main__":
    main()
