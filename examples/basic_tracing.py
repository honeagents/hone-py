"""
Basic Tracing Example

Demonstrates how to use the @traceable decorator to automatically
track function calls with Hone.
"""

import os
from hone import traceable, Client

# Set your API key (in production, use environment variables)
os.environ["HONE_API_KEY"] = "hone_xxx"  # Replace with your key
os.environ["HONE_PROJECT"] = "basic-tracing-example"


@traceable
def simple_function(x: int, y: int) -> int:
    """A simple function that gets traced."""
    return x + y


@traceable(name="custom-name", tags=["math", "example"])
def function_with_metadata(a: float, b: float) -> float:
    """A function with custom name and tags."""
    return a * b


@traceable
def nested_parent(query: str) -> str:
    """
    Parent function that calls nested children.
    The trace hierarchy is preserved automatically.
    """
    # These nested calls create child runs in the trace
    context = retrieve_context(query)
    response = generate_response(query, context)
    return response


@traceable
def retrieve_context(query: str) -> str:
    """Simulated RAG retrieval - becomes a child run."""
    # In real code, this would query a vector database
    return f"Context for: {query}"


@traceable
def generate_response(query: str, context: str) -> str:
    """Simulated LLM call - becomes a child run."""
    # In real code, this would call an LLM
    return f"Response to '{query}' using '{context}'"


@traceable
def function_with_error(should_fail: bool) -> str:
    """Demonstrates error tracking."""
    if should_fail:
        raise ValueError("This is an intentional error for demonstration")
    return "Success!"


def main():
    print("=== Hone Basic Tracing Example ===\n")

    # Example 1: Simple function tracing
    print("1. Simple function:")
    result = simple_function(5, 3)
    print(f"   simple_function(5, 3) = {result}\n")

    # Example 2: Custom metadata
    print("2. Function with custom name and tags:")
    result = function_with_metadata(2.5, 4.0)
    print(f"   function_with_metadata(2.5, 4.0) = {result}\n")

    # Example 3: Nested traces
    print("3. Nested trace hierarchy:")
    result = nested_parent("How do I reset my password?")
    print(f"   nested_parent(...) = {result}\n")

    # Example 4: Error tracking
    print("4. Error tracking:")
    try:
        function_with_error(should_fail=True)
    except ValueError as e:
        print(f"   Caught expected error: {e}\n")

    # Example 5: Using the client directly
    print("5. Direct client usage:")
    client = Client()
    print(f"   Client API URL: {client.api_url}")
    print(f"   Client connected successfully!\n")

    print("=== All examples completed! ===")
    print("Check your Hone dashboard to see the traces.")


if __name__ == "__main__":
    main()
