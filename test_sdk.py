"""
Test script for Hone Python SDK.
Run the mock server first: uvicorn test_server:app --port 8000
Then run this: python test_sdk.py
"""
import os
import time

# Configure to use local mock server
os.environ["HONE_ENDPOINT"] = "http://localhost:8000"
os.environ["HONE_API_KEY"] = "test_key_123"
os.environ["HONE_PROJECT"] = "test-project"
os.environ["HONE_TRACING"] = "true"

# Now import hone (patches apply on import)
from hone import traceable, Client
from hone.schemas import Run


@traceable(name="simple-function")
def simple_function(query: str) -> str:
    """Simple function to test basic tracing."""
    return f"Response to: {query}"


@traceable(name="parent-function")
def parent_function(query: str) -> dict:
    """Parent function that calls child function."""
    child_result = child_function(query)
    return {"parent_result": f"Processed: {child_result}"}


@traceable(name="child-function")
def child_function(query: str) -> str:
    """Child function to test nested tracing."""
    return f"Child processed: {query}"


def test_basic_tracing():
    """Test basic function tracing."""
    print("\n=== Testing Basic Tracing ===")
    result = simple_function("Hello world")
    print(f"Result: {result}")
    # Give background thread time to send
    time.sleep(1)


def test_nested_tracing():
    """Test nested function tracing."""
    print("\n=== Testing Nested Tracing ===")
    result = parent_function("Test nested")
    print(f"Result: {result}")
    # Give background thread time to send
    time.sleep(1)


def test_client_direct():
    """Test using Client directly."""
    print("\n=== Testing Client Directly ===")
    client = Client()
    print(f"Client API URL: {client.api_url}")
    print(f"Client configured successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("Hone Python SDK Test")
    print("=" * 60)
    print(f"HONE_ENDPOINT: {os.environ.get('HONE_ENDPOINT')}")
    print(f"HONE_PROJECT: {os.environ.get('HONE_PROJECT')}")
    print("=" * 60)

    test_client_direct()
    test_basic_tracing()
    test_nested_tracing()

    # Final wait for background thread
    print("\n=== Waiting for background uploads ===")
    time.sleep(3)
    print("\nTest complete! Check mock server output for captured data.")
