"""
OpenAI Auto-Instrumentation Example

Demonstrates how to use the wrap_openai function to automatically
trace all OpenAI API calls without decorators.

Requirements:
    pip install hone-sdk[openai]
"""

import os

# Set your API keys
os.environ["HONE_API_KEY"] = "hone_xxx"  # Replace with your Hone key
os.environ["OPENAI_API_KEY"] = "sk-xxx"  # Replace with your OpenAI key
os.environ["HONE_PROJECT"] = "openai-example"

try:
    import openai
    from hone.wrappers.openai import wrap_openai
except ImportError:
    print("This example requires OpenAI. Install with:")
    print("  pip install hone-sdk[openai]")
    exit(1)


def main():
    print("=== Hone OpenAI Auto-Instrumentation Example ===\n")

    # Create and wrap the OpenAI client
    # All subsequent API calls will be automatically traced
    client = wrap_openai(openai.OpenAI())

    print("1. Simple chat completion:")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        max_tokens=50
    )
    print(f"   Response: {response.choices[0].message.content}\n")

    print("2. Chat completion with more context:")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Explain why 2 + 2 = 4"},
            {"role": "assistant", "content": "When you have 2 items and add 2 more..."},
            {"role": "user", "content": "Can you give a simpler explanation?"}
        ],
        max_tokens=100
    )
    print(f"   Response: {response.choices[0].message.content}\n")

    print("3. Streaming response:")
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Count from 1 to 5, one number per line."}
        ],
        max_tokens=50,
        stream=True
    )
    print("   Response: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")

    print("=== All examples completed! ===")
    print("Check your Hone dashboard to see the traced OpenAI calls.")
    print("Each call shows: model, tokens, latency, and full message history.")


if __name__ == "__main__":
    main()
