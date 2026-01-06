"""
Nested Traces Example

Demonstrates complex trace hierarchies with the Hone SDK.
Shows how parent-child relationships are preserved across
function calls, including async functions.
"""

import os
import asyncio
from hone import traceable, Client, get_current_run_tree

# Set your API key
os.environ["HONE_API_KEY"] = "hone_xxx"  # Replace with your key
os.environ["HONE_PROJECT"] = "nested-traces-example"


# ============================================================================
# Sync Example: RAG Pipeline
# ============================================================================

@traceable(name="rag-pipeline", run_type="chain")
def rag_pipeline(query: str) -> str:
    """
    A complete RAG pipeline with multiple nested steps.
    Each step becomes a child run in the trace.
    """
    # Step 1: Query understanding
    processed_query = understand_query(query)

    # Step 2: Retrieval (multiple sources)
    docs = retrieve_documents(processed_query)

    # Step 3: Reranking
    ranked_docs = rerank_documents(processed_query, docs)

    # Step 4: Generation
    response = generate_answer(query, ranked_docs)

    return response


@traceable(run_type="chain")
def understand_query(query: str) -> str:
    """Preprocess and understand the user query."""
    # In production: query expansion, entity extraction, etc.
    return query.lower().strip()


@traceable(run_type="retriever")
def retrieve_documents(query: str) -> list[str]:
    """Retrieve relevant documents from multiple sources."""
    # Parallel retrieval from different sources
    web_docs = search_web(query)
    db_docs = search_database(query)
    return web_docs + db_docs


@traceable(run_type="retriever")
def search_web(query: str) -> list[str]:
    """Search the web for relevant content."""
    # In production: call a search API
    return [f"Web result 1 for: {query}", f"Web result 2 for: {query}"]


@traceable(run_type="retriever")
def search_database(query: str) -> list[str]:
    """Search internal database."""
    # In production: query vector database
    return [f"DB result for: {query}"]


@traceable(run_type="chain")
def rerank_documents(query: str, docs: list[str]) -> list[str]:
    """Rerank documents by relevance."""
    # In production: use a reranking model
    return sorted(docs, key=lambda x: len(x), reverse=True)


@traceable(run_type="llm")
def generate_answer(query: str, context: list[str]) -> str:
    """Generate final answer using LLM."""
    # In production: call OpenAI/Anthropic/etc.
    context_str = "\n".join(context)
    return f"Based on the context:\n{context_str}\n\nAnswer to '{query}': [Generated response]"


# ============================================================================
# Async Example: Parallel Processing
# ============================================================================

@traceable(name="async-pipeline", run_type="chain")
async def async_pipeline(queries: list[str]) -> list[str]:
    """
    Process multiple queries in parallel.
    Each parallel task creates its own branch in the trace tree.
    """
    # Process all queries concurrently
    tasks = [process_single_query(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return list(results)


@traceable(run_type="chain")
async def process_single_query(query: str) -> str:
    """Process a single query asynchronously."""
    # Simulate async work
    await asyncio.sleep(0.1)
    return f"Processed: {query}"


# ============================================================================
# Manual Run Tree Management
# ============================================================================

@traceable
def manual_tree_example():
    """
    Demonstrates accessing the current run tree for manual operations.
    """
    # Get the current run context
    run_tree = get_current_run_tree()

    if run_tree:
        print(f"   Current run ID: {run_tree.id}")
        print(f"   Current run name: {run_tree.name}")
        print(f"   Trace ID: {run_tree.trace_id}")

        # You can add metadata to the current run
        run_tree.add_metadata({"custom_field": "custom_value"})
        run_tree.add_tags(["manual-example"])

    return "Done with manual tree operations"


# ============================================================================
# Main
# ============================================================================

def main():
    print("=== Hone Nested Traces Example ===\n")

    # Example 1: RAG Pipeline (sync)
    print("1. RAG Pipeline (sync):")
    result = rag_pipeline("How do I configure authentication?")
    print(f"   Result preview: {result[:100]}...\n")

    # Example 2: Async parallel processing
    print("2. Async Parallel Processing:")
    queries = ["Query A", "Query B", "Query C"]
    results = asyncio.run(async_pipeline(queries))
    print(f"   Results: {results}\n")

    # Example 3: Manual run tree access
    print("3. Manual Run Tree Access:")
    result = manual_tree_example()
    print(f"   {result}\n")

    print("=== All examples completed! ===")
    print("\nCheck your Hone dashboard to see the trace hierarchies.")
    print("You should see:")
    print("  - RAG pipeline with nested retrieval and generation steps")
    print("  - Async pipeline with parallel branches")
    print("  - Manual tree with custom metadata")


if __name__ == "__main__":
    main()
