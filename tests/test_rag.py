import pytest
from src.rag_pipeline_updated import ComplaintRetriever, RAGGenerator
import os
from pathlib import Path

# --- Fixtures ---
# The 'retriever' fixture instantiates ComplaintRetriever with no arguments,
# matching its __init__ method in your source code.
@pytest.fixture
def retriever():
    return ComplaintRetriever()

# The 'rag' fixture no longer passes the retriever, as RAGGenerator instantiates
# its own retriever internally.
@pytest.fixture
def rag():
    return RAGGenerator()

# --- Tests ---
def test_retriever_initialization(retriever):
    """Test if retriever loads data and index properly."""
    assert retriever.index is not None, "FAISS index not loaded."
    assert len(retriever.metadata) > 0, "No metadata loaded."

def test_retrieval_quality(retriever):
    """Test semantic search quality."""
    test_queries = [
        ("unauthorized credit card charge", 3),
        ("mortgage payment issue", 2),
        ("bank account frozen", 1)
    ]
    
    for query, k in test_queries:
        results = retriever.retrieve(query, k=k)
        assert len(results) == k, f"Expected {k} results for '{query}' but got {len(results)}."
        assert all(isinstance(r.score, float) for r in results), f"Scores should be floats for '{query}'."
        assert all(r.score > 0.0 for r in results), f"Low similarity scores for '{query}'."

def test_rag_generation(rag):
    """Test end-to-end generation with different query types."""
    test_cases = [
        ("What are common credit card issues?", ["credit", "card"]),
        ("How to handle unauthorized transactions?", ["unauthorized", "dispute"]),
        ("Mortgage payment problems?", ["mortgage", "payment"])
    ]
    
    for query, keywords in test_cases:
        answer, contexts = rag.generate(query)
        assert isinstance(answer, str), "Answer should be a string."
        assert len(answer) > 20, "Answer is too short."
        assert any(keyword.lower() in answer.lower() for keyword in keywords), f"Missing keywords {keywords} in answer for query '{query}'."
        assert len(contexts) > 0, "No contexts retrieved."

def test_error_handling(rag):
    """Test edge cases and error handling."""
    # Test empty query
    answer, contexts = rag.generate("")
    assert "no relevant complaints found" in answer.lower(), "Should return no complaints message for empty query."
    assert len(contexts) == 0, "Contexts list should be empty."
    
    # Test non-financial query
    answer, contexts = rag.generate("What's the weather today?")
    assert "no relevant complaints found" in answer.lower(), "Should return no complaints message for irrelevant query."
    assert len(contexts) == 0, "Contexts list should be empty."
    
    # Test very long query
    long_query = "credit card " * 50
    answer, contexts = rag.generate(long_query)
    assert len(answer) > 0, "Failed to handle long query."
    assert len(contexts) > 0, "No contexts retrieved for a long query."