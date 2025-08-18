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