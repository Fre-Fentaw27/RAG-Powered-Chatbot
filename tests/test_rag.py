import pytest
from src.rag_pipeline_updated import ComplaintRetriever, RAGGenerator
import os
from pathlib import Path

# --- Fixtures ---
# Fixtures are a great way to provide a consistent environment for your tests.
# The `retriever` fixture instantiates ComplaintRetriever, which should
# automatically load the FAISS index and metadata from the vector_store directory.
@pytest.fixture
def retriever():
    # The CI/CD workflow will have already checked out the vector_store
    # directory due to the `lfs: true` setting.
    return ComplaintRetriever()

# The `rag` fixture instantiates the main RAGGenerator class.
# The tests will confirm that it can access the retriever internally.
@pytest.fixture
def rag():
    return RAGGenerator()

# --- Tests ---
def test_retriever_initialization(retriever):
    """
    Tests if the ComplaintRetriever successfully loads the data and index.
    This test is critical for confirming that the CI/CD pipeline correctly
    downloads the large vector store files via Git LFS. If this test fails,
    it's likely a problem with the LFS setup.
    """
    assert retriever.index is not None, "FAISS index not loaded."
    assert len(retriever.metadata) > 0, "No metadata loaded."
    # You could also add a check here for the file paths, e.g.:
    # assert Path('vector_store/faiss_index.bin').exists(), "FAISS index file is missing."

def test_retrieval_quality(retriever):
    """
    Tests the semantic search functionality with various test queries.
    It checks for the correct number of results and that the scores are
    valid, indicating a successful retrieval.
    """
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

def test_rag_generation_response(rag):
    """
    A simple end-to-end test to ensure the RAG pipeline generates a response.
    It verifies that the final output from the RAGGenerator is a string and is not empty.
    """
    query = "unauthorized credit card charge"
    response = rag.generate_response(query)
    assert isinstance(response, str), "Generated response should be a string."
    assert len(response) > 0, "Generated response should not be empty."
