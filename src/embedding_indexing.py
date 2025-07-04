"""
Text Chunking, Embedding, and Vector Store Indexing for RAG-Powered Chatbot

This script converts cleaned complaint narratives into searchable vectors for semantic search.
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
import seaborn as sns

def configure_environment():
    """Set up directories and visualization settings"""
    os.makedirs('vector_store', exist_ok=True)
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load and verify the preprocessed data"""
    df = pd.read_csv('data/preprocessed/filtered_complaints.csv')
    print(f"Loaded {len(df)} complaints")
    
    # Verify required columns exist
    required_columns = ['cleaned_narrative', 'Complaint ID', 'Product', 
                       'Sub-product', 'Issue', 'Sub-issue', 'Company']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df

def chunk_text(df):
    """Split narratives into chunks with metadata"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    metadata = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking narratives"):
        text_chunks = text_splitter.split_text(row['cleaned_narrative'])
        
        for chunk in text_chunks:
            chunks.append(chunk)
            metadata.append({
                'complaint_id': row['Complaint ID'],
                'product': row['Product'],
                'sub_product': row['Sub-product'],
                'issue': row['Issue'],
                'sub_issue': row['Sub-issue'],
                'company': row['Company']
            })

    return chunks, metadata

def generate_embeddings(chunks):
    """Generate embeddings using sentence-transformers model"""
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Process in batches for memory efficiency
    batch_size = 128
    embeddings = []
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch = chunks[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings).astype('float32'), model

def create_faiss_index(embeddings, metadata):
    """Create and save FAISS index with metadata"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save index and metadata
    faiss.write_index(index, 'vector_store/complaints_index.faiss')
    with open('vector_store/metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    print(f"Saved FAISS index with {len(embeddings)} embeddings")

def analyze_chunks(chunks):
    """Generate chunk statistics and visualization"""
    chunk_lengths = [len(chunk.split()) for chunk in chunks]
    avg_chunk_length = np.mean(chunk_lengths)
    
    print("\nChunking statistics:")
    print(f"Average chunk length: {avg_chunk_length:.1f} words")
    print(f"Total chunks: {len(chunks)}")
    
    plt.figure()
    sns.histplot(chunk_lengths, bins=30)
    plt.title('Distribution of Chunk Lengths (in words)')
    plt.xlabel('Word Count')
    plt.ylabel('Number of Chunks')
    plt.savefig('vector_store/chunk_length_distribution.png')
    plt.close()

def test_retrieval(model, chunks, metadata):
    """Test the retrieval system with sample queries"""
    test_queries = [
        "Why are people unhappy with BNPL?",
        "Credit card billing disputes",
        "Problems with money transfers"
    ]
    
    # Load the FAISS index
    index = faiss.read_index('vector_store/complaints_index.faiss')
    
    for query in test_queries:
        query_embedding = model.encode([query])
        D, I = index.search(query_embedding, k=3)
        
        print(f"\nTest query: '{query}'")
        for i, idx in enumerate(I[0]):
            print(f"\n--- Result {i+1} ---")
            print(f"Similarity score: {D[0][i]:.4f}")
            print(f"Product: {metadata[idx]['product']}")
            print(f"Text chunk: {chunks[idx][:200]}...")

def main():
    configure_environment()
    
    try:
        # Step 1: Load the preprocessed data
        df = load_data()
        
        # Step 2: Chunk the narratives
        chunks, metadata = chunk_text(df)
        
        # Step 3: Generate embeddings
        embeddings, model = generate_embeddings(chunks)
        
        # Step 4: Create and save FAISS index
        create_faiss_index(embeddings, metadata)
        
        # Step 5: Analyze chunks
        analyze_chunks(chunks)
        
        # Step 6: Test retrieval
        test_retrieval(model, chunks, metadata)
        
        print("\nProcessing complete. Vector store ready for RAG system.")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()