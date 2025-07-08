"""
Text Chunking, Embedding, and Vector Store Indexing for RAG-Powered Chatbot

This script converts cleaned complaint narratives into searchable vectors for semantic search,
and stores the original text chunks along with their metadata.
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def configure_environment():
    """Set up directories"""
    os.makedirs('vector_store', exist_ok=True)

def load_data():
    """Load and verify the preprocessed data"""
    # Ensure this path is correct relative to where you run this script
    df = pd.read_csv('data/preprocessed/filtered_complaints.csv')
    print(f"Loaded {len(df)} complaints from data/preprocessed/filtered_complaints.csv")
    
    # Verify required columns exist
    required_columns = ['cleaned_narrative', 'Complaint ID', 'Product', 
                        'Sub-product', 'Issue', 'Sub-issue', 'Company']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")
    
    # Ensure 'cleaned_narrative' is not empty for any rows
    if df['cleaned_narrative'].isnull().any():
        print("Warning: Some 'cleaned_narrative' entries are NaN. Filling with empty string.")
        df['cleaned_narrative'] = df['cleaned_narrative'].fillna('')
    
    return df

def chunk_text(df):
    """
    Split narratives into chunks with metadata, including the text chunk itself.
    This function is crucial for ensuring the 'text_chunk' key is present in metadata.json.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = [] # Stores the actual text content of each chunk
    all_metadata = [] # Stores metadata for each chunk, including its text

    print("Starting narrative chunking and metadata generation...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking narratives"):
        # Skip if the narrative is empty after cleaning
        if not row['cleaned_narrative'].strip():
            continue

        text_chunks_for_row = text_splitter.split_text(row['cleaned_narrative'])
        
        for chunk in text_chunks_for_row:
            all_chunks.append(chunk)
            all_metadata.append({
                'complaint_id': row['Complaint ID'],
                'product': row['Product'],
                'sub_product': row['Sub-product'],
                'issue': row['Issue'],
                'sub_issue': row['Sub-issue'],
                'company': row['Company'],
                'text_chunk': chunk # <--- CRUCIAL ADDITION: Store the actual text chunk
            })

    print(f"Generated {len(all_chunks)} text chunks.")
    return all_chunks, all_metadata

def generate_embeddings(chunks):
    """Generate embeddings using sentence-transformers model"""
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Process in batches for memory efficiency
    batch_size = 128
    embeddings = []
    
    print("Generating embeddings for text chunks...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding progress"):
        batch = chunks[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    combined_embeddings = np.vstack(embeddings).astype('float32')
    print(f"Generated {len(combined_embeddings)} embeddings with dimension {combined_embeddings.shape[1]}.")
    return combined_embeddings, model

def create_faiss_index(embeddings, metadata):
    """Create and save FAISS index with metadata"""
    dimension = embeddings.shape[1]
    # Using IndexFlatIP for Inner Product (cosine similarity is related to IP of normalized vectors)
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, 'vector_store/complaints_index.faiss')
    
    # Save metadata (which now includes 'text_chunk')
    with open('vector_store/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    print(f"Saved FAISS index with {len(embeddings)} embeddings to vector_store/complaints_index.faiss")
    print(f"Saved metadata for {len(metadata)} entries to vector_store/metadata.json")

def test_retrieval(model, index_path, metadata_path):
    """Test the retrieval system with sample queries"""
    test_queries = [
        "Why are people unhappy with BNPL?",
        "Credit card billing disputes",
        "Problems with money transfers",
        "Unauthorized transactions on my account"
    ]
    
    # Load the FAISS index
    index = faiss.read_index(index_path)
    
    # Load the metadata (which now contains the text chunks)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print("\n--- Testing Retrieval System ---")
    for query in test_queries:
        query_embedding = model.encode([query]).astype('float32') # Ensure query embedding is float32
        
        # Perform search (k=3 for top 3 results)
        D, I = index.search(query_embedding, k=3)
        
        print(f"\nTest query: '{query}'")
        if len(I[0]) == 0:
            print("No results found.")
            continue

        for i, idx in enumerate(I[0]):
            # Ensure idx is within bounds of metadata list
            if idx < len(metadata):
                retrieved_item = metadata[idx]
                print(f"\n--- Result {i+1} (Index: {idx}) ---")
                print(f"Similarity score: {D[0][i]:.4f}")
                print(f"Complaint ID: {retrieved_item.get('complaint_id', 'N/A')}")
                print(f"Product: {retrieved_item.get('product', 'N/A')}")
                # Access 'text_chunk' directly from the metadata
                print(f"Text chunk: {retrieved_item.get('text_chunk', 'No text available')[:200]}...")
            else:
                print(f"\n--- Result {i+1} ---")
                print(f"Invalid index {idx} retrieved. Metadata list has {len(metadata)} items.")

def main():
    configure_environment()
    
    try:
        # Step 1: Load the preprocessed data
        df = load_data()
        
        # Step 2: Chunk the narratives and get both chunks and enriched metadata
        chunks, metadata = chunk_text(df)
        
        # Step 3: Generate embeddings
        embeddings, model = generate_embeddings(chunks)
        
        # Step 4: Create and save FAISS index and the enriched metadata
        create_faiss_index(embeddings, metadata)
        
        # Step 5: Test retrieval using the newly created index and metadata
        test_retrieval(model, 'vector_store/complaints_index.faiss', 'vector_store/metadata.json')
        
        print("\nProcessing complete. Vector store ready for RAG system.")
    except Exception as e:
        print(f"\nError occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        # You might want to exit with an error code in a real script
        # sys.exit(1)

if __name__ == "__main__":
    main()
