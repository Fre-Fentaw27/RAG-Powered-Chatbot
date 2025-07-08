# ---------- IMPORTS ----------
import os
import json 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
import sys 
from langchain_core.prompts import PromptTemplate 

# Paths - Adjusted to match your original setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "complaints_index.faiss") 
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.json") 

# LLM Configuration
LLM_MODEL_NAME = "google/flan-t5-small" 
MAX_GENERATION_TOKENS = 300 

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------- STEP 1: RETRIEVER ----------
class ComplaintRetriever:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load FAISS index
        try:
            self.index = faiss.read_index(FAISS_INDEX_PATH)
        except Exception as e:
            print(f"Error loading FAISS index from {FAISS_INDEX_PATH}: {e}")
            print("Please ensure 'embedding_indexing.py' has been run successfully to create 'complaints_index.faiss'.")
            sys.exit(1)

        # Load metadata from JSON file
        try:
            with open(METADATA_PATH, 'r', encoding='utf-8') as f: 
                self.metadata = json.load(f) 
        except FileNotFoundError:
            print(f"Error: Metadata file '{METADATA_PATH}' not found.")
            print("Please ensure 'embedding_indexing.py' has been run successfully to create 'metadata.json'.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error decoding '{METADATA_PATH}': {e}")
            print("The JSON file might be malformed. Please check its content.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while loading metadata from {METADATA_PATH}: {e}")
            sys.exit(1)
    
    def retrieve(self, query: str, k=5):
        """
        Retrieve top-k relevant complaint chunks from the FAISS index.
        The metadata contains the actual text chunk under 'text_chunk'.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0] 
        D, I = self.index.search(np.array([query_embedding]), k)
        
        retrieved_chunks_info = []
        for i, idx in enumerate(I[0]):
            # Ensure the index is valid for the metadata list
            if 0 <= idx < len(self.metadata):
                retrieved_item = self.metadata[idx]
                retrieved_chunks_info.append({
                    # CORRECTED: Access 'text_chunk' as per your metadata.json
                    'text': retrieved_item.get('text_chunk', 'No text available'), 
                    'product': retrieved_item.get('product', 'N/A'),
                    'complaint_id': retrieved_item.get('complaint_id', 'N/A'),
                    # Score from FAISS search (D is distances, lower is better for L2, higher for IP)
                    'score': float(D[0][i]) 
                })
            else:
                print(f"Warning: Retrieved index {idx} is out of bounds for metadata (size {len(self.metadata)}). Skipping.")
        return retrieved_chunks_info

# ---------- STEP 2: PROMPT ENGINEERING ----------
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to help internal teams understand customer pain points by analyzing complaint excerpts.

Use only the information from the context below to answer the question. If the context does not contain enough information, respond with "I don't have enough information."

Context:
{context}

Question:
{question}

Answer:
"""
prompt_template_obj = PromptTemplate.from_template(PROMPT_TEMPLATE) 

# ---------- STEP 3: GENERATOR ----------
class RAGGenerator:
    def __init__(self, llm_model_name: str = LLM_MODEL_NAME):
        self.retriever = ComplaintRetriever()
        
        print(f"Initializing LLM components for model: {llm_model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name).to(device)
            self.model.eval() # Set model to evaluation mode
            print("LLM components initialized successfully.")
        except Exception as e:
            print(f"Error initializing LLM components for {llm_model_name}: {e}")
            print("This could be due to network issues, model not found, or insufficient memory.")
            print("Please check your internet connection and ensure the model name is correct.")
            sys.exit(1) # Exit if LLM initialization fails

    def generate(self, question: str):
        """End-to-end RAG pipeline"""
        contexts = self.retriever.retrieve(question)
        
        # Build context string from retrieved chunks
        # Using top 3 chunks for context, as in friend's generator.py
        context_str = "\n---\n".join(c['text'] for c in contexts[:3]) 
        
        # Format the complete prompt with context and question
        formatted_prompt = prompt_template_obj.format(
            context=context_str,
            question=question
        )
        
        generated_answer = "Error: Could not generate an answer." # Default error message
        try:
            # Prepare inputs for the LLM
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True).to(device)
            
            # Generate output from the LLM
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_GENERATION_TOKENS,
                do_sample=True, # Enable sampling
                top_p=0.9,      # Top-p sampling
                temperature=0.7 # Temperature for creativity
            )
            
            # Decode the generated tokens back to text
            generated_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            print(f"\n--- Error during LLM generation ---")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("This might be a memory issue, a problem with the model's inference, or a corrupted download.")
            # Do not sys.exit(1) here, allow the program to print the error message and continue.
            
        return generated_answer, contexts

# ==================== Helper for Display ====================
def extract_key_phrases(text):
    """Extracts a key phrase from the text for display."""
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if not sentences:
        return text[:100] + '...' if text else '...'
    return sentences[0][:120] + '...'

# ---------- MAIN EXECUTION ----------
def main():
    print("\n" + "="*60)
    print("🔍 CREDITRUST COMPLAINT INSIGHTBOT")
    print("="*60)
    print("Ask about specific customer service issues.")
    print("Type 'exit' to quit.\n")

    # Initialize RAG system with the specified LLM model
    try:
        rag = RAGGenerator(llm_model_name=LLM_MODEL_NAME)
        print("RAG system initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        sys.exit(1) # Exit if RAG system cannot be initialized

    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() == 'exit':
                print("Exiting...")
                break
            if not question:
                continue

            # Retrieve relevant chunks
            chunks = rag.retriever.retrieve(question)

            if not chunks:
                print("No matching complaints found in our database.")
                continue

            print("\n" + "="*60)
            print("📌 TOP RETRIEVED COMPLAINTS:")
            for i, c in enumerate(chunks, 1):
                print(f"{i}. {extract_key_phrases(c['text'])}") 

            print("\n" + "="*60)
            print("💡 LLM-GENERATED INSIGHT")
            print("="*60)
            
            # Generate answer using the LLM
            answer, _ = rag.generate(question) # We already have chunks, so discard the returned ones
            print(answer)
            print("="*60)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
