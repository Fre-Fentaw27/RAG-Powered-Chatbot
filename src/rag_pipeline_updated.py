# ---------- IMPORTS ----------
import os
import json 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys 
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass

# ---------- CONFIGURATION ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "complaints_index.faiss") 
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.json") 

# LLM Configuration
LLM_MODEL_NAME = "google/flan-t5-large"
MAX_GENERATION_TOKENS = 500 

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------- FINANCIAL CLASSES ----------
class FinancialProductType(Enum):
    CREDIT_CARD = "Credit card"
    PERSONAL_LOAN = "Personal loan"
    BNPL = "Buy Now, Pay Later"
    SAVINGS = "Savings account"
    MONEY_TRANSFER = "Money transfers"

@dataclass
class FinancialContext:
    text: str
    product: str
    complaint_id: str
    score: float
    date: Optional[str] = None
    company: Optional[str] = None

# ---------- RETRIEVER CLASS ----------
class ComplaintRetriever:
    def __init__(self):
        print("Initializing retriever...")
        try:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'r', encoding='utf-8') as f: 
                self.metadata = json.load(f)
            print("Retriever initialized successfully")
        except Exception as e:
            print(f"Failed to initialize retriever: {str(e)}")
            sys.exit(1)

    def retrieve(self, query: str, k: int = 5) -> List[FinancialContext]:
        try:
            query_embedding = self.model.encode([query])[0]
            D, I = self.index.search(np.array([query_embedding]), k)
            
            results = []
            for i, idx in enumerate(I[0]):
                if idx >= len(self.metadata):
                    continue
                
                item = self.metadata[idx]
                results.append(FinancialContext(
                    text=item['text_chunk'],
                    product=item['product'],
                    complaint_id=item['complaint_id'],
                    score=float(D[0][i]),
                    date=item.get('date'),
                    company=item.get('company')
                ))
            return results
        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            return []

# ---------- RAG GENERATOR CLASS ----------
class RAGGenerator:
    def __init__(self):
        print("\nInitializing RAG system...")
        self.retriever = ComplaintRetriever()
        
        try:
            print(f"Loading LLM: {LLM_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
            if torch.cuda.is_available():
                self.model = self.model.to(device)
            print("LLM loaded successfully")
        except Exception as e:
            print(f"Failed to load LLM: {str(e)}")
            sys.exit(1)

    def generate(self, question: str) -> tuple:
        try:
            contexts = self.retriever.retrieve(question)
            if not contexts:
                return "No relevant complaints found.", []
            
            context_str = "\n\n".join(
                f"Complaint {i+1} ({ctx.product}, Relevance: {ctx.score:.2f}):\n{ctx.text[:500]}"
                for i, ctx in enumerate(contexts[:3])
            )
            
            prompt = f"""Analyze these financial complaints and answer the question:

Complaints:
{context_str}

Question: {question}

Provide a concise response focusing on:
1. Main issues identified
2. Product categories affected
3. Any patterns observed


Answer:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_GENERATION_TOKENS,
                temperature=0.7
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return answer, contexts
            
        except Exception as e:
            return f"Error generating response: {str(e)}", []

if __name__ == "__main__":
    rag = RAGGenerator()
    while True:
        question = input("\nEnter your question (or 'exit'): ").strip()
        if question.lower() == 'exit':
            break
            
        answer, contexts = rag.generate(question)
        print("\nRESPONSE:", answer)
        if contexts:
            print("\nTop matching complaints:")
            for ctx in contexts[:2]:
                print(f"\n- {ctx.product} (score: {ctx.score:.2f}): {ctx.text[:100]}...")
