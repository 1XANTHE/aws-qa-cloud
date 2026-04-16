# evaluate.py — Evaluation Pipeline
import os
import re
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

TEST_QUESTIONS = [
    # ... (Paste your full list of 25 questions here from your earlier prompt) ...
    # For brevity, I am including the first few, but you should paste the WHOLE list from your previous message.
    {
        "id": "Q01",
        "query": "What is Amazon EC2?",
        "expected": "Amazon EC2 is a cloud computing service that provides scalable virtual servers called instances.",
        "type": "specific",
        "target_service": "Amazon EC2",
        "target_section": "Description"
    },
    # ... ADD ALL OTHER QUESTIONS HERE ...
]

def load_ml_system():
    from ml_app import load_structured_docs, HybridRetriever, get_answer
    chunks, metadata = load_structured_docs("data/aws_docs.txt")
    retriever = HybridRetriever(chunks, metadata)
    def ml_ask(query):
        answer, elapsed, results = get_answer(retriever, query)
        sources = list(set(r["metadata"]["service"] for r in results))
        return answer, elapsed, sources
    return ml_ask

def load_rag_system():
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from groq import Groq
    from app import ask

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists("vector_store"):
        raise FileNotFoundError("vector_store/ not found.")
    db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")
    client = Groq(api_key=api_key)

    def rag_ask(query):
        answer, elapsed, docs = ask(db, client, query)
        sources = list(set(d.metadata.get("service", "?") for d in docs))
        return answer, elapsed, sources
    return rag_ask

class Evaluator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def semantic_similarity(self, answer, expected):
        if not answer or not expected: return 0.0
        emb_a = self.model.encode(answer, convert_to_tensor=True)
        emb_e = self.model.encode(expected, convert_to_tensor=True)
        return util.cos_sim(emb_a, emb_e).item()

def run_evaluation(system_fn, system_name, evaluator, questions, delay=0.5):
    results = []
    print(f"Running {system_name}...")
    for i, q in enumerate(questions):
        try:
            answer, elapsed, sources = system_fn(q["query"])
            sim = evaluator.semantic_similarity(answer, q["expected"])
            results.append({"id": q["id"], "time": elapsed, "similarity": sim, "system": system_name})
        except Exception as e:
            print(f"Error on {q['id']}: {e}")
    return results

def main():
    print("Starting Evaluation...")
    evaluator = Evaluator()
    
    # Load ML
    ml_ask = load_ml_system()
    ml_results = run_evaluation(ml_ask, "ML Model", evaluator, TEST_QUESTIONS)

    # Load RAG (if API key exists)
    if os.environ.get("GROQ_API_KEY"):
        rag_ask = load_rag_system()
        rag_results = run_evaluation(rag_ask, "RAG System", evaluator, TEST_QUESTIONS)
    else:
        rag_results = []
        print("Skipping RAG (No API Key)")

    # Simple printout
    avg_ml = sum(r['similarity'] for r in ml_results) / len(ml_results)
    print(f"ML Avg Accuracy: {avg_ml:.4f}")
    if rag_results:
        avg_rag = sum(r['similarity'] for r in rag_results) / len(rag_results)
        print(f"RAG Avg Accuracy: {avg_rag:.4f}")

if __name__ == "__main__":
    main()
