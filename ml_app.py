# ml_app.py — Custom ML-Based QA System (System 2) + S3 Integration
import re
import time
import os
import numpy as np
import boto3
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# S3 CONFIGURATION
# ─────────────────────────────────────────────
# CHANGE THIS TO YOUR BUCKET NAME
S3_BUCKET = "aws-qa-dataset-yourname" 
S3_FILE_KEY = "aws_docs.txt"
LOCAL_FILE_PATH = "data/aws_docs.txt"

def download_data_from_s3():
    """
    Downloads data from S3 if local file is missing.
    Uses default AWS credentials (Environment variables or IAM Role).
    """
    try:
        s3 = boto3.client('s3')
        print(f"📡 Connecting to S3 Bucket: {S3_BUCKET}...")
        # Create directory if not exists
        os.makedirs(os.path.dirname(LOCAL_FILE_PATH), exist_ok=True)
        
        s3.download_file(S3_BUCKET, S3_FILE_KEY, LOCAL_FILE_PATH)
        print(f"✅ Successfully downloaded {S3_FILE_KEY} from S3.")
        return True
    except Exception as e:
        print(f"⚠️  S3 Download failed: {e}")
        print("🔹 Attempting to use local file if available...")
        return False

# ─────────────────────────────────────────────
# SECTION 1 — STRUCTURED DATA LOADING
# ─────────────────────────────────────────────

def load_structured_docs(filepath):
    """Parse aws_docs.txt into section-aware chunks with metadata."""
    
    # --- S3 CHECK: If local file missing, try S3 ---
    if not os.path.exists(filepath):
        print(f"🔸 Local file '{filepath}' not found.")
        success = download_data_from_s3()
        if not success:
            # If S3 failed and file still doesn't exist, raise error
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Could not load data from S3 or local path: {filepath}")

    # --- NORMAL PARSING ---
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    services = re.split(r"\n## ", text)
    services = [s.strip() for s in services if s.strip()]

    chunks = []
    metadata = []

    for service_block in services:
        lines = service_block.split("\n")
        service_name = lines[0].strip().lstrip("## ").strip()

        subsections = re.split(r"\n### ", service_block)

        for sub in subsections:
            if not sub.strip():
                continue
            sub_lines = sub.strip().split("\n")
            section_title = sub_lines[0].strip().lstrip("### ").strip()
            section_body = "\n".join(sub_lines[1:]).strip()

            if len(section_body) < 30:
                continue

            if section_title.startswith("Source:") or section_title.startswith("##"):
                continue

            chunks.append(section_body)
            metadata.append({
                "service": service_name,
                "section": section_title
            })

    return chunks, metadata

# ─────────────────────────────────────────────
# SECTION 2 — QUERY ENHANCEMENT
# ─────────────────────────────────────────────

QUERY_EXPANSION = {
    "use": "use cases applications examples",
    "uses": "use cases applications examples",
    "usage": "use cases applications examples",
    "feature": "features capabilities functions",
    "features": "features capabilities functions",
    "what is": "description overview definition",
    "explain": "description overview definition",
    "difference": "comparison versus vs",
    "store": "storage object data",
    "compute": "virtual servers instances processing",
    "serverless": "lambda no server event-driven",
    "database": "database storage NoSQL SQL",
    "monitor": "monitoring metrics logs alarms",
    "secure": "security authentication authorization IAM",
}

SECTION_INTENT = {
    "use case": "Use Cases",
    "use cases": "Use Cases",
    "uses": "Use Cases",
    "feature": "Key Features",
    "features": "Key Features",
    "what is": "Description",
    "explain": "Description",
}

def enhance_query(query):
    original = query.strip()
    q = original.lower()
    expanded_terms = []
    for trigger, expansion in QUERY_EXPANSION.items():
        if trigger in q:
            expanded_terms.append(expansion)
    if len(q.split()) <= 2:
        q = "explain about " + q
    if expanded_terms:
        q = q + " " + " ".join(expanded_terms)
    return q

def detect_section_intent(query):
    q = query.lower()
    for trigger, section in SECTION_INTENT.items():
        if trigger in q:
            return section
    return None

# ─────────────────────────────────────────────
# SECTION 3 — HYBRID RETRIEVAL
# ─────────────────────────────────────────────

class HybridRetriever:
    """Combines Semantic (Sentence Transformers) + TF-IDF retrieval."""

    def __init__(self, chunks, metadata):
        self.chunks = chunks
        self.metadata = metadata
        print("Loading semantic embedding model...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Encoding all document chunks...")
        self.chunk_embeddings = self.embed_model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
        print("Building TF-IDF index...")
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(chunks)
        print("Retrieval system ready.\n")

    def retrieve(self, query, top_k=2, section_filter=None, semantic_weight=0.8):
        """Retrieval logic with optimized defaults."""
        enhanced_q = enhance_query(query)

        # Semantic scores
        q_embed = self.embed_model.encode(enhanced_q, convert_to_tensor=True)
        semantic_scores = util.cos_sim(q_embed, self.chunk_embeddings)[0].cpu().numpy()

        # TF-IDF scores
        q_tfidf = self.tfidf.transform([enhanced_q])
        tfidf_scores = cosine_similarity(q_tfidf, self.tfidf_matrix)[0]

        # Normalize
        def normalize(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-9: return arr
            return (arr - mn) / (mx - mn)

        sem_norm = normalize(semantic_scores)
        tfidf_norm = normalize(tfidf_scores)

        hybrid_scores = (semantic_weight * sem_norm) + ((1 - semantic_weight) * tfidf_norm)

        if section_filter:
            for i, meta in enumerate(self.metadata):
                if meta["section"].lower() == section_filter.lower():
                    hybrid_scores[i] *= 1.4

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "metadata": self.metadata[idx],
                "score": float(hybrid_scores[idx]),
                "semantic_score": float(sem_norm[idx]),
                "tfidf_score": float(tfidf_norm[idx])
            })

        return results

# ─────────────────────────────────────────────
# SECTION 4 — EXTRACTIVE ANSWER
# ─────────────────────────────────────────────

def extract_answer(query, results, section_intent=None):
    q_lower = query.lower()
    answer_parts = []

    for result in results:
        chunk = result["chunk"]
        service = result["metadata"]["service"]
        section = result["metadata"]["section"]

        if section_intent and section.lower() == section_intent.lower():
            answer_parts.append(f"**{service} — {section}:**\n{chunk}")
            continue

        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)-\s*', chunk)
        relevant = []
        query_words = set(re.sub(r'[^\w\s]', '', q_lower).split())
        query_words -= {'what', 'is', 'are', 'the', 'a', 'an', 'of', 'for', 'how', 'why', 'does'}

        for sent in sentences:
            sent_lower = sent.lower()
            match_count = sum(1 for w in query_words if w in sent_lower)
            if match_count >= 1 or len(relevant) == 0:
                relevant.append(sent.strip())
            if len(relevant) >= 5: break

        if relevant:
            answer_parts.append(f"**{service} — {section}:**\n" + "\n".join(relevant))

    return "\n\n".join(answer_parts) if answer_parts else "No relevant information found."

# ─────────────────────────────────────────────
# SECTION 5 — MAIN QA FUNCTION
# ─────────────────────────────────────────────

def get_answer(retriever, query, top_k=3, verbose=False):
    """Full ML pipeline for a single query."""
    start = time.time()
    section_intent = detect_section_intent(query)
    # Pass top_k=2 for precision
    results = retriever.retrieve(query, top_k=2, section_filter=section_intent) 
    answer = extract_answer(query, results, section_intent)
    elapsed = time.time() - start

    if verbose:
        print(f"\n[Query enhanced to: '{enhance_query(query)}']")
        print(f"[Section intent detected: {section_intent}]")
        if results:
            print(f"[Top result: {results[0]['metadata']['service']} — {results[0]['metadata']['section']} (score: {results[0]['score']:.3f})]")

    return answer, elapsed, results

def main():
    print("=" * 55)
    print("  AWS Documentation QA — Advanced ML Model")
    print("=" * 55)

    # S3 Logic is inside load_structured_docs now
    chunks, metadata = load_structured_docs("data/aws_docs.txt")
    retriever = HybridRetriever(chunks, metadata)

    print("Type 'exit' to quit | Type 'verbose on' to see retrieval details\n")
    verbose = False

    while True:
        query = input("Ask a question: ").strip()
        if not query: continue
        if query.lower() == "exit": break
        if query.lower() == "verbose on": verbose = True; continue
        if query.lower() == "verbose off": verbose = False; continue

        answer, elapsed, results = get_answer(retriever, query, verbose=verbose)
        print(f"\nAnswer:\n{answer}")
        print(f"\n[Sources: {', '.join(set(r['metadata']['service'] for r in results))}]")
        print(f"[Response time: {elapsed:.4f}s]")
        print("\n" + "-" * 55 + "\n")

if __name__ == "__main__":
    main()
