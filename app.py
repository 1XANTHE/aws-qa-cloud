# app.py — RAG + LLM System (System 1)
import os
import time
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_systems():
    """Load embedding model and FAISS vector store."""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists("vector_store"):
        print("ERROR: vector_store/ not found. Please run ingest.py first.")
        exit(1)

    print("Loading FAISS vector database...")
    db = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        exit(1)

    client = Groq(api_key=api_key)

    return db, client


def retrieve_context(db, query, k=4):
    """Retrieve top-k relevant chunks from FAISS."""
    docs = db.similarity_search(query, k=k)
    return docs


def build_prompt(query, docs):
    """Build a focused prompt for the LLM."""
    context_parts = []
    for doc in docs:
        service = doc.metadata.get("service", "Unknown")
        section = doc.metadata.get("section", "General")
        context_parts.append(f"[{service} — {section}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an AWS documentation assistant. Answer the user's question using ONLY the provided context below.

Instructions:
- Be specific and direct. If the question asks about use cases, list only use cases.
- If the question asks about features, list only features.
- If the question is general, give a concise summary.
- Do not add information not present in the context.

Context:
{context}

Question: {query}

Answer:"""

    return prompt


def ask(db, client, query):
    """Full RAG + LLM pipeline for a single query."""
    start = time.time()

    docs = retrieve_context(db, query, k=4)
    prompt = build_prompt(query, docs)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )

    elapsed = time.time() - start
    answer = response.choices[0].message.content.strip()

    return answer, elapsed, docs


def main():
    print("=" * 55)
    print("  AWS Documentation QA System — RAG + LLM")
    print("=" * 55)

    db, client = load_systems()

    print("\nSystem ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("Goodbye!")
            break

        print("\nSearching documentation...")
        answer, elapsed, docs = ask(db, client, query)

        print(f"\nAnswer:\n{answer}")
        print(f"\n[Retrieved from: {', '.join(set(d.metadata.get('service','?') for d in docs))}]")
        print(f"[Response time: {elapsed:.2f}s]")
        print("\n" + "-" * 55 + "\n")


if __name__ == "__main__":
    main()
