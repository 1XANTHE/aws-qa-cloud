# ingest.py — Builds the FAISS vector store + S3 Support
import os
import re
import boto3
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────
# S3 CONFIGURATION (Must match ml_app.py)
# ─────────────────────────────────────────────
S3_BUCKET = "aws-qa-dataset-0347" 
S3_FILE_KEY = "aws_docs.txt"
LOCAL_FILE_PATH = "data/aws_docs.txt"

def download_data_from_s3():
    """Downloads data from S3 if local file is missing."""
    try:
        s3 = boto3.client('s3')
        print(f"📡 Connecting to S3 Bucket: {S3_BUCKET}...")
        os.makedirs(os.path.dirname(LOCAL_FILE_PATH), exist_ok=True)
        
        s3.download_file(S3_BUCKET, S3_FILE_KEY, LOCAL_FILE_PATH)
        print(f"✅ Downloaded {S3_FILE_KEY} from S3.")
        return True
    except Exception as e:
        print(f"⚠️  S3 Download failed: {e}")
        return False

def parse_structured_docs(filepath):
    """
    Parse aws_docs.txt into section-aware chunks.
    Each chunk gets metadata: service name + section type.
    """
    # --- S3 CHECK ---
    if not os.path.exists(filepath):
        print(f"🔸 Local file missing. Attempting download from S3...")
        if not download_data_from_s3():
            raise FileNotFoundError(f"Cannot find file locally or on S3.")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by service (## header)
    services = re.split(r"\n## ", text)
    services = [s.strip() for s in services if s.strip()]

    documents = []

    for service_block in services:
        lines = service_block.split("\n")
        service_name = lines[0].strip().lstrip("## ").strip()

        # Split service block into subsections (### header)
        subsections = re.split(r"\n### ", service_block)

        for sub in subsections:
            if not sub.strip():
                continue

            sub_lines = sub.strip().split("\n")
            section_title = sub_lines[0].strip().lstrip("### ").strip()
            section_body = "\n".join(sub_lines[1:]).strip()

            if len(section_body) < 30:
                continue

            # Create a rich chunk with context header so LLM knows what it's reading
            chunk_text = f"Service: {service_name}\nSection: {section_title}\n\n{section_body}"

            documents.append(Document(
                page_content=chunk_text,
                metadata={
                    "service": service_name,
                    "section": section_title,
                    "source": filepath
                }
            ))

    return documents


def main():
    print("=" * 50)
    print("AWS Documentation Ingestion Pipeline (S3 Enabled)")
    print("=" * 50)

    filepath = "data/aws_docs.txt"

    print("\nParsing structured AWS documentation...")
    documents = parse_structured_docs(filepath)
    print(f"  Created {len(documents)} section-aware chunks")

    services_found = list(set(d.metadata["service"] for d in documents))
    sections_found = list(set(d.metadata["section"] for d in documents))
    print(f"  Services: {len(services_found)}")
    print(f"  Section types: {sections_found}")

    print("\nLoading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS vector database...")
    db = FAISS.from_documents(documents, embeddings)

    os.makedirs("vector_store", exist_ok=True)
    db.save_local("vector_store")

    print("\nVector database saved to vector_store/")
    print(f"Total chunks indexed: {len(documents)}")
    print("\nDone! You can now run app.py")


if __name__ == "__main__":
    main()
