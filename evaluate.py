# evaluate.py — Evaluation Pipeline
import os
import re
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

TEST_QUESTIONS = [
    # --- SPECIFIC ---
    {
        "id": "Q01",
        "query": "What is Amazon EC2?",
        "expected": "Amazon EC2 is a cloud computing service that provides scalable virtual servers called instances. It allows on-demand computing without purchasing hardware.",
        "type": "specific",
        "target_service": "Amazon EC2",
        "target_section": "Description"
    },
    {
        "id": "Q02",
        "query": "What are the key features of Amazon S3?",
        "expected": "S3 features include object storage, 11 nines durability, versioning, lifecycle management, encryption, and S3 Intelligent-Tiering.",
        "type": "specific",
        "target_service": "Amazon S3",
        "target_section": "Key Features"
    },
    {
        "id": "Q03",
        "query": "How does AWS Lambda handle scaling?",
        "expected": "Lambda automatically scales by running code in parallel. It handles any number of requests simultaneously with no configuration needed.",
        "type": "specific",
        "target_service": "AWS Lambda",
        "target_section": "Key Features"
    },
    {
        "id": "Q04",
        "query": "What is DynamoDB latency?",
        "expected": "DynamoDB provides single-digit millisecond latency at any scale.",
        "type": "specific",
        "target_service": "Amazon DynamoDB",
        "target_section": "Key Features"
    },
    {
        "id": "Q05",
        "query": "What database engines does Amazon RDS support?",
        "expected": "RDS supports MySQL, PostgreSQL, MariaDB, Oracle, SQL Server, and Amazon Aurora.",
        "type": "specific",
        "target_service": "Amazon RDS",
        "target_section": "Key Features"
    },

    # --- VAGUE ---
    {
        "id": "Q06",
        "query": "lambda",
        "expected": "AWS Lambda is a serverless computing service that runs code without managing servers.",
        "type": "vague",
        "target_service": "AWS Lambda",
        "target_section": "Description"
    },
    {
        "id": "Q07",
        "query": "s3 storage",
        "expected": "Amazon S3 is an object storage service for storing and retrieving any amount of data.",
        "type": "vague",
        "target_service": "Amazon S3",
        "target_section": "Description"
    },
    {
        "id": "Q08",
        "query": "serverless compute",
        "expected": "AWS Lambda is a serverless compute service that runs event-driven code without server management.",
        "type": "vague",
        "target_service": "AWS Lambda",
        "target_section": "Description"
    },
    {
        "id": "Q09",
        "query": "nosql database aws",
        "expected": "Amazon DynamoDB is a managed NoSQL database with single-digit millisecond latency.",
        "type": "vague",
        "target_service": "Amazon DynamoDB",
        "target_section": "Description"
    },
    {
        "id": "Q10",
        "query": "api management",
        "expected": "Amazon API Gateway is a managed service for creating and managing APIs at scale.",
        "type": "vague",
        "target_service": "Amazon API Gateway",
        "target_section": "Description"
    },

    # --- SECTION-TARGETED ---
    {
        "id": "Q11",
        "query": "What are the use cases of AWS Lambda?",
        "expected": "Lambda use cases include backend APIs, real-time file processing, automation tasks, serverless web applications, and image processing.",
        "type": "section_targeted",
        "target_service": "AWS Lambda",
        "target_section": "Use Cases"
    },
    {
        "id": "Q12",
        "query": "What are the use cases of Amazon S3?",
        "expected": "S3 use cases include backup and archival, static website hosting, data lakes, media distribution, and machine learning data storage.",
        "type": "section_targeted",
        "target_service": "Amazon S3",
        "target_section": "Use Cases"
    },
    {
        "id": "Q13",
        "query": "What are the features of DynamoDB?",
        "expected": "DynamoDB features include managed NoSQL, millisecond latency, auto scaling, global tables, DynamoDB Streams, and DAX caching.",
        "type": "section_targeted",
        "target_service": "Amazon DynamoDB",
        "target_section": "Key Features"
    },
    {
        "id": "Q14",
        "query": "Give me the use cases for Amazon EC2",
        "expected": "EC2 use cases include web application hosting, high performance computing, batch processing, and machine learning workloads.",
        "type": "section_targeted",
        "target_service": "Amazon EC2",
        "target_section": "Use Cases"
    },
    {
        "id": "Q15",
        "query": "What can I use API Gateway for?",
        "expected": "API Gateway is used for building REST APIs, connecting frontends with backends, microservices architectures, and WebSocket APIs.",
        "type": "section_targeted",
        "target_service": "Amazon API Gateway",
        "target_section": "Use Cases"
    },

    # --- CROSS-SERVICE ---
    {
        "id": "Q16",
        "query": "How do Lambda and API Gateway work together?",
        "expected": "API Gateway acts as the front door and routes HTTP requests to Lambda functions as the backend compute service.",
        "type": "cross_service",
        "target_service": "Multiple",
        "target_section": "Mixed"
    },
    {
        "id": "Q17",
        "query": "What is the difference between SQS and SNS?",
        "expected": "SQS is a message queue for decoupling services. SNS is a pub-sub notification service for broadcasting messages to multiple subscribers.",
        "type": "cross_service",
        "target_service": "Multiple",
        "target_section": "Mixed"
    },
    {
        "id": "Q18",
        "query": "How can I store files and host a website on AWS?",
        "expected": "Amazon S3 can store files and host static websites. EC2 can host dynamic web applications.",
        "type": "cross_service",
        "target_service": "Multiple",
        "target_section": "Mixed"
    },
    {
        "id": "Q19",
        "query": "What services help with monitoring in AWS?",
        "expected": "Amazon CloudWatch provides monitoring, metrics, logs, and alarms for AWS services.",
        "type": "cross_service",
        "target_service": "Amazon CloudWatch",
        "target_section": "Description"
    },
    {
        "id": "Q20",
        "query": "How do I manage permissions and access in AWS?",
        "expected": "AWS IAM manages users, roles, and permissions using policies following the principle of least privilege.",
        "type": "cross_service",
        "target_service": "AWS IAM",
        "target_section": "Description"
    },

    # --- HARDER/DETAILED ---
    {
        "id": "Q21",
        "query": "What is Redshift used for?",
        "expected": "Amazon Redshift is a data warehouse used for analytics, business intelligence, and querying large datasets using SQL.",
        "type": "specific",
        "target_service": "Amazon Redshift",
        "target_section": "Use Cases"
    },
    {
        "id": "Q22",
        "query": "What is infrastructure as code in AWS?",
        "expected": "AWS CloudFormation is an infrastructure-as-code service that provisions resources using JSON or YAML templates.",
        "type": "specific",
        "target_service": "AWS CloudFormation",
        "target_section": "Description"
    },
    {
        "id": "Q23",
        "query": "How does ECS run containers?",
        "expected": "ECS orchestrates Docker containers using either EC2 instances or Fargate for serverless container execution.",
        "type": "specific",
        "target_service": "Amazon ECS",
        "target_section": "Description"
    },
    {
        "id": "Q24",
        "query": "What is Elastic Beanstalk?",
        "expected": "Elastic Beanstalk is a PaaS service that deploys and manages web applications automatically without infrastructure management.",
        "type": "specific",
        "target_service": "AWS Elastic Beanstalk",
        "target_section": "Description"
    },
    {
        "id": "Q25",
        "query": "How does AWS VPC provide network isolation?",
        "expected": "VPC provides a logically isolated virtual network with subnets, security groups, and network ACLs for traffic control.",
        "type": "specific",
        "target_service": "Amazon VPC",
        "target_section": "Description"
    },
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
