# pages/Evaluation.py — Dedicated Page for Metrics & Logs
import streamlit as st
import time
import os
import pandas as pd

# ─────────────────────────────────────────────
# LOAD SYSTEMS (Same as Dashboard)
# ─────────────────────────────────────────────
# We need API Key from session state or sidebar, 
# but for simplicity, we ask for it here if not set.
st.set_page_config(page_title="Evaluation", layout="wide")

@st.cache_resource
def load_ml_system_logic():
    try:
        from ml_app import load_structured_docs, HybridRetriever, get_answer
        chunks, metadata = load_structured_docs("data/aws_docs.txt")
        retriever = HybridRetriever(chunks, metadata)
        return retriever, get_answer, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def load_rag_system_logic(api_key):
    if not api_key: return None, None, "API Key missing"
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from groq import Groq
        from app import ask
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if not os.path.exists("vector_store"): return None, None, "vector_store/ missing"
        db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
        client = Groq(api_key=api_key)
        def rag_ask_fn(query): return ask(db, client, query)
        return rag_ask_fn, db, None
    except Exception as e:
        return None, None, str(e)

# Sidebar Settings
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Required for RAG System")

# Load Systems
ml_retriever, ml_get_answer, ml_err = load_ml_system_logic()
rag_ask_fn, rag_db, rag_err = load_rag_system_logic(groq_api_key)

# ─────────────────────────────────────────────
# EVALUATION LOGIC
# ─────────────────────────────────────────────
from evaluate import TEST_QUESTIONS, Evaluator

def run_and_collect(questions):
    evaluator = Evaluator()
    ml_res = []
    rag_res = []
    
    progress_bar = st.progress(0, text="Starting...")
    
    for i, q in enumerate(questions):
        progress_bar.progress((i + 1) / len(questions), text=f"Testing {q['id']}: {q['query'][:30]}...")
        
        # ML Run
        try:
            ans, t, r = ml_get_answer(ml_retriever, q['query'])
            sim = evaluator.semantic_similarity(ans, q['expected'])
            # Check if source matches (Precision)
            sources = [x["metadata"]["service"] for x in r]
            hit = int(any(q['target_service'].lower() in s.lower() for s in sources))
            ml_res.append({
                "ID": q['id'], "System": "ML", "Similarity": sim, "Time": t, 
                "Source Match": "Yes" if hit else "No", "Target": q['target_service']
            })
        except Exception as e:
            ml_res.append({"ID": q['id'], "System": "ML", "Similarity": 0, "Time": 0, "Source Match": "Error"})

        # RAG Run
        if not rag_err:
            try:
                ans, t, docs = rag_ask_fn(q['query'])
                sim = evaluator.semantic_similarity(ans, q['expected'])
                sources = [d.metadata.get("service", "?") for d in docs]
                hit = int(any(q['target_service'].lower() in s.lower() for s in sources))
                rag_res.append({
                    "ID": q['id'], "System": "RAG", "Similarity": sim, "Time": t,
                    "Source Match": "Yes" if hit else "No", "Target": q['target_service']
                })
            except Exception as e:
                rag_res.append({"ID": q['id'], "System": "RAG", "Similarity": 0, "Time": 0, "Source Match": "Error"})
    
    return ml_res, rag_res

# ─────────────────────────────────────────────
# UI DESIGN
# ─────────────────────────────────────────────
st.title("📊 System Evaluation Dashboard")
st.markdown("Run the test suite to generate comparison metrics and detailed logs.")

if st.button("🚀 Run Full Evaluation", use_container_width=True, type="primary"):
    ml_data, rag_data = run_and_collect(TEST_QUESTIONS)
    
    # --- 1. AGGREGATE METRICS (The Big Numbers) ---
    st.subheader("📌 Key Performance Indicators")
    
    # ML Metrics
    ml_avg_sim = sum(d['Similarity'] for d in ml_data) / len(ml_data)
    ml_avg_time = sum(d['Time'] for d in ml_data) / len(ml_data)
    ml_hit_rate = sum(1 for d in ml_data if d['Source Match'] == "Yes") / len(ml_data)
    
    # RAG Metrics
    if not rag_err:
        rag_avg_sim = sum(d['Similarity'] for d in rag_data) / len(rag_data)
        rag_avg_time = sum(d['Time'] for d in rag_data) / len(rag_data)
        rag_hit_rate = sum(1 for d in rag_data if d['Source Match'] == "Yes") / len(rag_data)
    else:
        rag_avg_sim = rag_avg_time = rag_hit_rate = 0

    # Display Metrics Side-by-Side
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("⚡ ML Accuracy (Sim)", f"{ml_avg_sim:.4f}")
        st.metric("⚡ ML Avg Time", f"{ml_avg_time:.4f}s")
        st.metric("🎯 ML Precision (Source Hit)", f"{ml_hit_rate:.1%}")
        
    with col2:
        if not rag_err:
            st.metric("🤖 RAG Accuracy (Sim)", f"{rag_avg_sim:.4f}")
            st.metric("🤖 RAG Avg Time", f"{rag_avg_time:.4f}s")
            st.metric("🎯 RAG Precision (Source Hit)", f"{rag_hit_rate:.1%}")
        else:
            st.warning("RAG System Offline")

    # --- 2. DETAILED LOGS (The Table) ---
    st.subheader("📋 Detailed Log")
    
    # Combine data for table view
    full_df = pd.DataFrame(ml_data + rag_data)
    
    # Style the dataframe for better reading
    st.dataframe(full_df, use_container_width=True, column_config={
        "Similarity": st.column_config.NumberColumn(format="%.4f"),
        "Time": st.column_config.NumberColumn(format="%.4f"),
        "Source Match": st.column_config.TextColumn()
    })
