# pages/Evaluation.py — Evaluation with Merged Headers (As per sketch)
import streamlit as st
import time
import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# LOAD SYSTEMS
# ─────────────────────────────────────────────
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

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
ml_retriever, ml_get_answer, ml_err = load_ml_system_logic()
rag_ask_fn, rag_db, rag_err = load_rag_system_logic(groq_api_key)

# ─────────────────────────────────────────────
# EVALUATION LOGIC (Revised for Table Structure)
# ─────────────────────────────────────────────
from evaluate import TEST_QUESTIONS, Evaluator

def run_and_collect(questions):
    evaluator = Evaluator()
    table_data = [] # Will hold the wide-format rows
    
    progress_bar = st.progress(0, text="Starting...")
    
    for i, q in enumerate(questions):
        progress_bar.progress((i + 1) / len(questions), text=f"Testing {q['id']}: {q['query'][:30]}...")
        
        # --- 1. Run ML System ---
        ml_sim = 0.0
        ml_time = 0.0
        try:
            ans, t, r = ml_get_answer(ml_retriever, q['query'])
            ml_sim = evaluator.semantic_similarity(ans, q['expected'])
            ml_time = t
        except:
            pass # Keep 0.0 if error
        
        # --- 2. Run RAG System ---
        rag_sim = 0.0
        rag_time = 0.0
        if not rag_err:
            try:
                ans, t, docs = rag_ask_fn(q['query'])
                rag_sim = evaluator.semantic_similarity(ans, q['expected'])
                rag_time = t
            except:
                pass # Keep 0.0 if error
        else:
            rag_sim = rag_time = None # If system is offline
        
        # --- 3. Create Row for Table ---
        # Structure: Ques | RAG Sim | RAG Time | ML Sim | ML Time
        row = {
            "Ques": q['id'],
            ("LLM + RAG", "Similarity"): rag_sim,
            ("LLM + RAG", "Time"): rag_time,
            ("ML", "Similarity"): ml_sim,
            ("ML", "Time"): ml_time
        }
        table_data.append(row)
    
    return table_data

# ─────────────────────────────────────────────
# UI DESIGN
# ─────────────────────────────────────────────
st.title("📊 System Evaluation Dashboard")
st.markdown("Run test suite to see results in the requested table format.")

if st.button("🚀 Run Full Evaluation", use_container_width=True, type="primary"):
    # 1. RUN
    raw_data = run_and_collect(TEST_QUESTIONS)
    
    # 2. CREATE DATAFRAME WITH MERGED HEADERS
    # This creates the structure: Ques | [LLM + RAG] (Sim, Time) | [ML] (Sim, Time)
    columns = [
        "Ques",
        ("LLM + RAG", "Similarity"),
        ("LLM + RAG", "Time"),
        ("ML", "Similarity"),
        ("ML", "Time")
    ]
    
    df = pd.DataFrame(raw_data, columns=columns)
    
    # 3. CALCULATE AGGREGATES (For the top cards)
    if not df.empty:
        # Calculate averages, ignoring None if RAG is offline
        ml_avg_sim = df[("ML", "Similarity")].mean()
        ml_avg_time = df[("ML", "Time")].mean()
        
        if rag_err:
            rag_avg_sim = 0
            rag_avg_time = 0
        else:
            # Filter out Nones for calculation
            rag_sim_col = df[("LLM + RAG", "Similarity")].replace(0, np.nan).mean()
            rag_time_col = df[("LLM + RAG", "Time")].replace(0, np.nan).mean()
            rag_avg_sim = rag_sim_col if not pd.isna(rag_sim_col) else 0.0
            rag_avg_time = rag_time_col if not pd.isna(rag_time_col) else 0.0

        # 4. DISPLAY AGGREGATE METRICS
        st.subheader("📌 Aggregated Scores")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ML Avg Sim", f"{ml_avg_sim:.4f}")
        c2.metric("ML Avg Time", f"{ml_avg_time:.4f}s")
        c3.metric("RAG Avg Sim", f"{rag_avg_sim:.4f}")
        c4.metric("RAG Avg Time", f"{rag_avg_time:.4f}s")

        st.divider()

        # 5. DISPLAY TABLE (The Sketch Format)
        st.subheader("📋 Detailed Comparison")
        
        # Format the numbers nicely
        st.dataframe(
            df, 
            use_container_width=True,
            column_config={
                ("LLM + RAG", "Similarity"): st.column_config.NumberColumn(format="%.4f"),
                ("LLM + RAG", "Time"): st.column_config.NumberColumn(format="%.4f"),
                ("ML", "Similarity"): st.column_config.NumberColumn(format="%.4f"),
                ("ML", "Time"): st.column_config.NumberColumn(format="%.4f"),
            }
        )
    else:
        st.warning("No data generated.")
