# pages/Evaluation.py — Fixed & Expanded Metrics
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
# EVALUATION LOGIC (Fixed & Expanded)
# ─────────────────────────────────────────────
from evaluate import TEST_QUESTIONS, Evaluator

def calculate_f1(similarity, precision):
    """Calculates F1 Score (Harmonic Mean of Accuracy and Precision)."""
    if similarity + precision == 0:
        return 0.0
    return 2 * (similarity * precision) / (similarity + precision)

def run_and_collect(questions):
    evaluator = Evaluator()
    table_data = []
    
    progress_bar = st.progress(0, text="Starting...")
    
    for i, q in enumerate(questions):
        progress_bar.progress((i + 1) / len(questions), text=f"Testing {q['id']}: {q['query'][:30]}...")
        
        # --- 1. Run ML System ---
        ml_sim = 0.0
        ml_time = 0.0
        ml_hit = 0.0 # Precision
        try:
            ans, t, r = ml_get_answer(ml_retriever, q['query'])
            ml_sim = evaluator.semantic_similarity(ans, q['expected'])
            sources = [x["metadata"]["service"] for x in r]
            # Check if source matches target (Precision)
            ml_hit = 1.0 if any(q['target_service'].lower() in s.lower() for s in sources) else 0.0
            ml_time = t
        except Exception as e:
            pass # Keep 0.0
        
        # --- 2. Run RAG System ---
        rag_sim = 0.0
        rag_time = 0.0
        rag_hit = 0.0
        if not rag_err:
            try:
                ans, t, docs = rag_ask_fn(q['query'])
                rag_sim = evaluator.semantic_similarity(ans, q['expected'])
                sources = [d.metadata.get("service", "?") for d in docs]
                rag_hit = 1.0 if any(q['target_service'].lower() in s.lower() for s in sources) else 0.0
                rag_time = t
            except Exception as e:
                pass
        else:
            # If RAG is offline, set to None for table
            rag_sim = rag_time = rag_hit = None

        # --- 3. Create Row for Table ---
        # Using Simple String Columns to avoid "Tuple" error
        row = {
            "Ques": q['id'],
            "LLM+RAG Sim": rag_sim,
            "LLM+RAG Time": rag_time,
            "ML Sim": ml_sim,
            "ML Time": ml_time
        }
        table_data.append(row)
    
    return table_data

# ─────────────────────────────────────────────
# UI DESIGN
# ─────────────────────────────────────────────
st.title("📊 System Evaluation Dashboard")
st.markdown("Run test suite to generate comparison metrics and detailed logs.")

if st.button("🚀 Run Full Evaluation", use_container_width=True, type="primary"):
    raw_data = run_and_collect(TEST_QUESTIONS)
    
    # --- 1. AGGREGATE METRICS (5 Metrics) ---
    st.subheader("📌 Aggregated Performance Indicators")
    
    # --- ML Metrics ---
    ml_sim_vals = [d['ML Sim'] for d in raw_data]
    ml_time_vals = [d['ML Time'] for d in raw_data]
    # We need raw data with hits for F1 calculation
    # Re-calculating hit rate simply as count of 1s in dummy data? 
    # Actually, let's just calculate F1 from the existing values we gathered.
    # Wait, I need raw hit data. Let's modify run_and_collect? No, too complex.
    # Let's just calculate Accuracy, Time, Precision (I'll need to re-run or infer? 
    # Actually, `run_and_collect` above calculates hit but doesn't save it to row to keep table clean.
    # I will calculate F1 inside run_and_collect and save it to table_data?
    # No, let's just stick to Sim and Time for the Table, and use logic for Aggregates.
    
    # For demonstration, let's assume Precision is simply calculated on the fly for metrics.
    # (Correction: I will add logic to fetch hits if needed, but for now let's use Sim/Time mainly)
    
    # Better: Let's calculate Aggregates directly from the list.
    # I need to re-run evaluation logic? No, let's just use what we have.
    # Let's modify run_and_collect slightly in the next step if needed. 
    # For now, I will compute Accuracy, Time, and add 3 more: 
    # 3. Precision (Hard without hits in table, let's assume high precision for demo)
    # 4. F1 Score (Hard)
    # 5. Max Sim
    
    # Since I can't calculate Precision/F1 without hit data in `raw_data` above,
    # I will update `run_and_collect` to return hits too.
    
    # RE-RUNNING DATA COLLECTION WITH HITS FOR METRICS
    ml_data_agg = [] 
    rag_data_agg = []
    
    # Quick re-run for metrics (optimized)
    evaluator = Evaluator()
    for i, q in enumerate(TEST_QUESTIONS):
        # ML
        sim = t = hit = 0.0
        try:
            ans, t, r = ml_get_answer(ml_retriever, q['query'])
            sim = evaluator.semantic_similarity(ans, q['expected'])
            sources = [x["metadata"]["service"] for x in r]
            hit = 1.0 if any(q['target_service'].lower() in s.lower() for s in sources) else 0.0
        except: pass
        ml_data_agg.append({'sim': sim, 'time': t, 'hit': hit})
        
        # RAG
        sim = t = hit = 0.0
        if not rag_err:
            try:
                ans, t, docs = rag_ask_fn(q['query'])
                sim = evaluator.semantic_similarity(ans, q['expected'])
                sources = [d.metadata.get("service", "?") for d in docs]
                hit = 1.0 if any(q['target_service'].lower() in s.lower() for s in sources) else 0.0
            except: pass
        rag_data_agg.append({'sim': sim, 'time': t, 'hit': hit})

    # Calculate 5 Metrics
    def get_metrics(data):
        sims = [d['sim'] for d in data]
        times = [d['time'] for d in data]
        hits = [d['hit'] for d in data]
        
        avg_acc = round(np.mean(sims), 4)
        avg_time = round(np.mean(times), 4)
        prec = round(np.mean(hits), 4)
        f1 = round(2 * (avg_acc * prec) / (avg_acc + prec), 4) if (avg_acc + prec) > 0 else 0.0
        max_acc = round(np.max(sims), 4)
        return avg_acc, avg_time, prec, f1, max_acc

    ml_m = get_metrics(ml_data_agg)
    rag_m = get_metrics(rag_data_agg)

    # Display Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    
    c1.metric("⚡ ML Sim", f"{ml_m[0]:.4f}")
    c2.metric("⚡ ML Time", f"{ml_m[1]:.4f}")
    c3.metric("⚡ ML Prec", f"{ml_m[2]:.1%}")
    c4.metric("⚡ ML F1", f"{ml_m[3]:.4f}")
    c5.metric("⚡ ML Max", f"{ml_m[4]:.4f}")

    st.divider()
    
    if not rag_err:
        c1.metric("🤖 RAG Sim", f"{rag_m[0]:.4f}")
        c2.metric("🤖 RAG Time", f"{rag_m[1]:.4f}")
        c3.metric("🤖 RAG Prec", f"{rag_m[2]:.1%}")
        c4.metric("🤖 RAG F1", f"{rag_m[3]:.4f}")
        c5.metric("🤖 RAG Max", f"{rag_m[4]:.4f}")

    st.divider()

    # --- 2. DETAILED LOGS (Fixed Table) ---
    st.subheader("📋 Detailed Comparison")
    
    # Define simple columns to fix "Tuple" error
    columns = [
        "Ques", "LLM+RAG Sim", "LLM+RAG Time", "ML Sim", "ML Time"
    ]
    
    df = pd.DataFrame(raw_data, columns=columns)
    
    # Style dataframe
    st.dataframe(
        df, 
            use_container_width=True,
            column_config={
                "LLM+RAG Sim": st.column_config.NumberColumn(format="%.4f"),
                "LLM+RAG Time": st.column_config.NumberColumn(format="%.4f"),
                "ML Sim": st.column_config.NumberColumn(format="%.4f"),
                "ML Time": st.column_config.NumberColumn(format="%.4f"),
            }
        )
