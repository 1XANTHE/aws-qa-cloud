# dashboard.py — Unified UI for AWS QA System Comparison
# Streamlit app

import streamlit as st
import time
import os
import sys

# ─────────────────────────────────────────────
# CONFIGURATION & PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AWS QA System Comparison",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ AWS Cloud Services QA: RAG vs Custom ML")
st.markdown("""
This dashboard compares two approaches to answering AWS documentation questions:
1. **RAG + LLM:** Uses Groq API and FAISS vector store (Context-aware, Generative).
2. **Custom ML:** Uses Hybrid Retrieval (Semantic + TF-IDF) without external APIs.
""")

# ─────────────────────────────────────────────
# SIDEBAR: API KEY & SETTINGS
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Required for System 1 (RAG)")
    st.caption("If key is empty, System 1 will be disabled.")
    
    st.divider()
    st.info("**Note:** Make sure `vector_store/` exists and `data/aws_docs.txt` is available.")

# ─────────────────────────────────────────────
# LOAD SYSTEMS (CACHED FOR PERFORMANCE)
# ─────────────────────────────────────────────

@st.cache_resource
def load_ml_system_logic():
    """Loads the Custom ML Model (System 2)."""
    try:
        from ml_app import load_structured_docs, HybridRetriever, get_answer
        with st.spinner("Loading ML System (Sentence Transformers + TF-IDF)..."):
            chunks, metadata = load_structured_docs("data/aws_docs.txt")
            retriever = HybridRetriever(chunks, metadata)
        return retriever, get_answer, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def load_rag_system_logic(api_key):
    """Loads the RAG System (System 1)."""
    if not api_key:
        return None, None, "API Key missing"
    
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from groq import Groq
        from app import ask
        
        with st.spinner("Loading RAG System (FAISS + Groq)..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            if not os.path.exists("vector_store"):
                return None, None, "vector_store/ folder not found. Run ingest.py first."
                
            db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
            client = Groq(api_key=api_key)
            
            def rag_ask_fn(query):
                return ask(db, client, query)
                
        return rag_ask_fn, db, None
    except Exception as e:
        return None, None, str(e)

# Load systems
ml_retriever, ml_get_answer, ml_err = load_ml_system_logic()
rag_ask_fn, rag_db, rag_err = load_rag_system_logic(groq_api_key)

# ─────────────────────────────────────────────
# TABS NAVIGATION
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Side-by-Side Comparison", "📊 Automated Evaluation"])

# ─────────────────────────────────────────────
# TAB 1: DUAL QA INTERFACE
# ─────────────────────────────────────────────
with tab1:
    st.subheader("Real-Time Comparison")
    
    # Input Area
    user_query = st.text_area("Enter your AWS Question:", height=100, placeholder="e.g., What are the use cases of AWS Lambda?")
    
    col_run = st.columns(1)[0]
    run_btn = col_run.button("🚀 Get Answers from Both Systems", type="primary", use_container_width=True)
    
    if run_btn and user_query:
        # Create two columns for side-by-side view
        col1, col2 = st.columns(2)
        
        # --- SYSTEM 1: RAG ---
        with col1:
            st.markdown("### System 1: RAG + LLM")
            if rag_err:
                st.error(f"System Offline: {rag_err}")
            else:
                with st.spinner("Generating LLM Answer..."):
                    try:
                        start_time = time.time()
                        answer, elapsed, docs = rag_ask_fn(user_query)
                        end_time = time.time()
                        
                        sources = list(set(d.metadata.get("service", "?") for d in docs))
                        
                        st.success(f"Answer (in {elapsed:.2f}s)")
                        st.write(answer)
                        
                        st.caption(f"**Sources:** {', '.join(sources)}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # --- SYSTEM 2: ML ---
        with col2:
            st.markdown("### System 2: Custom ML Model")
            if ml_err:
                st.error(f"System Offline: {ml_err}")
            else:
                with st.spinner("Retrieving & Extracting..."):
                    try:
                        start_time = time.time()
                        # ml_get_answer returns (answer, elapsed, results)
                        answer, elapsed, results = ml_get_answer(ml_retriever, user_query, verbose=False)
                        end_time = time.time()
                        
                        sources = list(set(r["metadata"]["service"] for r in results))
                        
                        st.success(f"Answer (in {elapsed:.2f}s)")
                        st.write(answer)
                        
                        st.caption(f"**Sources:** {', '.join(sources)}")
                        
                        # Optional: Show retrieval scores for "science" feel
                        with st.expander("🔬 View Retrieval Scores (ML Only)"):
                            for r in results:
                                st.write(f"**{r['metadata']['service']}** — Score: `{r['score']:.3f}`")
                    except Exception as e:
                        st.error(f"Error: {e}")

# ─────────────────────────────────────────────
# TAB 2: EVALUATION DASHBOARD
# ─────────────────────────────────────────────
with tab2:
    st.subheader("Automated Performance Evaluation")
    st.markdown("Run the 25-question test suite to compare metrics.")
    
    eval_btn = st.button("🧪 Run Full Evaluation", type="secondary")
    
    if eval_btn:
        if rag_err:
            st.warning("RAG System is not available. Running ML evaluation only.")
        
        # We need to import the evaluation logic here
        # Note: We might need to slightly modify evaluate.py to work in Streamlit context 
        # or we replicate the core loop here to avoid 'sys.exit' issues.
        
        from evaluate import TEST_QUESTIONS, Evaluator
        from evaluate import compute_summary
        
        evaluator = Evaluator()
        
        # Progress Bar
        progress_bar = st.progress(0, text="Starting evaluation...")
        
        ml_results_eval = []
        rag_results_eval = []
        
        total_questions = len(TEST_QUESTIONS)
        
        for i, q in enumerate(TEST_QUESTIONS):
            progress_bar.progress((i + 1) / total_questions, text=f"Processing {q['id']}: {q['query'][:30]}...")
            
            # 1. Run ML
            try:
                ans, time_t, res = ml_get_answer(ml_retriever, q['query'])
                sim = evaluator.semantic_similarity(ans, q['expected'])
                ml_results_eval.append({
                    "id": q['id'],
                    "time": time_t,
                    "similarity": sim,
                    "system": "ML"
                })
            except:
                ml_results_eval.append({"id": q['id'], "time": 0, "similarity": 0, "system": "ML"})

            # 2. Run RAG (if available)
            if not rag_err:
                try:
                    ans, time_t, docs = rag_ask_fn(q['query'])
                    sim = evaluator.semantic_similarity(ans, q['expected'])
                    rag_results_eval.append({
                        "id": q['id'],
                        "time": time_t,
                        "similarity": sim,
                        "system": "RAG"
                    })
                except:
                    rag_results_eval.append({"id": q['id'], "time": 0, "similarity": 0, "system": "RAG"})
        
        progress_bar.empty()
        st.success("Evaluation Complete!")
        
        # --- DISPLAY RESULTS ---
        st.divider()
        
        # Calculate Averages
        avg_ml_acc = sum(r['similarity'] for r in ml_results_eval) / len(ml_results_eval)
        avg_ml_time = sum(r['time'] for r in ml_results_eval) / len(ml_results_eval)
        
        col_metrics1, col_metrics2 = st.columns(2)
        
        with col_metrics1:
            st.metric("ML Model (Avg Accuracy)", f"{avg_ml_acc:.4f}")
            st.metric("ML Model (Avg Time)", f"{avg_ml_time:.4f} s")
            
        if not rag_err:
            avg_rag_acc = sum(r['similarity'] for r in rag_results_eval) / len(rag_results_eval)
            avg_rag_time = sum(r['time'] for r in rag_results_eval) / len(rag_results_eval)
            
            with col_metrics2:
                st.metric("RAG System (Avg Accuracy)", f"{avg_rag_acc:.4f}")
                st.metric("RAG System (Avg Time)", f"{avg_rag_time:.4f} s")
        
        st.divider()
        st.subheader("Detailed Breakdown")
        
        # Prepare data for table
        eval_data = []
        for r in ml_results_eval:
            row = {"ID": r['id'], "System": "ML", "Similarity": r['similarity'], "Time": r['time']}
            eval_data.append(row)
            
        if not rag_err:
            for r in rag_results_eval:
                row = {"ID": r['id'], "System": "RAG", "Similarity": r['similarity'], "Time": r['time']}
                eval_data.append(row)
                
        st.dataframe(eval_data, use_container_width=True)
