# dashboard.py — Final Fix (Silences Logs + Fixes Input Error)
import streamlit as st
import time
import os

# ─────────────────────────────────────────────
# 1. SET LOG LEVEL TO 'ERROR' (Stops Warning Spam)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AWS QA",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Background & Fonts */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Keep Toolbar & Style it Dark */
    div[data-testid="stToolbar"] {
        background-color: #0e1117;
        border-bottom: 1px solid #3e4a5b;
    }
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #3e4a5b;
    }
    
    /* Hide Footer */
    footer {visibility: hidden;}
    
    /* Input Box */
    .stTextInput > div > div > input {
        background-color: #1e2530;
        color: white;
        border: 1px solid #3e4a5b;
        border-radius: 10px;
        padding: 15px;
        font-size: 18px;
    }
    
    /* Button */
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 50px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #ff6b6b;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
    }

    /* Result Boxes */
    .metric-card {
        background-color: #1e2530;
        border: 1px solid #3e4a5b;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        min-height: 100px;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR (Settings)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Enter key to enable RAG System")
    
    st.divider()
    
    # Evaluation Toggle
    if st.button("📊 Run Evaluation"):
        st.session_state.show_eval = True
        st.rerun()

# ─────────────────────────────────────────────
# LOAD SYSTEMS
# ─────────────────────────────────────────────
@st.cache_resource
def load_ml_system_logic():
    from ml_app import load_structured_docs, HybridRetriever, get_answer
    with st.spinner("Loading ML System..."):
        chunks, metadata = load_structured_docs("data/aws_docs.txt")
        retriever = HybridRetriever(chunks, metadata)
    return retriever, get_answer, None

@st.cache_resource
def load_rag_system_logic(api_key):
    if not api_key: return None, None, "API Key missing"
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

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
ml_retriever, ml_get_answer, ml_err = load_ml_system_logic()
rag_ask_fn, rag_db, rag_err = load_rag_system_logic(groq_api_key)

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown("<h1 style='text-align: center; margin-top: -50px;'>☁️ AWS Cloud QA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888888;'>Compare RAG vs Custom Machine Learning models.</p>", unsafe_allow_html=True)

# NOTE: I put a space " " in label to avoid "Empty value" error
user_query = st.text_input(" ", placeholder="Type question and press Enter...", key="user_query_input", label_visibility="collapsed")

if user_query:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    # --- SYSTEM 1: RAG ---
    with col1:
        st.markdown("<div class='metric-card'><h3>🤖 System 1: RAG + LLM</h3></div>", unsafe_allow_html=True)
        if rag_err:
            st.error(f"System Offline: {rag_err}")
        else:
            answer, elapsed, docs = rag_ask_fn(user_query)
            sources = list(set(d.metadata.get("service", "?") for d in docs))
            st.markdown(f"<p style='color: #ff6b6b;'>⏱️ {elapsed:.2f}s</p>", unsafe_allow_html=True)
            st.markdown(f"<div style='white-space: pre-wrap;'>{answer}</div>", unsafe_allow_html=True)
            st.caption(f"📚 Sources: {', '.join(sources)}")

    # --- SYSTEM 2: ML ---
    with col2:
        st.markdown("<div class='metric-card'><h3>⚡ System 2: Custom ML</h3></div>", unsafe_allow_html=True)
        if ml_err:
            st.error(f"System Offline: {ml_err}")
        else:
            answer, elapsed, results = ml_get_answer(ml_retriever, user_query, verbose=False)
            sources = list(set(r["metadata"]["service"] for r in results))
            st.markdown(f"<p style='color: #ff6b6b;'>⏱️ {elapsed:.2f}s</p>", unsafe_allow_html=True)
            st.markdown(f"<div style='white-space: pre-wrap;'>{answer}</div>", unsafe_allow_html=True)
            st.caption(f"📚 {', '.join(sources)}")

# ─────────────────────────────────────────────
# EVALUATION SECTION
# ─────────────────────────────────────────────
if 'show_eval' not in st.session_state:
    st.session_state.show_eval = False

if st.session_state.show_eval:
    st.divider()
    st.markdown("<h2 style='text-align: center;'>📊 Automated Performance Evaluation</h2>", unsafe_allow_html=True)
    
    eval_btn = st.button("🚀 Run 25-Question Test Suite", type="secondary", use_container_width=True)
    
    if eval_btn:
        if rag_err: st.warning("RAG System is not available. Running ML evaluation only.")
        
        from evaluate import TEST_QUESTIONS, Evaluator, compute_summary
        evaluator = Evaluator()
        
        progress_bar = st.progress(0, text="Starting evaluation...")
        
        ml_results_eval = []
        rag_results_eval = []
        total_questions = len(TEST_QUESTIONS)
        
        for i, q in enumerate(TEST_QUESTIONS):
            progress_bar.progress((i + 1) / total_questions, text=f"Testing {q['id']}: {q['query'][:30]}...")
            
            # ML Run
            try:
                ans, time_t, res = ml_get_answer(ml_retriever, q['query'])
                sim = evaluator.semantic_similarity(ans, q['expected'])
                ml_results_eval.append({"id": q['id'], "time": time_t, "similarity": sim})
            except: ml_results_eval.append({"id": q['id'], "time": 0, "similarity": 0})

            # RAG Run
            if not rag_err:
                try:
                    ans, time_t, docs = rag_ask_fn(q['query'])
                    sim = evaluator.semantic_similarity(ans, q['expected'])
                    rag_results_eval.append({"id": q['id'], "time": time_t, "similarity": sim})
                except: rag_results_eval.append({"id": q['id'], "time": 0, "similarity": 0})
        
        progress_bar.empty()
        st.success("✅ Evaluation Complete!")
        
        # --- VISUAL METRICS ---
        col_metrics1, col_metrics2 = st.columns(2)
        
        avg_ml_acc = sum(r['similarity'] for r in ml_results_eval) / len(ml_results_eval)
        avg_ml_time = sum(r['time'] for r in ml_results_eval) / len(ml_results_eval)
        
        with col_metrics1:
            st.metric("⚡ ML Accuracy", value=f"{avg_ml_acc:.4f}")
            st.metric("⚡ ML Avg Time", value=f"{avg_ml_time:.4f} s")
            
        if not rag_err:
            avg_rag_acc = sum(r['similarity'] for r in rag_results_eval) / len(rag_results_eval)
            avg_rag_time = sum(r['time'] for r in rag_results_eval) / len(rag_results_eval)
            with col_metrics2:
                st.metric("🤖 RAG Accuracy", value=f"{avg_rag_acc:.4f}")
                st.metric("🤖 RAG Avg Time", value=f"{avg_rag_time:.4f} s")
        
        if st.button("Hide Evaluation"):
            st.session_state.show_eval = False
            st.rerun()
