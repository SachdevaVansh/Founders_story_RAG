import os 
import streamlit as st 
import requests

from utils.chunking import create_chunks
from utils.completion import generate_completion
from utils.embedding import generate_embeddings
from utils.prompt import build_prompt
from utils.retrieval import create_faiss_index, retrieve_top_k_chunks

# --- CSS: Clean Large Text Layout ---
# --- CSS + Material Icons Fix ---
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">

<style>
    html, body, [class*="st-"] {
        font-family: 'Segoe UI', sans-serif;
        font-size: 18px;
        line-height: 1.7;
    }

    .stTextInput input {
        font-size: 18px !important;
        padding: 12px 10px !important;
    }

    .stButton > button {
        font-size: 17px !important;
        padding: 10px 20px;
        border-radius: 6px;
    }

    .chunk-box {
        background-color: #f0f2f6;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #007BFF;
        border-radius: 6px;
        font-size: 17px;
        color: #111111;
        white-space: pre-wrap;
    }

    .stExpanderContent {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    .stMarkdown h4, .stMarkdown h3, .stMarkdown h2 {
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    h1 {
        font-size: 2.2rem !important;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Title Section ---
st.title("🚀 RAG Chat: The Founder's Story")
st.markdown("### 🧐 Curious about  Startups ?  Ask away!")

# --- Input Section ---
st.markdown("#### 💬 Type your question below:")
query = st.text_input("e.g. What inspired the creation of iNeuron?")

# --- Query Logic ---
if query:
    index, chunk_mapping = create_faiss_index()
    top_k_chunks = retrieve_top_k_chunks(query, index, chunk_mapping)
    prompt = build_prompt(top_k_chunks, query)
    response = generate_completion(prompt)

    # --- Display Answer ---
    st.markdown("---")
    st.subheader("🧠 Answer")
    st.markdown(f"**{response}**")

    # # --- Retrieved Chunks Section ---
    # with st.expander("📄 Retrieved Context Chunks", expanded=False):
    #     for chunk in top_k_chunks:
    #         st.markdown(f"<div class='chunk-box'>{chunk}</div>", unsafe_allow_html=True)
