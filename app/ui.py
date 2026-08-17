"""Streamlit chat interface.

Run locally:  streamlit run app/ui.py
In docker-compose this is the `app` service on http://localhost:8501
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app import config, db, rag

st.set_page_config(page_title="Technote Copilot", page_icon=None, layout="wide")

st.title("Technote Copilot")
st.caption(
    "RAG over the TechQA corpus of IBM support technotes. "
    "Ask a technical support question; the assistant answers only from "
    "retrieved documents and says NOT_FOUND when the corpus has no answer."
)

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Retrieval mode", rag.MODES, index=2)
    prompt_version = st.selectbox("Prompt", list(rag.PROMPTS), index=1)
    top_k = st.slider("Documents to retrieve", 1, 10, 5)
    rewrite = st.checkbox("Rewrite query before retrieval", value=False)

# One-time DB init per session. Cheap because CREATE TABLE IF NOT EXISTS.
if "db_ready" not in st.session_state:
    db.init_db()
    st.session_state.db_ready = True

if "history" not in st.session_state:
    st.session_state.history = []


def render_feedback(conversation_id, key):
    col1, col2, _ = st.columns([1, 1, 10])
    if col1.button("Helpful", key=f"up_{key}"):
        db.log_feedback(conversation_id, 1)
        st.toast("Thanks, logged.")
    if col2.button("Not helpful", key=f"down_{key}"):
        db.log_feedback(conversation_id, -1)
        st.toast("Thanks, logged.")


for i, item in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        with st.expander("Sources"):
            for fname in item["sources"]:
                st.code(fname)
        render_feedback(item["conversation_id"], key=i)

question = st.chat_input("Ask about WebSphere, MQ, Tivoli, DataPower, SPSS...")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching technotes..."):
            result = rag.answer(
                question,
                mode=mode,
                prompt_version=prompt_version,
                limit=top_k,
                rewrite=rewrite,
            )
        st.write(result["answer"])
        with st.expander("Sources"):
            for fname in result["sources"]:
                st.code(fname)

        conversation_id = db.log_conversation(result)
        render_feedback(conversation_id, key=len(st.session_state.history))

    st.session_state.history.append(
        {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "conversation_id": conversation_id,
        }
    )
