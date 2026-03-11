import sys
import os
sys.path.append(".")

import streamlit as st
from pipeline.rag_chain import ask

st.set_page_config(
    page_title="FinRAG — Apple 10-K Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 FinRAG — Apple 10-K Assistant")
st.caption("Ask any question about Apple's 2023 Annual Report")

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=3, max_value=10, value=5)
    st.divider()
    st.markdown("**Model:** LLaMA 3.3 70B")
    st.markdown("**Document:** Apple 10-K 2023")
    st.markdown("**Chunks:** 626")
    st.divider()
    st.markdown("**Sample Questions:**")
    st.caption("What was Apple's total revenue in 2023?")
    st.caption("How much did Apple spend on R&D?")
    st.caption("What products did Apple sell in 2023?")
    st.caption("What were Apple's operating expenses?")
    st.caption("How many employees does Apple have?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about Apple's 10-K..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            result = ask(prompt, top_k=top_k)

        st.markdown(result["answer"])

        with st.expander("📄 View Source Chunks"):
            for i, source in enumerate(result["sources"]):
                st.markdown(f"**Source {i+1} — Page {source['metadata']['page']}**")
                st.info(source["content"][:400])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"]
        })