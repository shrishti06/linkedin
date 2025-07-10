import streamlit as st
import os
from dotenv import load_dotenv
from utils.loader import load_resumes
from utils.embedder import create_vectorstore
from utils.search import query_resumes
import pickle

load_dotenv()

st.title("🔍 Resume Matcher App")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if st.button("Load and Index Resumes"):
    with st.spinner("Processing resumes..."):
        resumes = load_resumes("./resumes")
        vectorstore = create_vectorstore(resumes)
        st.session_state.vectorstore = vectorstore
        with open("vectorstore/faiss_index/index.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
        st.success("Resumes indexed successfully!")

query = st.text_input("Enter your query (e.g. 'Python developer with 5 years experience')")

if st.button("Search") and st.session_state.vectorstore:
    results = query_resumes(st.session_state.vectorstore, query)
    st.subheader("Top Matching Profiles:")
    for doc in results:
        st.markdown(f"**Source:** {doc.metadata['source']}")
        st.text(doc.page_content[:1000])  # Limit displayed text
        st.markdown("---")
