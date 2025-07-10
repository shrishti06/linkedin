import streamlit as st
import os
import shutil
import logging
from dotenv import load_dotenv
from utils.loader import load_resumes
from utils.embedder import create_vectorstore
from utils.search import query_resumes
import pickle
from utils.linkedinsearch import search_linkedin

load_dotenv()
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


st.title("🔍 Resume Matcher App")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if st.button("Load and Index Resumes"):
    with st.spinner("Processing resumes..."):
        if os.path.exists("vectorstore/faiss_index"):
            shutil.rmtree("vectorstore/faiss_index") 
        resumes = load_resumes("./resumes")
        vectorstore = create_vectorstore(resumes)
        st.session_state.vectorstore = vectorstore
        os.makedirs("vectorstore/faiss_index", exist_ok=True)
        with open("vectorstore/faiss_index/index.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
        st.success("Resumes indexed successfully!")

query = st.text_input("Enter your query (e.g. 'Python developer with 5 years experience')")

if st.button("Search") and query:
    tab_linkedin, tab_local = st.tabs(["LinkedIn Search", "Local Resume Search"])
    logging.debug("linear search initiated")
    linkedin_results = search_linkedin(query, os.getenv("SERP_API_KEY"))
    logging.debug("LinkedIn search completed")
    logging.debug(f"Found {len(linkedin_results)} LinkedIn profiles")
    
    if not linkedin_results:
        st.warning("No LinkedIn profiles found for the given query.")
            
    logging.debug("Querying local resumes")
    local_results = query_resumes(st.session_state.vectorstore, query)
    logging.debug("Local resume search completed")
    logging.debug(f"Found {len(local_results)} local resumes matching the query")
    
    if not local_results:
        st.warning("No local resumes found for the given query.")
    with tab_linkedin:
        st.subheader("LinkedIn Profiles:")
        for profile in linkedin_results:
            st.markdown(f"**Name:** {profile['title']}")
            st.markdown(f"[Profile Link]({profile['link']})")
            st.text(profile['snippet'])
            st.markdown("---")
    with tab_local:
        st.subheader("Top Matching Profiles:")
        for doc in local_results:
            st.markdown(f"**Source:** {doc.metadata['source']}")
            st.text(doc.page_content[:1000])  # Limit displayed text
            st.markdown("---")
