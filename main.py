import streamlit as st
import os
import logging
from dotenv import load_dotenv
from utils.loader import load_resumes
# from utils.embedderinmemory import init_faiss_inmemory
from utils.embedder import init_FAISSDB
from utils.search import retrieve_top_docs, query_resumes
from utils.linkedinsearch import search_linkedin
from utils.decompose import decompose_query
import warnings
warnings.filterwarnings("ignore", message="Failed to load GPU Faiss")
load_dotenv()

# logging.basicConfig(filename='app.log', level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

st.title("🔍 Resume Matcher App")

query = st.text_input("Enter your query (e.g. 'Python developer with 5 years experience')")

if st.button("Search") and query:
    
    # logging.debug("linear search initiated")
    # linkedin_results = search_linkedin(query, os.getenv("SERP_API_KEY"))
    # logging.debug("LinkedIn search completed")
    # logging.debug(f"Found {len(linkedin_results)} LinkedIn profiles")
    
            
    # logging.debug("Starting local resume search")
    documents = load_resumes(os.environ.get("RESUME_FOLDER"))
    print("load complete")
    # vectorstore = init_faiss_inmemory(documents)
    vectorstore = init_FAISSDB(documents, os.getenv("FAISS_PERSIST_DIR") or "vectorstore/faiss_index")
    
    print("embedding done")
    # logging.debug("Local vectorstore initialized")
    # logging.debug("Retrieving top documents from local vectorstore")
    subqueries = decompose_query(query,os.getenv("OPENAI_API_KEY"))
    print("subqueries done",subqueries)
    top_docs = retrieve_top_docs(vectorstore, subqueries, k=20)
    print("embedding load")
    # logging.debug(f"Retrieved {len(top_docs)} top documents from local vectorstore")
    # logging.debug("Querying local resumes")
    local_results = query_resumes(vectorstore=vectorstore,query=query)
    print(local_results)
    # logging.debug("Local resume search completed")
    # logging.debug(f"Found {len(local_results)} local resumes matching the query")
    
    tab_linkedin, tab_local = st.tabs(["Linked Results", "Resume Search"])
    linkedin_results = []
    with tab_linkedin:
        st.subheader("LinkedIn Profiles:")
        for profile in linkedin_results:
            st.markdown(f"**Name:** {profile['title']}")
            st.markdown(f"[Profile Link]({profile['link']})")
            st.text(profile['snippet'])
            st.markdown("---")
    with tab_local:
        st.subheader("Top Matching Profiles:")
        st.write(f"Found {len(local_results)} matching profiles in local resumes.")
        for i, doc in enumerate(local_results):
            st.markdown(f"**Source:** {doc.metadata['source']}")
            st.text(doc.page_content[:1000])  # Limit displayed text
            st.markdown("---")
