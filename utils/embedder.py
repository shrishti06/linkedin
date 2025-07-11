from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import pickle

def init_FAISSDB(documents, persist_dir="vectorstore/faiss_index"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    if os.path.exists(persist_dir):
        os.rmdir(persist_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("vectorstore/faiss_index")
    # with open("vectorstore/faiss_index/vectorstore.pkl", "wb") as f:
    #     pickle.dump(vectorstore, f)
    
    # Save the vectorstore to the specified directory
    vectorstore.save_local(persist_dir)
    print(f"Vectorstore saved to {persist_dir}")
    return vectorstore
