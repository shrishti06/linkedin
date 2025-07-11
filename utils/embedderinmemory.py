import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def init_faiss_inmemory(documents):
    # Step 1: Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Step 2: Extract content and metadata
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    print(f"Total text chunks after splitting: {len(texts)}")

    # Step 3: Embed using HuggingFace
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = embedding_model.embed_documents(texts)
    embeddings = np.array(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # Step 4: Normalize embeddings (for cosine similarity)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print(type(normalized_embeddings))      
    print(normalized_embeddings.dtype)       
    print(normalized_embeddings.shape)
    
    print(f"Embedding shape: {embeddings.shape if isinstance(embeddings, np.ndarray) else 'not numpy array'}")  
    # Step 5: Create FAISS index (cosine similarity via inner product)
    dimension = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    # index.add(normalized_embeddings)

    # Step 6: Create LangChain Document objects
    langchain_docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]

    # Step 7: Create ID mapping and docstore
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(langchain_docs)})
    index_to_docstore_id = {i: str(i) for i in range(len(texts))}

    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    return vectorstore
