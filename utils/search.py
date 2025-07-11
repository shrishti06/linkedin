from langchain import hub
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.ollama import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def init():
    llm = Ollama(model="llama3", temperature=0.2)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.load_local("vectorstore/faiss_index", 
                        embeddings=embedding_model,
                        allow_dangerous_deserialization=True)

    rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt_llama},
    )
    return qa_chain

def RAG(qa_chain:RetrievalQA,query:str):
    # setup llm
    result=qa_chain.invoke({"query": query})
    return result

def query_resumes(query, k=5):
    qa_chain=init()
    answer = RAG(qa_chain,query)
    print(answer)
    # query = "how do you determine the value of an option"
    # answer = RAG(qa_chain,query)
    # pprint(answer)

def retrieve_top_docs(vectorstore, subqueries, k=20):
    all_docs = []
    for q in subqueries:
        results = vectorstore.similarity_search(q, k=k)
        all_docs.extend(results)
    # Optional: dedup by source
    unique_docs = {doc.metadata["source"]: doc for doc in all_docs}
    return list(unique_docs.values())