def query_resumes(vectorstore, query, k=5):
    results = vectorstore.similarity_search(query, k=k)
    return results
