# linkedin
Loads the resumes from the internal database and build the index using Hugging Face Embeddings and then store them in a local FAISS Vector store.
Takes the user input from the frontend created using Streamlit and then passes that input to an LLM which is powered by Gemini model to decompose that query into 5 different subqueries.
Uses these subqueries to retrieve top 20 resumes relevant to each subquery and cumulatively pass all valid resumes to an LLM to further extract important keywords with respect to resume text, subqueries and user query.
As the final output, we output top 5 relevant resumes back to our frontend along with the relevance score.
Along with this, we use SerpAPI to fetch relevant LinkedIn profiles related to subqueries which are finally displayed to user on the frontend.