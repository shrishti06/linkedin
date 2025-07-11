from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def decompose_query(query, api_key):

    template = PromptTemplate.from_template(
        "Decompose this query into 5 meaningful subqueries:\n\nQuery: {query}\n\nSubqueries:"
    )
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        api_key= api_key 
    )
    chain = template | llm | StrOutputParser()
    result = chain.invoke({"query": query})
    subqueries = result.split("\n")
    return subqueries
    # print("Decomposed Subqueries:", subqueries)
    # for i, subquery in enumerate(subqueries, 1):
    #     print(f"{i}. {subquery.strip()}")
