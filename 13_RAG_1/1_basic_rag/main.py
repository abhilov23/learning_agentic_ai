import os
from operator import itemgetter
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_pinecone import PineconeVectorStore


load_dotenv()
print("Configuring runtime environment...")

# Embeddings
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=os.getenv("NVIDIA_NV_API"),
)

# Vector Store
vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name="simplerag",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatNVIDIA(
    model="mistralai/devstral-2-123b-instruct-2512",  # stable model
    api_key=os.getenv("NVIDIA_MISTRAL_API_KEY"),
)

# Prompt
prompt_template = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:
""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Implementation 1: Without LCEL
def retrieval_chain_without_lcel(query: str):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content


# Implementation 2: With LCEL 
def create_retrieval_chain():
    return (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    query = "What are the major vulns in OAuth 2.0 and how do they work?"

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM (No RAG)")
    print("=" * 70)

    raw = llm.invoke([HumanMessage(content=query)])
    print(raw.content)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)

    result1 = retrieval_chain_without_lcel(query)
    print(result1)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL (BEST)")
    print("=" * 70)

    chain = create_retrieval_chain()
    result2 = chain.invoke({"question": query})
    print(result2)



