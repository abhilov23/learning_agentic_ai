import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

# Local broken proxy values can block NVIDIA API calls.
for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if os.getenv(key) in {"http://127.0.0.1:9", "https://127.0.0.1:9"}:
        os.environ.pop(key, None)

llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    api_key=os.getenv("NVIDIA_MISTRAL_API_KEY") or os.getenv("NVIDIA_API_KEY"),
)

text_chunks_chain = RunnableLambda(
    lambda x: [
        {"chunk": text_chunk}
        for text_chunk in [
            x[i : i + 3000].strip()
            for i in range(0, len(x), 2900)
            if x[i : i + 3000].strip()
        ]
    ]
)

summarize_chunk_prompt = PromptTemplate.from_template(
    "Write a concise summary of the following text and include the main details.\nText: {chunk}"
)
summarize_chunk_chain = summarize_chunk_prompt | llm

summarize_map_chain = RunnableParallel(
    {
        "summary": summarize_chunk_chain | StrOutputParser(),
    }
)

summarize_summaries_prompt = PromptTemplate.from_template(
    "Combine the summaries below into one clear final summary.\nSummaries:\n{summaries}"
)

summarize_reduce_chain = (
    RunnableLambda(lambda x: {"summaries": "\n".join([i["summary"] for i in x])})
    | summarize_summaries_prompt
    | llm
    | StrOutputParser()
)

map_reduce_chain = text_chunks_chain | summarize_map_chain.map() | summarize_reduce_chain
