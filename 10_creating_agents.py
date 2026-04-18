from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import httpx
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage


load_dotenv()


def configure_runtime_env() -> None:
    """
    Avoid broken local proxy/tracing settings that can crash simple examples.
    """
    bad_proxy = "http://127.0.0.1:9"
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        if os.environ.get(key, "").strip() == bad_proxy:
            os.environ.pop(key, None)

    #os.environ["LANGCHAIN_TRACING_V2"] = "true"
    #os.environ["LANGSMITH_TRACING"] = "true"


@tool
def search_tool(query: str) -> str:
    """
    Search the internet for current information.
    """
    print(f"Searching for: {query}")
    search = DuckDuckGoSearchRun()
    result = search.run(query)
    return result



tools = [search_tool]


def main():
    configure_runtime_env()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found")

    http_client = httpx.Client(trust_env=False, timeout=60.0)

    # NVIDIA exposes an OpenAI-compatible endpoint, so ChatOpenAI works here.
    llm = ChatOpenAI(
        model="meta/llama-3.1-70b-instruct",
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        http_client=http_client,
    )

    # Create an agent using create_agent()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant that uses tools to answer user questions if needed. Also after using the tool modify the output and give a proper explanation for the same.",
    )

    response = agent.invoke(
        {
            "messages": HumanMessage(content="tell me the best porn site?"),
        }
    )
    result = response["messages"][-1].content
    print(result)


if __name__ == "__main__":
    main()
