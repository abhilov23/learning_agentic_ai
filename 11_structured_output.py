from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import httpx
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from typing import List, Union
from pydantic import BaseModel, Field # the ptdantic package is used to create a pydantic model for the structured output

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
def search_tool(query: str) -> List[dict]:
    """
    Search the internet for current information.
    """
    print(f"Searching for: {query}")
    try:
        search = DuckDuckGoSearchResults(output_format="list")
        results = search.invoke(query)
        return [{"title": r["title"], "url": r["link"]} for r in results]
    except Exception as e:
        # Keep the tool non-fatal when optional search deps are unavailable.
        print(f"Search tool unavailable: {e}")
        return []


# this function creates the print_logs() function
def print_logs(messages):
    for i, msg in enumerate(messages):
        print(f"\n[Step {i+1}]")

        if isinstance(msg, HumanMessage):
            print("User:")
            print(msg.content)

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print("AI (Tool Call):")
                for call in msg.tool_calls:
                    print(f"   -> Tool: {call['name']}")
                    print(f"   -> Args: {call['args']}")
            else:
                print("AI (Final Response):")
                print(msg.content)

        elif isinstance(msg, ToolMessage):
            print("Tool Output:")
            print(msg.content)



class Source(BaseModel):
    """
    Schema for a source used by the agent
    """
    title: str
    url: str
    
    

class AgentResponse(BaseModel):
    """
    Schema for the agent response with answer and sources
    """
    answer: str = Field(description="The agent's answer to the query")
    sources: Union[List[Source], str] = Field(
        default_factory=list,
        description="Sources used for the response as a list of {title, url}.",
    )




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
        system_prompt = """
        You MUST include sources from tool output.
        Do NOT hallucinate URLs.
        Only use URLs provided by the tool.
        """,
        response_format=AgentResponse
    )

    response = agent.invoke(
        {
            "messages": HumanMessage(content="who is the PM of korea?"),
        }
    )
    result = response.get("structured_response") or response["messages"][-1].content
    print_logs(response["messages"])
    print(f" Final Answer: {result}")


if __name__ == "__main__":
    main()

