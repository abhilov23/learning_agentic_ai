from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from langchain_community.tools import DuckDuckGoSearchRun  # fixed
from langchain_classic.agents import initialize_agent, Tool

load_dotenv()


def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found")

    llm = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        nvidia_api_key=api_key,
    )

    search = DuckDuckGoSearchRun()

    tools = [  
        Tool(
            name="Search",
            func=search.run,
            description="useful for current events",
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
    )

    result = agent.invoke("who is the pm of india?")  # fixed
    print(result)

if __name__ == "__main__":
    main()
