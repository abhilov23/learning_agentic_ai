from typing import TypedDict, List 
from langchain_core.messages import HumanMessage
from langchain_nvidia import ChatNVIDIA
from langgraph.graph import StateGraph, START, END 
from dotenv import load_dotenv 
import os
load_dotenv()

#model="meta/llama-3.1-70b-instruct",
#NVIDIA_API_KEY


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm  = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=os.getenv("NVIDIA_API_KEY"),)


def process(state:AgentState)-> AgentState:
    response = llm.invoke(state["messages"])
    print(f"Response: {response.content}")
    return state 



graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter your message: ")
while user_input != "quit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter your message: ")