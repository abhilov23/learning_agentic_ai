import os
from typing import TypedDict, List , Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_nvidia import ChatNVIDIA
from langgraph.graph import StateGraph, START, END 
from dotenv import load_dotenv 

load_dotenv()

llm  = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=os.getenv("NVIDIA_API_KEY"))

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]] # List[HumanMessage]
    
def process(state: AgentState) -> AgentState:
    """This node will do solve the request"""
    response = llm.invoke(state["messages"])
    state['messages'].append(AIMessage(content=response.content))
    print(f"Response: {response.content}")
    return state


graph =StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter your message: ")
while user_input != "quit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history}) # here we send the entire conversation history instead of a single message
    print(result["messages"])
    user_input = input("Enter your message: ")