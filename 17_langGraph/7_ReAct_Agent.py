import os
from typing import TypedDict, List, Annotated, Sequence
# BaseMessage : the base class for all messages
# ToolMessage : a message that represents a tool call
# SystemMessage : a message that represents a system message
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_nvidia import ChatNVIDIA
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# Annotated - provides additional context without affecting the type itself
# Sequence - represents a sequence of elements
# List - represents a list of elements

class AgentState(TypedDict):
    messages:   Annotated[Sequence[BaseMessage], add_messages]

@tool 
def add(a: int, b: int):
    """This is an addition function that adds two numbers"""
    return a + b

tools = [add]

model = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=os.getenv("NVIDIA_API_KEY")).bind_tools(tools)

def model_call(state: AgentState)-> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else: 
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("our_agent")

# Add conditional edges: "our_agent" -> "tools"
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tool_node",
        "end": END,
    }
)

graph.add_edge("tool_node", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [{"role": "user", "content": "Add 34 + 55"}]}
print_stream(app.stream(inputs, stream_mode="values"))
