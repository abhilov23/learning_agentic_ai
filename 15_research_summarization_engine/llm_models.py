
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY") 


def get_llm():
    return ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    api_key=nvidia_api_key,
)
