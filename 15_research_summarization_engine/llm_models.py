
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()
nvidia_mistral_api_key = os.getenv("NVIDIA_MISTRAL_API_KEY") 


def get_llm():
    # Centralized LLM factory so all modules use the same model config.
    return ChatNVIDIA(
    model="mistralai/devstral-2-123b-instruct-2512",
    api_key=nvidia_mistral_api_key,
)
