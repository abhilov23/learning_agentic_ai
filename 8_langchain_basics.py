from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

load_dotenv()

def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found")

    llm = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        nvidia_api_key=api_key,
    )
    
    information = """
    Elon Musk is a South African-born American entrepreneur and businessman known for leading multiple high-impact technology companies.
    Born on June 28, 1971, in Pretoria, South Africa, he later moved to North America.
    He co-founded Zip2 and X.com (which became PayPal).
    He founded SpaceX, leads Tesla, and started Neuralink and The Boring Company.
    He also acquired Twitter (now X).
    """

    summary_template = f"""
    Given the following information about a person:

    {information}

    Please provide:
    1. A short summary
    2. Two interesting facts
    """

    result = llm.invoke(summary_template)
    print(result.content)

if __name__ == "__main__":
    main()