from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from langchain_core.prompts import PromptTemplate  # fixed import

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

    # ❌ removed f-string
    summary_template = """
    Given the following information about a person:

    {information}

    Please provide:
    1. A short summary
    2. Two interesting facts
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    chain = summary_prompt_template | llm
    result = chain.invoke(summary_prompt_template.format(information=information))

    print(result.content)

if __name__ == "__main__":
    main()