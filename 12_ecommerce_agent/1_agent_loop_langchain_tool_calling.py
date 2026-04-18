from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import httpx
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage




load_dotenv()


MAX_ITERATIONS = 10 
MODEL="meta/llama-3.1-70b-instruct"


def configure_runtime_env() -> None:
    bad_proxy = "http://127.0.0.1:9"
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        if os.environ.get(key, "").strip() == bad_proxy:
            os.environ.pop(key, None)
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

# ---Tools {Langchain @tool decorator}---

@tool
def get_product_price(product:str)-> float:
    """Lookup the price of a product"""
    print(f"Looking up the price of {product}")
    prices = {"laptop": 1000, "phone": 500, "tablet": 700}
    return prices.get(product, 0)

@tool 
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a price
    Available tiers: bronze, silver, gold
    """
    print(" >> executing apply_discount tool")
    discounts_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discounts_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100),2)





# ---Agent Loop---

def run_agent(question: str):
    configure_runtime_env()

    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    http_client = httpx.Client(trust_env=False, timeout=60.0)

    llm = ChatOpenAI(
        temperature=0,
        model=MODEL,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ.get("NVIDIA_API_KEY"),
        http_client=http_client,
    )
    llm_with_tools = llm.bind_tools(tools)  # allow model to call tools

    print("Question:", question)
    print("=" * 60)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant.\n"
                "You have access to a product catalog tool.\n"
                "You have access to a discount tool.\n"
                "STRICT RULES - You must follow:\n"
                "1. Never guess or assume any product's price.\n"
                "2. You must use the product catalog tool to get the product price.\n"
                "3. You must use the discount tool to apply the discount.\n"
                "4. Call apply_discount only after get_product_price returns a price.\n"
                "5. Pass the exact numeric price from get_product_price.\n"
                "6. Use discount_tier as one of: bronze, silver, gold."
            )
        ),
        HumanMessage(content=question),
    ]

    for _ in range(MAX_ITERATIONS):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            return ai_msg.content

        for call in ai_msg.tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args", {})
            selected_tool = tools_dict[tool_name]
            tool_result = selected_tool.invoke(tool_args)
            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=call["id"])
            )

    return "Stopped after max iterations without final answer."


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop after applying the gold discount?")
    print("Final:", result)
