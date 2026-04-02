import os
import asyncio
from agents import Agent, Runner
from openai import AsyncOpenAI

# here the client is initialized to connect to the local LM Studio instance, 
# and a basic agent is created with a simple prompt. 
# The agent is then run with a sample question, and the final output is printed.
client = AsyncOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",  # dummy key (LM Studio doesn't require real one
)

basic_agent = Agent( 
    name = "basic_agent",
    instructions=(
        "You are a helpful assistant. When given a question, "
        "answer it to the best of your ability. If you don't know, say so."
    ),
    model="mistralai/mistral-nemo-instruct-2407",
    client=client
)

async def main():
    result = await Runner.run(
        basic_agent,
        "What is the capital of France?"
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())