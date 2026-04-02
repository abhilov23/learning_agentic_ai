import os
from getpass import getpass
from agents import Agent, Runner
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI 
import asyncio

client = AsyncOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",  # dummy key (LM Studio doesn't require real one
    )

# here the agent is created with a name, instructions, model, 
# and the client that was initialized to connect to the local LM Studio 
# instance. 
# The agent is then ready to be used for running tasks or answering questions.
agent = Agent(
    name="my_agent",
    instructions="You're a helpful assistant",
    model=OpenAIChatCompletionsModel(  # wrapping model name + client together
        model="mistralai/mistral-nemo-instruct-2407",
        openai_client=client  # passing the client here
    ),
)

#Running our Agent
#OpenAI gives us three methods for running our agent, all via a Runner class — those methods are:

#Runner.run() which runs in async.
#Runner.run_sync() which runs in sync.
#Runner.run_streamed() which runs in async and streams the response back to us.

result = asyncio.run(Runner.run( #Runner.run() is async, so it returns a coroutine. You need to either use asyncio.run() or switch to Runner.run_sync()
    starting_agent=agent, # passing the agent to starting_agent
    input="What is the capital of France?"
))

print(result.final_output)