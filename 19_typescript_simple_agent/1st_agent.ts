import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import {z} from "zod";
import { createAgent, tool } from "langchain";



// defining the tool
const search = tool(
    async ({query}) => {
     if(
      query.toLowerCase().includes("sf") ||
      query.toLowerCase().includes("san francisco")
     ) {
        return "It's 60 degree and foggy."
     } 
     return "it's 90 degree and sunny"
    },
    {
        name: "Search",
        description: "Search for the current weather in San Francisco",
        schema: z.object({
            query: z.string().describe("The query to use in your search.")
        })
    }
);


// connecting with the api 
const model = new ChatOpenAI({
  model: "meta/llama-3.1-70b-instruct",
  apiKey: process.env.NVIDIA_API_KEY,
  configuration: {
    baseURL: "https://integrate.api.nvidia.com/v1",
  },
});



// creating the ReAct Agent
const agent = createAgent({
   model, 
   tools: [search]
});


// invoking the agent
const result = await agent.invoke({
  messages: [
    {
      role: "user",
      content: "what is the weather in San Francisco?"
    }
  ]
});


console.log(result);
