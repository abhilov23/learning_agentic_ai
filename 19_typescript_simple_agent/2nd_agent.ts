import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import {z} from "zod";
import { createAgent, Tool, tool } from "langchain";


// tool for adding two numbers
const addTools = tool(
    async({a, b}) =>{
        return `${a} + ${b} = ${a + b}`
    }, 
    {
        name: "add",
        description: "Add two numbers",
        schema: z.object({
            a: z.number(),
            b: z.number()
        })

    }
)

// tool for subtracting two numbers
const subtractTools = tool(
    async({a, b}) => {
        return `${a} - ${b} = ${a - b}`
    }, 
    {
        name: "subtract",
        description: "Subtract two numbers",
        schema: z.object({
            a: z.number(),
            b: z.number()
        })
    }
)

// tool for multiplying two numbers
const multiplyTools = tool(
    async({a, b}) => {
        return `${a} * ${b} = ${a * b}`
    }, 
    {
        name: "multiply",
        description: "Multiply two numbers",
        schema: z.object({
            a: z.number(),
            b: z.number()
        })
    }
)

// tool for dividing two numbers
const divideTools = tool(
    async({a, b}) => {
        return `${a} / ${b} = ${a / b}`
    }, 
    {
        name: "divide",
        description: "Divide two numbers",
        schema: z.object({
            a: z.number(),
            b: z.number()
        })
    }
)

const model = new ChatOpenAI({
  model: "meta/llama-3.1-70b-instruct",
  apiKey: process.env.NVIDIA_API_KEY,
  configuration: {
    baseURL: "https://integrate.api.nvidia.com/v1",
  },
});


const agent = createAgent({
    model: model,
    tools: [
        addTools,
        subtractTools,
        multiplyTools,
        divideTools
    ]
})


const result = await agent.invoke({
    "messages":{
        "content": "what is 2 + 2",
        "role": "user"
    }
})
console.log(result)