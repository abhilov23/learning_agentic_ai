import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import {z} from "zod";
import { createAgent, Tool, tool } from "langchain";
import { promisify } from "node:util";
import { exec } from "node:child_process";
import fs from "fs/promises";


const model = new ChatOpenAI({
  model: "meta/llama-3.1-70b-instruct",
  apiKey: process.env.NVIDIA_API_KEY,
  configuration: {
    baseURL: "https://integrate.api.nvidia.com/v1",
  },
});

// defined all the tools for the agent 
const readfile = tool(
    async ({directory}) =>{
        try {
            const files =  await fs.readdir(directory);
            return JSON.stringify(files, null, 2);
        } catch (error: any){
            return `Error: ${error.message}`
        }
    },
    {
        name: "readfile",
        description: "Read a file",
        schema: z.object({
        directory: z.string().describe("The directory to read")
        })
    }
)


const listFileTool = tool(
    async({directory}) =>{
      try{
       const files = await fs.readdir(directory);
       return JSON.stringify(files, null, 2);
      } catch(error: any){
        return `Error: {error}`
      }
    },{
     name: "listfile",
     description: "List files in a directory",
     schema: z.object({
        directory: z.string().describe("The directory to list")
     })
    }
)

const writingFileTool = tool(
    async ({directory, content}) =>{
        try {
            await fs.writeFile(directory, content);
            return "File written successfully";
        } catch (error: any){
            return `Error: ${error.message}`
        }
    }, 
    {
        name: "writefile",
        description: "Write a file",
        schema: z.object({
            directory: z.string().describe("The directory to write"),
            content: z.string().describe("The content to write")
        })
    }
)



// creating ReAct Agent
const agent = createAgent({
    model: model,
    tools: [
        readfile,
        listFileTool,
        writingFileTool
    ]
})



const result = await agent.invoke({
    "messages":{
        "content": "List all files in current directory",
        "role": "user"
    }
})
console.log(result)