import { createDeepAgent } from "deepagents";

const agent = createDeepAgent();

const result = await agent.invoke({
  messages: [
    {
      role: "user",
      content: "Research LangGraph and write a summary in summary.md",
    },
  ],
});