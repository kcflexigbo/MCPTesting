import asyncio
import operator
import os
from contextlib import asynccontextmanager
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolCall, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.constants import END
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class UpdateScratchpad(BaseModel):
    """
    This is a class used to update the scratchpad with new content
    """
    content:str = Field(default="",description="The contents of the scratchpad")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    scratchpad: str

@asynccontextmanager
async def get_mcp_tools():
    """
    Connects to the FastMCP server and yields its tools.
    :return:
        mcp_tools
    """
    math_server_path = os.path.join(os.path.dirname(__file__), "math_server.py")
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [math_server_path],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    yield tools


class Agent:
    def __init__(self, llm_model: str = "gemini-2.5-flash-preview-05-20", tools: List[BaseTool] = None):
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini-2025-04-14",
            temperature=1.0,
            openai_api_base=OPENAI_BASE_URL,
            openai_api_key=OPENAI_API_KEY,
        )

        self.tools = tools
        self.llm = self.llm.bind_tools(self.tools + [UpdateScratchpad])
        self.graph = self._build_graph()

    async def ainitialize(self):
        """
        Asyncronously Initializes the tools and binds to agent
        :return:
        """
        all_mcp_tools = []
        async with get_mcp_tools() as mcp_tools:
            all_mcp_tools = mcp_tools
        self.tools.extend(all_mcp_tools)
        self.llm.bind_tools(self.tools)
        self.graph = await self._build_graph()

    async def _call_model(self, state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        response = await self.llm.ainvoke(messages)
        print(response)
        return_state = {"messages": [response]}

        scratchpad_update_content = None
        if response.tool_calls:
            regular_tool_calls = []
            for tc in response.tool_calls:
                if tc['name'] == UpdateScratchpad.__name__:
                    scratchpad_update_content = tc['args']['content']
                    return_state["messages"].append(ToolMessage(content=scratchpad_update_content,
                                                                tool_call_id=tc['id']))
                    print(f"📝 Agent requested to update scratchpad with: '{scratchpad_update_content}'")
                else:
                    regular_tool_calls.append(tc)
            response.tool_calls = regular_tool_calls

        if scratchpad_update_content is not None:
            return_state["scratchpad"] = scratchpad_update_content



        return return_state

    async def _call_tools(self, state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        last_message = messages[-1]
        print([tool for tool in last_message.tool_calls])

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            error_message = "Error: The last message was not an AIMessage with tool calls. Cannot execute action."
            return {"messages": [ToolMessage(content=error_message,tool_call_id="")]}

        tool_runs = []
        for tool_call in last_message.tool_calls:
            try:
                if not tool_call.get('name') or not isinstance(tool_call.get('args'), dict):
                    error_message = f"Error: Invalid tool call format: {tool_call}. Expected a ToolCall object."
                    return {"messages": [ToolMessage(content=error_message, tool_call_id = tool_call["id"])]}
                tool_name = tool_call["name"]
                tool: Optional[BaseTool] = next((t for t in self.tools if t.name == tool_name), None)

                if not tool:
                    error_message = f"Error: Tool {tool_call['name']} not found. Cannot execute action."
                    return {"messages": [ToolMessage(content=error_message)]}
                try:
                    tool_run = await tool.ainvoke(tool_call["args"])
                    print(f"Tool Ran: {tool_run}")
                    tool_run = ToolMessage(
                                            content=str(tool_run),
                                            tool_call_id=tool_call["id"]
                    )
                except Exception as e:
                    error_message = f"Error: Failed to execute tool {tool_call['name']}. {str(e)}"
                    tool_run = ToolMessage(
                                            content=f"Error: {error_message}",
                                            tool_call_id=tool_call["id"]
                    )
                tool_runs.append(tool_run)
            except Exception as e:
                error_message = f"Error: Failed to execute tool {tool_call['name']}. {str(e)}"
                tool_run = ToolMessage(
                                        content=f"Error: {error_message}",
                                        id=tool_call["id"]
                )
                tool_runs.append(tool_run)
        return {"messages": tool_runs}

    def _should_continue(self, state:  AgentState) -> str:
        """
        Function to decide if agent should continue to end or execute
        :param state:
        :return:
            string
        """
        messages = state["messages"]
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage):
            return "END"

        if not last_message.tool_calls:
            return "END"
        return "call_tools"


    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("LLM", self._call_model)
        workflow.add_node("Tools", self._call_tools)
        workflow.add_edge(START, "LLM")
        workflow.add_conditional_edges(
            source= "LLM",
            path= self._should_continue,
            path_map={
                "call_tools": "Tools",
                "END": END,
            }
        )
        workflow.add_edge("Tools", "LLM")
        # workflow.add_edge("LLM", END)
        graph = workflow.compile()
        return graph

    async def ainvoke(self) -> Dict[str, Any]:
        """
        THis invokes the agent and begins the chatbot
        :return:
        """
        system_message_text = (
            "You are a helpful assistant. You MUST use the provided tools to answer questions when they are relevant. "
            "Use the `GetUsername` tool to find the user's name. "
            "Use `add_numbers` or `subtract_numbers` for any calculations. "
            "For all other questions, answer from your general knowledge."
        )
        state = AgentState(messages=[SystemMessage(content=system_message_text)],
                           scratchpad="")

        while True:
            user_input = str(input("User: "))
            if not len(state["messages"]):
                state["messages"].append(HumanMessage(content=user_input))
            else:
                state["messages"].append(HumanMessage(content=user_input))
            state = await self.graph.ainvoke(state)
            print("Assistant:", state["messages"][-1].content)
            print("Scratchpad:", state["scratchpad"])

async def main():
    """Main entry point for the application."""
    # The MCP server process starts here and lives until the 'with' block is exited.
    async with get_mcp_tools() as mcp_tools:
        # Pass the live, functioning tools to the agent.
        agent = Agent(tools=mcp_tools)
        # The agent is now ready to chat, with a live connection to its tools.
        await agent.ainvoke()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting application.")