import asyncio
import operator
import os
from contextlib import asynccontextmanager
from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

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
    async with client as mcp_client:
        tools = await mcp_client.get_tools()
        yield tools

class Agent:
    def __init___(self, llm_model: str = "gemini-2.5-flash-preview-05-20", tools: List[BaseTool] = None):
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=1)
        self.tools = get_mcp_tools()

