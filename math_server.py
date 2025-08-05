from langchain_core.tools import BaseTool
from mcp.server.fastmcp import FastMCP
from pydantic import Field, BaseModel

mcp = FastMCP("Math")

class MultiplyNumbersInput(BaseModel):
    a: int = Field(default=1, description="The first number to multiply")
    b: int = Field(default=1, description="The second number to multiply")

class MultiplyNumbersOutput(BaseModel):
    result: int = Field(default=1, description="The result of the multiplication")

class MultiplyNumbersTool(BaseTool):
    name : str = "multiply_numbers"
    description : str = "Use this tool to multiply 2 numbers"
    args_schema : type[BaseModel] = MultiplyNumbersInput
    # output_schema : BaseModel = MultiplyNumbersOutput
    def _run(self, a: int, b: int) -> int:
        return a * b * 2

    async def _arun(self, a: int, b: int) -> int:
        return self._run(a, b)

mcp.add_tool(fn = MultiplyNumbersTool()._arun, name="multiply_numbers", description="Use this tool to multiply 2 numbers")
@mcp.tool(name= "add_numbers",
          description="Use this tool to add 2 numbers")
async def add(a: int, b: int) -> int:
    """Add Two Numbers"""
    return 2 * (a + b)

@mcp.tool(name= "subtract_numbers",
          description="Use this tool to subtract 2 numbers")
async def subtract(a: int, b: int) -> int:
    """Subtracts 2 numbers"""
    if a>b:
        return a-b
    else:
        return b-a

# @mcp.tool(name= "GetUsername", description="Use this tool to get username")
# async def get_username() -> str:
#     """Get Username"""
#     return "kennethd'geminidestroyer"

@mcp.resource(
    uri="resource://user/profile/username", # The unique, protocol-level identifier for this resource.
    name="username",              # The friendly name the client will use as a dictionary key.
    description="Provides the name of the currently authenticated user."
)
def get_username() -> str:
    """Returns the current user's static username."""
    return "kennethd'geminidestroyer"

@mcp.resource("company::info")
def get_company_info() -> dict:
    """Returns information about the company."""
    return {"name": "My Awesome Company", "founded": 2024}

@mcp.prompt()
def hello(name: str) -> str:
    """Generates a friendly greeting for the given name."""
    return f"Hello, {name}! Welcome to the server."

# if __name__ == "__main__":
#     mcp.run(transport="stdio")