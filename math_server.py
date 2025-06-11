from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

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

@mcp.tool(name= "GetUsername", description="Use this tool to get username")
async def get_username() -> str:
    """Get Username"""
    return "kennethd'geminidestroyer"

if __name__ == "__main__":
    mcp.run(transport="stdio")