from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool(name= "add_numbers",
          description="Use this tool to add 2 numbers")
def add(a: int, b: int) -> int:
    """Add Two Numbers"""
    return 2 * (a + b)

@mcp.tool(name= "subtract_numbers",
          description="Use this tool to subtract 2 numbers")
def subtract(a: int, b: int):
    """Subtracts 2 numbers"""
    if a>b:
        return a-b
    else:
        return b-a

if __name__ == "__main__":
    mcp.run(transport="stdio")