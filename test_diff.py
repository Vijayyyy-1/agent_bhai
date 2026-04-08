import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-git", "--repository", "."],
        env=None,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            diff_result = await session.call_tool(
                "git_diff",
                arguments={"repo_path": ".", "target": "HEAD"},
            )
            print("CONTENT:")
            for block in diff_result.content:
                if hasattr(block, "text"):
                    print(block.text)

asyncio.run(main())
