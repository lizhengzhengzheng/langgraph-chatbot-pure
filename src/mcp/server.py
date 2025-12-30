# src/mcp/server.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPServer:
    def __init__(self):
        self.tools = [
            {
                "name": "get_weather",
                "description": "获取指定城市的当前天气",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "calculate",
                "description": "执行数学计算",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "数学表达式，如 '2+2*3'"}
                    },
                    "required": ["expression"]
                }
            }
        ]

    async def list_tools(self):
        return self.tools

    async def call_tool(self, tool_name: str, arguments: dict):
        if tool_name == "get_weather":
            city = arguments.get("city", "北京")
            # 这里可以调用真实天气API，暂时返回模拟数据
            return {"weather": f"{city}的天气是晴天，25°C"}
        elif tool_name == "calculate":
            import ast
            expr = arguments.get("expression", "0")
            try:
                # 安全评估表达式
                result = eval(expr, {"__builtins__": {}}, {})
                return {"result": result}
            except:
                return {"error": "计算表达式无效"}
        else:
            return {"error": f"未知工具: {tool_name}"}