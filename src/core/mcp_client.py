# src/core/mcp_client.py
import asyncio
import json
from typing import List, Dict, Any, Optional

import httpx
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.core.llm_client import llm_client
from src.core.vector_store import vector_store
from src.utils.logger import logger


class MCPToolClient:
    """生产级MCP客户端，支持智能工具检索和标准JSON-RPC调用"""

    def __init__(self, server_url: str = "http://localhost:8001"):
        self.server_url = server_url
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.embedding_model = SentenceTransformer(settings.embedding_model)

    async def fetch_all_tools(self) -> List[Dict[str, Any]]:
        """从MCP服务器获取所有可用工具（兼容旧接口）"""
        try:
            response = await self.http_client.get(
                f"{self.server_url}/mcp/v1/tools"
            )
            if response.status_code == 200:
                result = response.json()
                tools = result.get("tools", [])
                logger.info(f"从MCP服务器获取到 {len(tools)} 个工具")
                return tools
            else:
                logger.error(f"MCP服务器返回异常: {response.status_code}")
        except Exception as e:
            logger.error(f"获取MCP工具列表失败: {e}")
        return []

    async def retrieve_relevant_tools(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """基于语义检索相关工具"""
        try:
            query_vector = self.embedding_model.encode(query).tolist()
            search_results = vector_store.client.query_points(
                collection_name="mcp_tools_index",
                query=query_vector,
                limit=top_k * 2,
                score_threshold=0.5,
                with_payload=True,
                with_vectors=False
            ).points

            tools = []
            seen_names = set()
            for result in search_results:
                tool_schema = result.payload.get("metadata", {}).get("full_schema")
                tool_name = result.payload.get("metadata", {}).get("name")
                if tool_schema and tool_name not in seen_names:
                    tools.append({
                        **tool_schema,
                        "relevance_score": result.score
                    })
                    seen_names.add(tool_name)
                if len(tools) >= top_k:
                    break

            logger.info(f"语义检索到 {len(tools)} 个相关工具")
            return tools
        except Exception as e:
            logger.error(f"工具检索失败: {e}")
            all_tools = await self.fetch_all_tools()
            return all_tools[:top_k]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用MCP工具（标准JSON-RPC）"""
        try:
            # 标准 JSON-RPC 2.0 请求
            payload = {
                "jsonrpc": "2.0",
                "method": tool_name,  # 直接用工具名作为方法
                "params": arguments,  # 直接传参数
                "id": f"req_{asyncio.get_event_loop().time()}"  # 唯一请求ID
            }

            response = await self.http_client.post(
                f"{self.server_url}/mcp/v1/rpc",  # 正确的RPC入口
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("error") is not None:
                    logger.error(f"工具调用失败: {result['error']}")
                    return {"error": result["error"]}
                return {"result": result.get("result", {}).get("value")}
            else:
                logger.error(f"工具调用失败: HTTP {response.status_code}")
                return {"error": f"调用失败: HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"工具调用异常: {e}")
            return {"error": str(e)}

    async def intelligent_tool_call(self, user_query: str, context: Optional[str] = "") -> Dict[str, Any]:
        """智能工具调用完整流程"""
        logger.info(f"智能工具调用: {user_query}")

        relevant_tools = await self.retrieve_relevant_tools(user_query, top_k=3)
        if not relevant_tools:
            return {"error": "未找到相关工具", "response": "我暂时无法处理这个请求"}

        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']} "
            f"(参数: {json.dumps(tool.get('parameters', {}), ensure_ascii=False)})"
            for tool in relevant_tools
        ])

        prompt = f"""
        用户问题: {user_query}
        上下文: {context}

        请从以下工具中选择最合适的一个，并提取调用参数。
        只使用提供的工具。

        可用工具:
        {tools_description}

        请以JSON格式回复:
        {{
            "selected_tool": "工具名称",
            "arguments": {{"参数名": "参数值"}},
            "reasoning": "选择理由"
        }}
        """

        llm_response = await llm_client.async_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )

        try:
            import re
            json_match = re.search(r'\{.*\}', llm_response["content"], re.DOTALL)
            if json_match:
                selection = json.loads(json_match.group())
                tool_name = selection["selected_tool"]
                arguments = selection["arguments"]
                logger.info(f"LLM选择工具: {tool_name}, 参数: {arguments}")

                tool_result = await self.call_tool(tool_name, arguments)
                return {
                    "tool_used": tool_name,
                    "tool_arguments": arguments,
                    "tool_result": tool_result,
                    "reasoning": selection.get("reasoning", "")
                }
            else:
                return {"error": "无法解析LLM的响应"}
        except Exception as e:
            logger.error(f"解析或调用失败: {e}")
            return {"error": f"处理失败: {str(e)}"}


# 全局客户端实例
mcp_client = MCPToolClient()