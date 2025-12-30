from typing import List, Dict, Any, Optional
import httpx
from openai import OpenAI, AsyncOpenAI
from src.config.settings import settings
from src.utils.logger import logger

# 封装了对云端大模型API（如OpenAI GPT）的调用，负责发送请求并接收模型生成的文本回复。
class LLMClient:
    """大模型客户端"""

    def __init__(self):
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        self.model = settings.openai_model

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0, connect=5.0)
        )

        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0, connect=5.0)
        )

    def chat_completion(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stream: bool = False
    ) -> Dict[str, Any]:
        """同步聊天补全"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                return response

            return {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "usage": response.usage.dict() if response.usage else None
            }
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {str(e)}")
            raise

    async def async_chat_completion(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stream: bool = False
    ) -> Dict[str, Any]:
        """异步聊天补全"""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                return response

            return {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "usage": response.usage.dict() if response.usage else None
            }
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"生成嵌入失败: {str(e)}")
            raise


llm_client = LLMClient()