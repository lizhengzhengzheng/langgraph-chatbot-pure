import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """应用配置"""

    # OpenAI配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-1106")

    # Qdrant配置
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "chatbot_documents")

    # 嵌入模型配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL")

    # 应用配置
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # redis配置
    redis_host: str = os.getenv("REDIS_HOST", "")
    redis_port: str = os.getenv("REDIS_PORT", "")
    redis_password: str = os.getenv("REDIS_PASSWORD", "")

settings = Settings()