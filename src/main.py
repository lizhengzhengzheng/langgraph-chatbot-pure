from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, APIRouter, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.session import session_manager, verify_session
from src.agents.chatbot_agent import ChatbotAgent
from src.core.vector_store import vector_store
from src.utils.logger import logger
from utils.loader import DocumentLoader
from utils.splitter import DocumentSplitter

# 创建一个会话相关的路由器
session_router = APIRouter(
    prefix="/session",
    tags=["会话相关"],
    dependencies=[Depends(verify_session)]
)
# 创建FastAPI应用
app = FastAPI(
    title="智能对话机器人API",
    description="基于云端大模型API + Qdrant向量库 + LangGraph的智能对话机器人",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建聊天机器人实例
chatbot_agent = ChatbotAgent()


# 数据模型
class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    stream: bool = Field(False, description="是否流式响应")


class ChatResponse(BaseModel):
    response: str = Field(..., description="助手回复")
    sources: List[Dict[str, Any]] = Field(default=[], description="参考来源")
    conversation_id: Optional[str] = Field(None, description="会话ID")


class DocumentUploadRequest(BaseModel):
    directory_path: Optional[str] = Field(None, description="文档目录路径")
    chunk_size: int = Field(1000, description="分块大小")
    chunk_overlap: int = Field(200, description="分块重叠")


class DocumentUploadResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    document_count: int = Field(0, description="文档数量")
    chunk_count: int = Field(0, description="分块数量")


class CollectionInfoResponse(BaseModel):
    name: str = Field(..., description="集合名称")
    vectors_count: int = Field(..., description="向量数量")
    status: str = Field(..., description="状态")

# API路由
@app.get("/")
async def root():
    """根路由"""
    return {
        "message": "智能对话机器人API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/chat",
            "/upload-documents",
            "/collection-info",
            "/clear-history"
        ]
    }


@session_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session_id: str = Depends(verify_session)):
    """聊天接口"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")

        # 处理聊天请求
        result = await chatbot_agent.chat(request.message, session_id)

        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            conversation_id=session_id  # 可以扩展会话管理
        )

    except Exception as e:
        logger.error(f"聊天处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/upload-documents", response_model=DocumentUploadResponse)
async def upload_documents(request: DocumentUploadRequest):
    """上传文档到向量库"""
    try:
        if not request.directory_path:
            raise HTTPException(status_code=400, detail="请提供文档目录路径")

        # 加载文档
        documents = DocumentLoader.load_directory(request.directory_path)

        if not documents:
            return DocumentUploadResponse(
                success=False,
                message="没有找到可处理的文档",
                document_count=0,
                chunk_count=0
            )

        # 分割文档
        splitter = DocumentSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        chunks = splitter.split_documents(documents)

        # 添加到向量库
        vector_store.create_collection()
        vector_store.add_documents(chunks)

        return DocumentUploadResponse(
            success=True,
            message=f"成功上传 {len(documents)} 个文档，分割为 {len(chunks)} 个块",
            document_count=len(documents),
            chunk_count=len(chunks)
        )

    except Exception as e:
        logger.error(f"上传文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.get("/collection-info", response_model=CollectionInfoResponse)
async def get_collection_info():
    """获取向量库信息"""
    try:
        info = vector_store.get_collection_info()

        if not info:
            raise HTTPException(status_code=404, detail="集合未找到")

        return CollectionInfoResponse(**info)

    except Exception as e:
        logger.error(f"获取集合信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@session_router.post("/clear-history")
async def clear_history(session_id: str = Query(..., description="会话唯一标识")):
    """清空指定会话的聊天历史（会话隔离版）"""
    try:
        # 按Session ID清空，而非全局清空
        session_manager.clear_session_history(session_id)
        return {"success": True, "message": f"会话 {session_id} 历史已清空"}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"清空会话 {session_id} 历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清空失败: {str(e)}")

# 新增：创建会话接口（前端初始化会话时调用）
@app.post("/create-session")
async def create_session(
    user_id: str = Query("", description="用户ID（可选）")
):
    """创建新会话（返回Session ID）"""
    try:
        context = session_manager.create_session(user_id=user_id)
        return {
            "success": True,
            "session_id": context.session_id,
            "create_time": context.create_time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"创建会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "ai-chatbot",
        "timestamp": "2024-01-01T00:00:00Z"
    }
# 挂载路由
app.include_router(session_router)

if __name__ == "__main__":
    # 启动服务
    logger.info("启动智能对话机器人服务...")

    # 确保向量库集合存在
    try:
        vector_store.create_collection()
        logger.info("向量库集合已初始化")
    except Exception as e:
        logger.warning(f"初始化向量库集合失败: {str(e)}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )