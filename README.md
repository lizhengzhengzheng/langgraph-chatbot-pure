# 智能对话机器人 API

基于云端大模型 API + Qdrant 向量库 + LangGraph 构建的智能对话机器人，支持文档上传、向量检索、智能问答等核心功能。

## 功能特性

- 🤖 智能对话：基于大模型的自然语言交互，支持上下文理解
- 📄 文档处理：支持加载 Word 文档（.docx），自动分割并存储到向量库
- 📊 向量检索：基于 Qdrant 向量库实现高效的文档内容检索
- 🌐 RESTful API：提供标准化的 HTTP 接口，易于集成
- 🔌 跨域支持：内置 CORS 中间件，支持前端跨域调用
- 📈 状态监控：提供健康检查和向量库集合信息查询接口
- 🧹 会话管理：支持清空聊天历史记录

## 技术栈

- **后端框架**：FastAPI
- **向量数据库**：Qdrant
- **文档处理**：python-docx
- **工作流**：LangGraph

## 快速开始

### 环境要求

- Python 3.8+
- Qdrant 向量库（本地/云端实例）
- 大模型 API 密钥（如 OpenAI/阿里云/百度等）

### 安装依赖

```bash
# 克隆代码库
git clone <your-repo-url>
cd ai-chatbot

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖包
pip install -r requirements.txt
```

### 配置项

在项目根目录创建 `.env` 文件，配置以下参数：

```env
# Qdrant 配置
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=chatbot_documents

# 大模型配置
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=your-llm-base-url
LLM_MODEL_NAME=your-model-name

# 日志配置
LOG_LEVEL=INFO
```

### 启动服务

```bash
# 直接运行
python src/main.py

# 或通过 uvicorn 启动
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后，可通过以下地址访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## 项目结构

```
src/
├── agents/                # 智能体相关逻辑
│   └── chatbot_agent.py   # 聊天机器人核心逻辑
├── core/                  # 核心功能模块
│   ├── loader.py          # 文档加载器（Word）
│   ├── splitter.py        # 文档分块器
│   └── vector_store.py    # 向量库操作封装
├── graph/                 # LangGraph 工作流
│   └── edges.py           # 工作流边逻辑
├── utils/                 # 工具类
│   └── logger.py          # 日志配置
└── main.py                # FastAPI 主程序入口
```

## 扩展开发

### 会话管理
扩展 `/chat` 接口，增加 `conversation_id` 参数，实现多会话隔离：
```python
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, conversation_id: Optional[str] = None):
    # 根据 conversation_id 区分不同会话
    result = await chatbot_agent.chat(request.message, conversation_id)
```

### 流式响应
修改 `chat` 接口，支持 SSE（Server-Sent Events）流式输出：
```python
from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.stream:
        return StreamingResponse(chatbot_agent.stream_chat(request.message), media_type="text/event-stream")
    # 非流式逻辑...
```

## 许可证

本项目基于 MIT 许可证开源，详情请参见 LICENSE 文件。