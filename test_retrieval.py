# 创建一个临时测试脚本 test_retrieval.py
import sys
from pathlib import Path

from config.settings import settings

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from sentence_transformers import SentenceTransformer
from src.core.vector_store import vector_store

# 1. 编码用户问题
model = SentenceTransformer(settings.embedding_model)
query = "算一下1+1等于几"
query_vector = model.encode(query).tolist()

# 2. 执行检索
results = vector_store.client.query_points(
    collection_name="mcp_tools_index",
    query=query_vector,
    limit=3,
    with_payload=True
).points

print(f"查询: '{query}'")
print(f"检索到 {len(results)} 个结果:")
for i, r in enumerate(results):
    print(f"\n{i+1}. 工具名: {r.payload.get('metadata', {}).get('name', 'N/A')}")
    print(f"   相似度分数: {r.score:.4f}")
    print(f"   工具描述: {r.payload.get('text', 'N/A')[:100]}...")