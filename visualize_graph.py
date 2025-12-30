# visualize_graph.py
import sys
from pathlib import Path

# 添加项目路径，确保能导入模块
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入你的图实例
from src.graph import chatbot_graph

# 1. 生成Mermaid格式的文本定义（可嵌入文档）
print("=== Mermaid 文本定义 ===")
mermaid_text = chatbot_graph.compiled_graph.get_graph().draw_mermaid()
print(mermaid_text)

# 2. 尝试生成并保存为图片文件
print("\n=== 尝试生成图片 ===")
try:
    # 生成PNG图片数据
    image_data = chatbot_graph.compiled_graph.get_graph().draw_mermaid_png()
    # 保存到文件
    output_path = project_root / "langgraph_flowchart.png"
    with open(output_path, "wb") as f:
        f.write(image_data)
    print(f"✅ 流程图已保存至: {output_path}")
except Exception as e:
    print(f"⚠️  无法生成图片，可能缺少依赖。错误: {e}")
    print("请使用方法二或三进行查看。")

# 3. 打印文本结构
print("\n=== 图结构概览 ===")
print(f"图包含的节点: {list(chatbot_graph.compiled_graph.nodes.keys())}")
print(f"图包含的边: {chatbot_graph.compiled_graph.edges}")