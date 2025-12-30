from typing import List, Dict, Any
from pathlib import Path
from pypdf import PdfReader
import docx
from src.utils.logger import logger

# PDF/TXT/DOCX文件 → 原始纯文本字符串
class DocumentLoader:
    """文档加载器"""

    @staticmethod
    def load_pdf(file_path: str) -> List[Dict[str, Any]]:
        """加载PDF文档"""
        try:
            documents = []
            reader = PdfReader(file_path)

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": file_path,
                            "page": i + 1,
                            "total_pages": len(reader.pages)
                        },
                        "source": file_path,
                        "title": Path(file_path).stem
                    })

            logger.info(f"加载PDF文档: {file_path}, 共 {len(documents)} 页")
            return documents

        except Exception as e:
            logger.error(f"加载PDF失败 {file_path}: {str(e)}")
            return []

    @staticmethod
    def load_txt(file_path: str) -> List[Dict[str, Any]]:
        """加载文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            documents = [{
                "text": content,
                "metadata": {"source": file_path},
                "source": file_path,
                "title": Path(file_path).stem
            }]

            logger.info(f"加载文本文件: {file_path}")
            return documents

        except Exception as e:
            logger.error(f"加载文本文件失败 {file_path}: {str(e)}")
            return []

    @staticmethod
    def load_docx(file_path: str) -> List[Dict[str, Any]]:
        """加载Word文档"""
        try:
            documents = []
            doc = docx.Document(file_path)

            # 提取段落
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)

            # 合并段落
            content = "\n".join(paragraphs)

            documents.append({
                "text": content,
                "metadata": {"source": file_path},
                "source": file_path,
                "title": Path(file_path).stem
            })

            logger.info(f"加载Word文档: {file_path}")
            return documents

        except Exception as e:
            logger.error(f"加载Word文档失败 {file_path}: {str(e)}")
            return []

    @staticmethod
    def load_directory(directory_path: str) -> List[Dict[str, Any]]:
        """加载目录中的所有文档"""
        all_documents = []
        directory = Path(directory_path)

        if not directory.exists():
            logger.error(f"目录不存在: {directory_path}")
            return all_documents

        # 支持的文件类型
        supported_extensions = {
            '.pdf': DocumentLoader.load_pdf,
            '.txt': DocumentLoader.load_txt,
            '.docx': DocumentLoader.load_docx,
            '.md': DocumentLoader.load_txt
        }

        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                loader_func = supported_extensions[file_path.suffix.lower()]
                documents = loader_func(str(file_path))
                all_documents.extend(documents)

        logger.info(f"从目录加载文档完成: {directory_path}, 共 {len(all_documents)} 个文档")
        return all_documents