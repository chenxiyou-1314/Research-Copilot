"""FAISS 向量存储 — 支持增量索引"""
import os
import json
import pickle
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config


class VectorStore:
    """FAISS 向量库，支持增量添加和持久化。"""

    def __init__(self):
        self.embeddings = self._get_embeddings()
        self.db: Optional[FAISS] = None
        self._index_meta_path = os.path.join(
            os.path.dirname(config.FAISS_INDEX_PATH), "index_meta.json"
        )
        self._indexed_paper_ids: set = set()
        self._load()

    def _get_embeddings(self):
        if config.EMBEDDING_PROVIDER == "local":
            return HuggingFaceEmbeddings(
                model_name=config.LOCAL_EMBEDDING_MODEL,
            )
        return OpenAIEmbeddings(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            model=config.EMBEDDING_MODEL,
        )

    def _load(self):
        """加载已有的 FAISS 索引和元数据。"""
        if os.path.exists(config.FAISS_INDEX_PATH):
            try:
                self.db = FAISS.load_local(
                    config.FAISS_INDEX_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"[VectorStore] 已加载索引，共 {self.db.index.ntotal} 条向量")
            except Exception as e:
                print(f"[VectorStore] 加载索引失败: {e}，将创建新索引")
                self.db = None
        
        # 加载已索引的 paper_id 集合
        if os.path.exists(self._index_meta_path):
            with open(self._index_meta_path, "r") as f:
                meta = json.load(f)
                self._indexed_paper_ids = set(meta.get("indexed_papers", []))

    def _save_meta(self):
        """保存索引元数据。"""
        os.makedirs(os.path.dirname(self._index_meta_path), exist_ok=True)
        with open(self._index_meta_path, "w") as f:
            json.dump({"indexed_papers": list(self._indexed_paper_ids)}, f)

    def add_documents(self, paper_id: str, chunks: list[dict]):
        """
        增量添加文档到向量库。
        
        Args:
            paper_id: 论文ID，用于去重
            chunks: chunk 列表，每项含 content + metadata
        """
        if paper_id in self._indexed_paper_ids:
            print(f"[VectorStore] {paper_id} 已索引，跳过")
            return

        documents = [
            Document(
                page_content=chunk["content"],
                metadata=chunk.get("metadata", {}),
            )
            for chunk in chunks
        ]

        if not documents:
            return

        if self.db is None:
            self.db = FAISS.from_documents(documents, self.embeddings)
        else:
            self.db.add_documents(documents)

        self._indexed_paper_ids.add(paper_id)
        self._save()
        print(f"[VectorStore] 已添加 {paper_id}: {len(documents)} 条chunk")

    def search(self, query: str, top_k: int = 10) -> str:
        """
        向量检索，返回拼接的上下文字符串。
        
        Args:
            query: 查询文本
            top_k: 返回 top-k 结果
        
        Returns:
            拼接的上下文文本
        """
        if self.db is None:
            return ""

        results = self.db.similarity_search_with_score(query, k=top_k)
        
        context_parts = []
        for doc, score in results:
            source = doc.metadata.get("title", "Unknown")
            context_parts.append(f"[来源: {source}]\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)

    def is_indexed(self, paper_id: str) -> bool:
        """检查论文是否已入向量库。"""
        return paper_id in self._indexed_paper_ids

    def _save(self):
        """持久化 FAISS 索引。"""
        if self.db is not None:
            os.makedirs(os.path.dirname(config.FAISS_INDEX_PATH), exist_ok=True)
            self.db.save_local(config.FAISS_INDEX_PATH)
            self._save_meta()

    def count(self) -> int:
        """返回索引的论文数量。"""
        return len(self._indexed_paper_ids)
