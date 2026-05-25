"""Analysis Agent — PDF解析、RAG检索、论文对比分析"""
from langchain_core.language_models import BaseLanguageModel
from tools.pdf_parser import parse_papers
from tools.vector_store import VectorStore
from memory.paper_store import PaperStore
from agent.prompts import PAPER_COMPARISON


class AnalysisAgent:
    """Analysis Agent: 负责PDF解析、向量化、RAG检索和论文对比。"""

    def __init__(self, vector_store: VectorStore, paper_store: PaperStore):
        self.vector_store = vector_store
        self.paper_store = paper_store

    def process_papers(self, papers: list[dict]) -> dict:
        """
        处理论文：PDF解析 → 切分 → Embedding → 入FAISS。
        
        Returns:
            {"new_count": int, "chunks_count": int, "total": int}
        """
        new_count = 0
        chunks_count = 0

        for p in papers:
            if self.paper_store.is_indexed(p["paper_id"]):
                continue

            chunks = parse_papers(p)
            if chunks:
                self.vector_store.add_documents(p["paper_id"], chunks)
                p["is_indexed"] = True
                new_count += 1
                chunks_count += len(chunks)

            self.paper_store.add_paper(p)

        self.paper_store.save()
        return {
            "new_count": new_count,
            "chunks_count": chunks_count,
            "total": self.paper_store.count(),
        }

    def rag_search(self, query: str, top_k: int = 20) -> str:
        """RAG向量检索。"""
        return self.vector_store.search(query, top_k=top_k)

    def compare_papers(self, llm: BaseLanguageModel, papers: list[dict], focus: str = "") -> str:
        """
        论文对比分析：生成对比矩阵。
        
        Args:
            llm: LLM实例
            papers: 论文列表
            focus: 对比焦点
        
        Returns:
            对比分析的Markdown文本
        """
        papers_info = ""
        for i, p in enumerate(papers, 1):
            papers_info += f"\n论文{i}: {p.get('title', '')}\n"
            papers_info += f"  作者: {', '.join(p.get('authors', [])[:3])}\n"
            papers_info += f"  年份: {p.get('year', '')}\n"
            papers_info += f"  摘要: {p.get('abstract', '')[:300]}\n"

        prompt = PAPER_COMPARISON.format(papers_info=papers_info, focus=focus)
        response = llm.invoke(prompt)
        return response.content
