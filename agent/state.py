"""Agent 状态定义 — LangGraph 使用的状态 Schema"""
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel


class PaperMeta(BaseModel):
    """单篇论文元数据"""
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    citations: int = 0
    source: str = "arxiv"  # arxiv / semantic_scholar
    url: str = ""
    pdf_path: str = ""
    is_indexed: bool = False  # 是否已入向量库


class ResearchState(TypedDict, total=False):
    """LangGraph Agent 全局状态"""
    # 用户输入
    query: str                       # 原始查询
    intent: Literal["search", "qa", "summarize", "refine"]  # 意图

    # 检索结果
    papers: list[dict]               # PaperMeta 列表（dict 形式便于序列化）
    filtered_papers: list[dict]      # 筛选后的论文

    # RAG
    rag_context: str                 # 向量检索召回的上下文
    pdf_chunks_count: int            # 本次解析的 chunk 数

    # 生成结果
    summary: str                     # 综述 / 摘要结果
    answer: str                      # QA 回答

    # 记忆
    new_papers_count: int            # 本次新增论文数
    total_papers_count: int          # 论文库总数

    # 流式输出
    stream_buffer: str               # SSE 流式缓冲

    # 错误
    error: Optional[str]
