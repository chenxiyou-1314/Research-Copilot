"""Agent 状态定义 — LangGraph 使用的状态 Schema（多Agent版）"""
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
    # ── 用户输入 ──
    query: str                       # 原始查询
    intent: Literal["search", "qa", "summarize", "refine"]  # 意图

    # ── Coordinator ──
    search_queries: list[str]        # 改写后的检索query列表
    focus_areas: list[str]           # 重点关注方向

    # ── Search Agent ──
    papers: list[dict]               # 检索到的所有论文
    filtered_papers: list[dict]      # 筛选后的论文

    # ── Analysis Agent ──
    papers_context: str              # 论文列表文本（给Writing/Critic用）
    rag_context: str                 # 向量检索召回的上下文
    pdf_chunks_count: int            # 本次解析的 chunk 数
    new_papers_count: int            # 本次新增论文数
    total_papers_count: int          # 论文库总数

    # ── Writing Agent ──
    summary: str                     # 综述结果
    answer: str                      # QA 回答

    # ── Critic Agent ──
    critic_passed: bool              # Critic是否通过
    critic_score: float              # 总分
    critic_coverage: float           # 覆盖度
    critic_accuracy: float           # 准确性
    critic_coherence: float          # 连贯性
    critic_feedback: str             # Critic反馈
    rerun_count: int                 # 重试次数

    # ── Novelty Agent ──
    gaps: dict                       # Gap分析结果
    transfers: list                  # 跨域迁移结果
    ideas: list                      # 生成的研究思路
    verified_ideas: list             # 新颖性验证后的思路

    # ── Method Decomposition Agent ──
    decomposition: list              # 方法解构结果（每篇论文的原子组件）
    recombinations: list             # 跨论文方法重组方案
    validated_recombinations: list   # 可行性验证后的重组方案

    # ── Trend Forecasting Agent ──
    timeline: dict                   # 时间线分析结果
    evolution: dict                  # 方法演化追踪结果
    trend_forecast: dict             # 趋势预测结果

    # ── 流式输出 ──
    stream_buffer: str               # SSE 流式缓冲

    # ── 错误 ──
    error: Optional[str]
