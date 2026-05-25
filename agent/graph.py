"""LangGraph 多Agent工作流 — Coordinator/Search/Analysis/Writing/Critic"""
import json
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from agent.state import ResearchState
from agent.coordinator import plan_task
from agent.search_agent import search_papers, filter_papers
from agent.analysis_agent import AnalysisAgent
from agent.writing_agent import generate_summary, generate_qa_answer, revise_summary
from agent.critic_agent import evaluate_summary
from agent.novelty_agent import run_novelty_analysis
from agent.decomposition_agent import run_decomposition
from agent.trend_agent import run_trend_forecast
from tools.vector_store import VectorStore
from memory.paper_store import PaperStore
from memory.user_profile import UserProfile
from agent.prompts import INTENT_CLASSIFICATION
import config


# ── 初始化 ──
def _get_llm():
    if config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            temperature=0.1,
        )
    return ChatOpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        model=config.OPENAI_MODEL,
        temperature=0.1,
    )


llm = _get_llm()
paper_store = PaperStore()
user_profile = UserProfile()
vector_store = VectorStore()
analysis_agent = AnalysisAgent(vector_store, paper_store)

# 最大重试次数（Critic不通过时最多重跑几次）
MAX_RERUN = 2


# ── 节点1: 意图识别 ──
def intent_node(state: ResearchState) -> dict:
    """识别用户意图"""
    query = state.get("query", "")
    prompt = INTENT_CLASSIFICATION.format(query=query)
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()
    valid_intents = ["search", "qa", "summarize", "refine"]
    if intent not in valid_intents:
        intent = "search"
    return {"intent": intent}


# ── 节点2: Coordinator规划 ──
def coordinator_node(state: ResearchState) -> dict:
    """Coordinator Agent: 规划任务，生成检索策略"""
    query = state["query"]
    preferences = user_profile.get_preferences_str()
    feedback = state.get("critic_feedback", "")
    
    plan = plan_task(llm, query, preferences, feedback)
    
    return {
        "search_queries": plan.get("search_queries", [query]),
        "focus_areas": plan.get("focus_areas", []),
    }


# ── 节点3: Search Agent ──
def search_node(state: ResearchState) -> dict:
    """Search Agent: 查询改写→多源检索→去重→筛选"""
    query = state["query"]
    search_queries = state.get("search_queries", [query])
    
    # 多源检索
    papers = search_papers(
        llm=llm,
        query=query,
        search_queries=search_queries,
        max_results=config.ARXIV_MAX_RESULTS,
    )
    
    # 筛选
    filtered = filter_papers(
        papers,
        min_citations=config.PAPER_MIN_CITATIONS,
        year_from=config.PAPER_YEAR_FROM,
        max_papers=10,
    )
    
    return {"papers": papers, "filtered_papers": filtered}


# ── 节点4: Analysis Agent ──
def analysis_node(state: ResearchState) -> dict:
    """Analysis Agent: PDF解析→向量化→RAG检索"""
    papers = state.get("filtered_papers", [])
    
    # 解析+向量化
    result = analysis_agent.process_papers(papers)
    
    # RAG检索
    query = state["query"]
    rag_context = analysis_agent.rag_search(query, top_k=20)
    
    # 构建论文上下文
    papers_context = ""
    for p in papers:
        authors_str = ', '.join(p.get('authors', [])[:3])
        papers_context += f"\n[{authors_str}, {p.get('year')}] {p.get('title')}\n  摘要: {p.get('abstract', '')[:300]}\n"
    
    return {
        "rag_context": rag_context,
        "papers_context": papers_context,
        "new_papers_count": result["new_count"],
        "total_papers_count": result["total"],
        "pdf_chunks_count": result["chunks_count"],
    }


# ── 节点5: Writing Agent ──
def writing_node(state: ResearchState) -> dict:
    """Writing Agent: 生成综述或QA回答"""
    query = state["query"]
    intent = state.get("intent", "summarize")
    rag_context = state.get("rag_context", "")
    
    # 如果是重写（Critic不通过），用修订模式
    critic_feedback = state.get("critic_feedback", "")
    existing_summary = state.get("summary", "")
    
    if critic_feedback and existing_summary:
        summary = revise_summary(
            llm, existing_summary, critic_feedback, query, rag_context
        )
    elif intent == "qa":
        answer = generate_qa_answer(llm, query, rag_context)
        return {"answer": answer}
    else:
        papers_context = state.get("papers_context", "")
        summary = generate_summary(llm, query, papers_context, rag_context)
    
    return {"summary": summary}


# ── 节点6: Critic Agent ──
def critic_node(state: ResearchState) -> dict:
    """Critic Agent: 评估综述质量，决定是否重写"""
    summary = state.get("summary", "")
    if not summary:
        return {"critic_passed": True, "critic_score": 0.0, "critic_feedback": ""}
    
    query = state["query"]
    papers_context = state.get("papers_context", "")
    
    result = evaluate_summary(llm, query, summary, papers_context)
    
    return {
        "critic_passed": result["passed"],
        "critic_score": result["overall_score"],
        "critic_coverage": result["coverage_score"],
        "critic_accuracy": result["accuracy_score"],
        "critic_coherence": result["coherence_score"],
        "critic_feedback": result["feedback"],
        "rerun_count": state.get("rerun_count", 0),
    }


# ── 节点7: Novelty Agent ──
def novelty_node(state: ResearchState) -> dict:
    """Novelty Agent: Gap分析→跨域迁移→思路生成→新颖性验证"""
    summary = state.get("summary", "")
    query = state["query"]
    
    if not summary:
        return {}
    
    result = run_novelty_analysis(llm, summary, query)
    
    return {
        "gaps": result["gaps"],
        "transfers": result["transfers"],
        "ideas": result["ideas"],
        "verified_ideas": result["verified_ideas"],
    }


# ── 节点8: Method Decomposition Agent ──
def decomposition_node(state: ResearchState) -> dict:
    """Method Decomposition Agent: 方法解构→跨论文重组→可行性验证"""
    papers_context = state.get("papers_context", "")
    gaps = state.get("gaps", {})
    query = state["query"]

    if not papers_context:
        return {}

    result = run_decomposition(llm, papers_context, gaps, query)

    return {
        "decomposition": result["decomposition"],
        "recombinations": result["recombinations"],
        "validated_recombinations": result["validated_recombinations"],
    }


# ── 节点9: Trend Forecasting Agent ──
def trend_node(state: ResearchState) -> dict:
    """Trend Forecasting Agent: 时间线分析→方法演化追踪→趋势预测"""
    papers = state.get("filtered_papers", state.get("papers", []))
    decomposition = state.get("decomposition", [])
    gaps = state.get("gaps", {})
    query = state["query"]

    if not papers:
        return {}

    result = run_trend_forecast(llm, papers, decomposition, gaps, query)

    return {
        "timeline": result["timeline"],
        "evolution": result["evolution"],
        "trend_forecast": result["trend_forecast"],
    }


# ── 节点10: 记忆更新 ──
def memory_node(state: ResearchState) -> dict:
    """持久化论文库 + 更新用户偏好"""
    paper_store.save()
    query = state.get("query", "")
    user_profile.update_from_query(query)
    return {}


# ── 条件路由 ──
def route_by_intent(state: ResearchState) -> str:
    intent = state.get("intent", "search")
    if intent in ("search", "summarize", "refine"):
        return "coordinator"
    elif intent == "qa":
        return "analysis"
    return "coordinator"


def route_by_critic(state: ResearchState) -> str:
    """Critic路由：通过→记忆更新，未通过→重写"""
    if state.get("critic_passed", True):
        return "memory"
    
    rerun_count = state.get("rerun_count", 0)
    if rerun_count >= MAX_RERUN:
        return "memory"  # 超过最大重试次数，强制通过
    
    return "coordinator"  # 返回Coordinator重新规划


# ── 构建图 ──
def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # 添加节点
    graph.add_node("intent", intent_node)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("search", search_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("writing", writing_node)
    graph.add_node("critic", critic_node)
    graph.add_node("novelty", novelty_node)
    graph.add_node("decomposition", decomposition_node)
    graph.add_node("trend", trend_node)
    graph.add_node("memory", memory_node)

    # 入口
    graph.set_entry_point("intent")

    # 意图路由
    graph.add_conditional_edges(
        "intent",
        route_by_intent,
        {
            "coordinator": "coordinator",
            "analysis": "analysis",  # QA直接走Analysis
        },
    )

    # 主流程
    graph.add_edge("coordinator", "search")
    graph.add_edge("search", "analysis")
    graph.add_edge("analysis", "writing")
    graph.add_edge("writing", "critic")

    # Critic路由
    graph.add_conditional_edges(
        "critic",
        route_by_critic,
        {
            "novelty": "novelty",         # 通过→Novelty分析
            "coordinator": "coordinator",  # 重跑
        },
    )

    # Novelty → Decomposition → Trend → Memory
    graph.add_edge("novelty", "decomposition")
    graph.add_edge("decomposition", "trend")
    graph.add_edge("trend", "memory")
    graph.add_edge("memory", END)

    return graph.compile()


# 全局 Agent 实例
agent = build_graph()
