"""LangGraph Agent 工作流定义"""
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from agent.state import ResearchState
from agent.prompts import INTENT_CLASSIFICATION
from tools.arxiv_search import search_arxiv
from tools.scholar_search import search_scholar
from tools.pdf_parser import parse_papers
from tools.vector_store import VectorStore
from tools.summary_writer import generate_summary, generate_qa_answer
from memory.paper_store import PaperStore
from memory.user_profile import UserProfile
import config
import json


# ── 初始化 ──
def _get_llm():
    if config.LLM_PROVIDER == "ollama":
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


# ── 节点1: 意图识别 ──
def intent_node(state: ResearchState) -> dict:
    """识别用户意图: search / qa / summarize / refine"""
    query = state.get("query", "")
    prompt = INTENT_CLASSIFICATION.format(query=query)
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()

    valid_intents = ["search", "qa", "summarize", "refine"]
    if intent not in valid_intents:
        intent = "search"  # 默认走检索

    return {"intent": intent}


# ── 节点2: 论文检索 ──
def search_node(state: ResearchState) -> dict:
    """从 arXiv + Semantic Scholar 检索论文"""
    query = state["query"]
    preferences = user_profile.get_preferences_str()

    # 并行检索两个源
    arxiv_results = search_arxiv(query, max_results=config.ARXIV_MAX_RESULTS)
    scholar_results = search_scholar(query, max_results=config.SCHOLAR_MAX_RESULTS)

    # 去重合并
    seen_ids = set()
    all_papers = []
    for p in arxiv_results + scholar_results:
        if p["paper_id"] not in seen_ids:
            seen_ids.add(p["paper_id"])
            all_papers.append(p)

    return {"papers": all_papers}


# ── 节点3: 论文筛选 ──
def filter_node(state: ResearchState) -> dict:
    """根据引用数/年份/偏好筛选论文"""
    papers = state.get("papers", [])
    filtered = []
    for p in papers:
        if p.get("citations", 0) >= config.PAPER_MIN_CITATIONS:
            if p.get("year", 0) >= config.PAPER_YEAR_FROM:
                filtered.append(p)

    # 如果筛选后太少，放宽引用限制
    if len(filtered) < 3:
        filtered = [p for p in papers if p.get("year", 0) >= config.PAPER_YEAR_FROM]
    if len(filtered) < 3:
        filtered = papers[:5]

    return {"filtered_papers": filtered}


# ── 节点4: PDF解析 + 向量化 ──
def parse_and_index_node(state: ResearchState) -> dict:
    """下载PDF → 解析切分 → Embedding → 入FAISS"""
    papers = state.get("filtered_papers", [])
    new_count = 0
    chunks_count = 0

    for p in papers:
        if paper_store.is_indexed(p["paper_id"]):
            continue

        # 解析PDF（下载+切分）
        chunks = parse_papers(p)
        if chunks:
            # 入向量库
            vector_store.add_documents(p["paper_id"], chunks)
            p["is_indexed"] = True
            p["pdf_path"] = chunks[0].get("source", "")
            new_count += 1
            chunks_count += len(chunks)

        # 存入论文库
        paper_store.add_paper(p)

    total = paper_store.count()
    return {
        "new_papers_count": new_count,
        "total_papers_count": total,
        "pdf_chunks_count": chunks_count,
    }


# ── 节点5: RAG检索 + 综述生成 ──
def generate_node(state: ResearchState) -> dict:
    """RAG检索相关上下文 → 生成综述/QA回答"""
    query = state["query"]
    intent = state.get("intent", "summarize")

    # RAG: 从向量库检索相关片段
    rag_context = vector_store.search(query, top_k=20)
    papers_context = ""
    for p in state.get("filtered_papers", []):
        papers_context += f"\n[{', '.join(p.get('authors', [])[:3])}, {p.get('year')}] {p.get('title')}\n  摘要: {p.get('abstract', '')[:300]}\n"

    if intent == "qa":
        answer = generate_qa_answer(llm, query, rag_context)
        return {"answer": answer, "rag_context": rag_context}
    else:
        summary = generate_summary(llm, query, papers_context, rag_context)
        return {"summary": summary, "rag_context": rag_context}


# ── 节点6: 记忆更新 ──
def memory_node(state: ResearchState) -> dict:
    """持久化论文库 + 更新用户偏好"""
    paper_store.save()
    # 根据查询更新用户偏好
    query = state.get("query", "")
    user_profile.update_from_query(query)
    return {}


# ── 条件路由 ──
def route_by_intent(state: ResearchState) -> str:
    intent = state.get("intent", "search")
    if intent in ("search", "summarize"):
        return "search"
    elif intent == "qa":
        return "qa"
    elif intent == "refine":
        return "search"  # 重新检索
    return "search"


# ── 构建图 ──
def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # 添加节点
    graph.add_node("intent", intent_node)
    graph.add_node("search", search_node)
    graph.add_node("filter", filter_node)
    graph.add_node("parse_and_index", parse_and_index_node)
    graph.add_node("generate", generate_node)
    graph.add_node("memory", memory_node)

    # 添加边
    graph.set_entry_point("intent")

    # 意图路由
    graph.add_conditional_edges(
        "intent",
        route_by_intent,
        {
            "search": "search",
            "qa": "generate",  # QA 直接走 RAG + 生成
        },
    )

    graph.add_edge("search", "filter")
    graph.add_edge("filter", "parse_and_index")
    graph.add_edge("parse_and_index", "generate")
    graph.add_edge("generate", "memory")
    graph.add_edge("memory", END)

    return graph.compile()


# 全局 Agent 实例
agent = build_graph()
