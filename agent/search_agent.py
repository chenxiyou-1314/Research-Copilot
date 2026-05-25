"""Search Agent — 论文检索、去重、元数据补全"""
from langchain_core.language_models import BaseLanguageModel
from tools.arxiv_search import search_arxiv
from tools.scholar_search import search_scholar, enrich_citations
from agent.prompts import QUERY_REWRITE


def search_papers(
    llm: BaseLanguageModel,
    query: str,
    search_queries: list[str] = None,
    max_results: int = 10,
) -> list[dict]:
    """
    Search Agent: 查询改写 → 多源检索 → 去重合并 → 引用数补全。
    
    Args:
        llm: LLM实例（用于查询改写）
        query: 原始查询
        search_queries: 改写后的查询列表（由Coordinator提供），为空则用原始query
        max_results: 每个源的最大返回数
    
    Returns:
        去重后的论文列表
    """
    # Step 1: 查询改写（如果没有预先改写）
    if not search_queries:
        search_queries = rewrite_query(llm, query)
    
    # Step 2: 多源检索
    all_papers = []
    for q in search_queries:
        arxiv_results = search_arxiv(q, max_results=max_results)
        scholar_results = search_scholar(q, max_results=max_results)
        all_papers.extend(arxiv_results)
        all_papers.extend(scholar_results)
    
    # Step 3: 去重
    seen_ids = set()
    deduped = []
    for p in all_papers:
        # 按标题去重（不同源可能返回同一篇）
        title_key = p.get("title", "").lower().strip()
        id_key = p.get("paper_id", "")
        if id_key not in seen_ids and title_key not in seen_ids:
            seen_ids.add(id_key)
            seen_ids.add(title_key)
            deduped.append(p)
    
    # Step 4: 引用数补全（arXiv论文没有引用数，用Semantic Scholar补）
    deduped = enrich_citations(deduped)
    
    return deduped


def filter_papers(
    papers: list[dict],
    min_citations: int = 5,
    year_from: int = 2022,
    max_papers: int = 10,
) -> list[dict]:
    """
    论文筛选：按引用数、年份过滤，保留top-N。
    """
    # 先按引用数排序
    sorted_papers = sorted(papers, key=lambda p: p.get("citations", 0), reverse=True)
    
    filtered = []
    for p in sorted_papers:
        if len(filtered) >= max_papers:
            break
        year = p.get("year", 0)
        citations = p.get("citations", 0)
        
        # 宽松筛选：年份达标或引用数很高
        if year >= year_from or citations >= 50:
            filtered.append(p)
    
    # 如果筛选后太少，放宽限制
    if len(filtered) < 3:
        filtered = sorted_papers[:max_papers]
    
    return filtered


def rewrite_query(llm: BaseLanguageModel, query: str) -> list[str]:
    """
    查询改写：将模糊query改写为多个具体检索query。
    """
    prompt = QUERY_REWRITE.format(query=query)
    try:
        response = llm.invoke(prompt)
        lines = [l.strip().strip('- ').strip('"\'') for l in response.content.strip().split('\n') if l.strip()]
        # 过滤空行和编号
        queries = [l for l in lines if l and not l[0].isdigit()]
        if len(queries) >= 2:
            return queries[:3]  # 最多3个改写query
    except Exception:
        pass
    return [query]  # 改写失败就用原始query
