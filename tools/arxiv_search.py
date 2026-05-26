"""arXiv 论文检索工具"""
import arxiv
from typing import Optional
import config


def search_arxiv(query: str, max_results: int = None) -> list[dict]:
    """
    从 arXiv 检索论文，返回标准化的 PaperMeta 字典列表。
    
    Args:
        query: 搜索关键词，支持 arXiv 查询语法 (ti: title, au: author, abs: abstract)
        max_results: 最大返回数量
    
    Returns:
        标准化论文列表
    """
    max_results = max_results or config.ARXIV_MAX_RESULTS
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    papers = []
    for result in client.results(search):
        papers.append({
            "paper_id": f"arxiv_{result.entry_id.split('/')[-1]}",
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary.replace("\n", " ").strip(),
            "year": result.published.year,
            "citations": 0,  # arXiv 不提供引用数，后续由 Semantic Scholar 补充
            "source": "arxiv",
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
        })
    
    return papers


def search_arxiv_advanced(
    keywords: list[str],
    category: Optional[str] = None,
    max_results: int = 10,
) -> list[dict]:
    """
    高级检索：支持关键词列表 + 分类过滤。
    
    Args:
        keywords: 关键词列表，会用 AND 连接
        category: arXiv 分类，如 "cs.AI", "cs.CV"
        max_results: 最大返回数量
    """
    query = " AND ".join(f"all:{kw}" for kw in keywords)
    if category:
        query += f" AND cat:{category}"
    
    return search_arxiv(query, max_results=max_results)


def search_arxiv_trend(query: str, years: list[int] = None) -> dict:
    """
    面向趋势分析的专用检索：按年份分桶检索 arXiv，
    拉取更多结果用于趋势统计（每年份 50 篇）。

    Args:
        query: 搜索关键词
        years: 需要统计的年份列表，默认近两年

    Returns:
        {
            "year_stats": {"2025": {"total": count, "sample_papers": [...]}, ...},
            "query": query,
        }
    """
    from datetime import datetime
    if years is None:
        current_year = datetime.now().year
        years = [current_year - 1, current_year]

    year_stats = {}
    for year in years:
        papers = []
        try:
            client = arxiv.Client()
            # arXiv 搜索语法：按年份过滤
            search_query = f"{query} AND submittedDate:[{year}01010000 TO {year}12312359]"
            search = arxiv.Search(
                query=search_query,
                max_results=50,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            for result in client.results(search):
                papers.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors][:3],
                    "year": result.published.year,
                    "citations": 0,
                })
        except Exception as e:
            print(f"[arXiv Trend] {year}年检索失败: {e}")

        year_stats[str(year)] = {
            "total": len(papers),  # arXiv API 不返回 total，用实际获取数
            "sample_papers": papers,
        }

    return {"year_stats": year_stats, "query": query}
