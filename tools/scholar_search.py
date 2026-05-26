"""Semantic Scholar 论文检索工具 — 补充引用数和关联论文"""
import httpx
from typing import Optional
import config

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"


def search_scholar(query: str, max_results: int = None) -> list[dict]:
    """
    从 Semantic Scholar 检索论文，包含引用数。
    
    Args:
        query: 搜索关键词
        max_results: 最大返回数量
    
    Returns:
        标准化论文列表（含 citationCount）
    """
    max_results = max_results or config.SCHOLAR_MAX_RESULTS
    
    fields = "paperId,title,authors,abstract,year,citationCount,url,openAccessPdf"
    
    try:
        resp = httpx.get(
            f"{S2_BASE_URL}/paper/search",
            params={
                "query": query,
                "limit": max_results,
                "fields": fields,
                "year": f"{config.PAPER_YEAR_FROM}-",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, KeyError) as e:
        print(f"[Scholar] 检索失败: {e}")
        return []
    
    papers = []
    for item in data.get("data", []):
        pdf_url = ""
        oa = item.get("openAccessPdf")
        if oa and isinstance(oa, dict):
            pdf_url = oa.get("url", "")
        
        papers.append({
            "paper_id": f"s2_{item.get('paperId', '')}",
            "title": item.get("title", ""),
            "authors": [a.get("name", "") for a in item.get("authors", [])],
            "abstract": (item.get("abstract") or "").replace("\n", " ").strip(),
            "year": item.get("year") or 0,
            "citations": item.get("citationCount", 0),
            "source": "semantic_scholar",
            "url": item.get("url", ""),
            "pdf_url": pdf_url,
        })
    
    return papers


def get_paper_details(paper_id: str) -> Optional[dict]:
    """
    获取单篇论文详情（引用数、参考文献、被引论文）。
    
    Args:
        paper_id: Semantic Scholar Paper ID
    """
    fields = "paperId,title,authors,abstract,year,citationCount,references,title,url"
    
    try:
        resp = httpx.get(
            f"{S2_BASE_URL}/paper/{paper_id}",
            params={"fields": fields},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


def enrich_citations(papers: list[dict]) -> list[dict]:
    """
    为 arXiv 论文补充 Semantic Scholar 的引用数。
    通过标题匹配查找对应的 S2 论文。
    """
    for p in papers:
        if p.get("citations", 0) > 0:
            continue  # 已有引用数，跳过
        
        try:
            resp = httpx.get(
                f"{S2_BASE_URL}/paper/search",
                params={
                    "query": p["title"],
                    "limit": 1,
                    "fields": "citationCount",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                results = resp.json().get("data", [])
                if results:
                    p["citations"] = results[0].get("citationCount", 0)
        except httpx.HTTPError:
            pass
    
    return papers


def search_trend_stats(query: str, years: list[int] = None) -> dict:
    """
    面向趋势分析的专用检索：按年份分桶查询 Semantic Scholar，
    利用 S2 API 返回的 total 字段获取每年的论文总量，无需拉取全部数据。

    内置指数退避重试，应对 429 限流。

    Args:
        query: 搜索关键词（建议英文）
        years: 需要统计的年份列表，默认近两年

    Returns:
        {
            "year_stats": {"2025": {"total": 1234, "sample_papers": [...]}, ...},
            "query": query,
        }
    """
    import time
    from datetime import datetime
    if years is None:
        current_year = datetime.now().year
        years = [current_year - 1, current_year]

    year_stats = {}
    fields = "paperId,title,authors,year,citationCount"

    for year in years:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = httpx.get(
                    f"{S2_BASE_URL}/paper/search",
                    params={
                        "query": query,
                        "limit": 50,
                        "fields": fields,
                        "year": str(year),
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                total = data.get("total", 0)
                sample = []
                for item in data.get("data", []):
                    sample.append({
                        "title": item.get("title", ""),
                        "authors": [a.get("name", "") for a in item.get("authors", [])][:3],
                        "year": item.get("year"),
                        "citations": item.get("citationCount", 0),
                    })

                year_stats[str(year)] = {
                    "total": total,
                    "sample_papers": sample,
                }
                break  # 成功，跳出重试

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = 2 ** attempt * 3  # 3s, 6s, 12s
                    print(f"[Scholar Trend] {year}年 429限流，{wait}s后重试 ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    print(f"[Scholar Trend] {year}年检索失败: {e}")
                    year_stats[str(year)] = {"total": 0, "sample_papers": []}
                    break
            except httpx.HTTPError as e:
                print(f"[Scholar Trend] {year}年检索失败: {e}")
                year_stats[str(year)] = {"total": 0, "sample_papers": []}
                break
        else:
            # 重试耗尽
            year_stats[str(year)] = {"total": 0, "sample_papers": []}

        # 年份间加间隔，避免连续触发限流
        if year != years[-1]:
            time.sleep(1)

    return {"year_stats": year_stats, "query": query}
