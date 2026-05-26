"""Trend Forecasting Agent — 大规模统计检索→时间线分析→方法演化追踪→趋势预测

趋势分析不再依赖综述检索的小样本（10篇），而是单独做一轮大规模检索：
- Semantic Scholar: 利用 total 字段获取每年论文总量（全领域统计）+ 50篇样本
- arXiv: 每年份50篇样本
- 两者合并后送入 LLM 做分析，确保趋势判断基于领域级数据而非局部样本。
"""
import json
from datetime import datetime
from langchain_core.language_models import BaseLanguageModel
from agent.prompts import TIMELINE_ANALYSIS, METHOD_EVOLUTION, TREND_FORECAST
from tools.scholar_search import search_trend_stats
from tools.arxiv_search import search_arxiv_trend


def _parse_json_response(content: str):
    """从LLM响应中提取JSON"""
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())
    except (json.JSONDecodeError, ValueError, IndexError):
        return None


def _build_papers_timeline(papers: list[dict]) -> str:
    """将论文列表按年份排列，构建时间线文本"""
    sorted_papers = sorted(papers, key=lambda p: p.get("year", 0))

    timeline = ""
    current_year = None
    for p in sorted_papers:
        year = p.get("year", "unknown")
        if year != current_year:
            current_year = year
            timeline += f"\n{'='*40}\n📅 {year}年\n{'='*40}\n"
        authors_str = ', '.join(p.get("authors", [])[:3])
        title = p.get("title", "Unknown")
        abstract = p.get("abstract", "")[:200]
        timeline += f"  [{authors_str}] {title}\n    摘要: {abstract}...\n\n"

    return timeline


def _build_trend_stats_text(trend_stats: dict) -> str:
    """将大规模趋势统计数据格式化为文本，供 LLM 分析"""
    year_stats = trend_stats.get("year_stats", {})
    lines = [f"研究方向: {trend_stats.get('query', '')}\n"]
    lines.append("=== 领域级论文发表统计（来源: Semantic Scholar + arXiv）===\n")

    for year in sorted(year_stats.keys()):
        stat = year_stats[year]
        total = stat.get("total", 0)
        sample = stat.get("sample_papers", [])
        lines.append(f"\n📅 {year}年 — Semantic Scholar 命中总量: {total} 篇")
        lines.append(f"  样本论文 ({len(sample)} 篇):")
        for p in sample[:15]:  # 限制每年份展示15篇，避免过长
            authors = ', '.join(p.get("authors", [])[:2])
            title = p.get("title", "Unknown")[:80]
            lines.append(f"    [{authors}] {title}")

    return "\n".join(lines)


def fetch_trend_stats(query: str) -> dict:
    """
    执行趋势专用的大规模检索。
    返回合并后的 year_stats，S2 的 total 是领域级总量（最关键），
    arXiv 的样本作为补充。
    """
    current_year = datetime.now().year
    years = [current_year - 1, current_year]

    # S2: 有 total 字段，是最可靠的领域级数据
    s2_stats = search_trend_stats(query, years=years)
    # arXiv: 补充样本
    arxiv_stats = search_arxiv_trend(query, years=years)

    # 合并：以 S2 的 total 为准，arXiv 样本追加到 sample_papers
    merged = s2_stats.copy()
    s2_years = merged.get("year_stats", {})
    arxiv_years = arxiv_stats.get("year_stats", {})

    for year in years:
        y_str = str(year)
        if y_str in s2_years and y_str in arxiv_years:
            # 去重合并样本（按标题去重）
            existing_titles = {p.get("title", "").lower().strip() for p in s2_years[y_str].get("sample_papers", [])}
            for p in arxiv_years[y_str].get("sample_papers", []):
                if p.get("title", "").lower().strip() not in existing_titles:
                    s2_years[y_str]["sample_papers"].append(p)
                    existing_titles.add(p.get("title", "").lower().strip())

    return merged


def analyze_timeline(
    llm: BaseLanguageModel,
    papers: list[dict],
    query: str,
    trend_stats: dict = None,
) -> dict:
    """
    Step 1: 时间线分析 — 基于领域级统计数据 + 本地论文，识别趋势。

    优先使用 trend_stats（S2 全领域 total + 大样本），本地论文作为补充。

    Returns:
        {
            "year_distribution": {"2022": {"count": 5, "keywords": [...]}, ...},
            "emerging_topics": [...],
            "declining_topics": [...],
            "steady_topics": [...],
        }
    """
    # 构建输入文本：优先用领域级统计
    if trend_stats and trend_stats.get("year_stats"):
        analysis_text = _build_trend_stats_text(trend_stats)
        # 如果本地论文有更多年份跨度，追加补充
        local_timeline = _build_papers_timeline(papers)
        if local_timeline.strip():
            analysis_text += "\n\n=== 本地已索引论文（补充）===\n" + local_timeline[:2000]
    else:
        # fallback: 仍用本地论文
        analysis_text = _build_papers_timeline(papers)

    if not analysis_text.strip():
        return {
            "year_distribution": {},
            "emerging_topics": [],
            "declining_topics": [],
            "steady_topics": [],
        }

    prompt = TIMELINE_ANALYSIS.format(
        query=query,
        papers_timeline=analysis_text[:6000],
    )
    response = llm.invoke(prompt)

    result = _parse_json_response(response.content)
    if isinstance(result, dict):
        # 将领域级 total 注入 year_distribution
        if trend_stats and trend_stats.get("year_stats"):
            for year, stat in trend_stats["year_stats"].items():
                if year not in result.get("year_distribution", {}):
                    result.setdefault("year_distribution", {})[year] = {
                        "count": stat.get("total", 0),
                        "keywords": [],
                    }
                else:
                    # 用 S2 total 覆盖 count，更准确
                    result["year_distribution"][year]["total_in_field"] = stat.get("total", 0)
        return result

    return {
        "year_distribution": {},
        "emerging_topics": [],
        "declining_topics": [],
        "steady_topics": [],
    }


def track_method_evolution(
    llm: BaseLanguageModel,
    timeline: dict,
    decomposition: list[dict],
    query: str,
) -> dict:
    """
    Step 2: 方法演化追踪 — 追踪技术路线演变，识别范式转换点。

    Returns:
        {
            "evolution_paths": [...],
            "current_paradigm": {...},
            "next_paradigm_hints": [...],
        }
    """
    timeline_text = json.dumps(timeline, ensure_ascii=False, indent=2)[:2000]

    # 将解构结果格式化
    decomp_text = ""
    for d in decomposition:
        decomp_text += f"\n{d.get('paper_title', '?')} ({d.get('paper_year', '?')}): "
        components = d.get("components", {})
        comp_summaries = []
        for comp_name, comp_detail in components.items():
            if isinstance(comp_detail, dict):
                comp_summaries.append(f"{comp_name}={comp_detail.get('name', 'N/A')}")
            else:
                comp_summaries.append(f"{comp_name}={comp_detail}")
        decomp_text += ", ".join(comp_summaries)

    prompt = METHOD_EVOLUTION.format(
        query=query,
        timeline_result=timeline_text,
        decomposition=decomp_text[:2000],
    )
    response = llm.invoke(prompt)

    result = _parse_json_response(response.content)
    if isinstance(result, dict):
        return result

    return {
        "evolution_paths": [],
        "current_paradigm": {},
        "next_paradigm_hints": [],
    }


def forecast_trend(
    llm: BaseLanguageModel,
    timeline: dict,
    evolution: dict,
    gaps: dict,
    query: str,
) -> dict:
    """
    Step 3: 趋势预测 — 综合时间线和演化分析，预测方向趋势。

    Returns:
        {
            "direction_score": {"heat": 4, "saturation": 2, "potential": 5, "entry_barrier": 3},
            "overall_phase": "上升期",
            "phase_reasoning": "...",
            "forecast": [...],
            "investment_advice": {...},
            "red_flags": [...],
            "green_flags": [...],
        }
    """
    timeline_text = json.dumps(timeline, ensure_ascii=False, indent=2)[:1500]
    evolution_text = json.dumps(evolution, ensure_ascii=False, indent=2)[:1500]
    gaps_text = json.dumps(gaps, ensure_ascii=False, indent=2)[:1000] if gaps else "暂无Gap分析"

    prompt = TREND_FORECAST.format(
        query=query,
        timeline_result=timeline_text,
        evolution_result=evolution_text,
        gaps=gaps_text,
    )
    response = llm.invoke(prompt)

    result = _parse_json_response(response.content)
    if isinstance(result, dict):
        return result

    return {
        "direction_score": {"heat": 3, "saturation": 3, "potential": 3, "entry_barrier": 3},
        "overall_phase": "未知",
        "phase_reasoning": "趋势预测解析失败",
        "forecast": [],
        "investment_advice": {},
        "red_flags": [],
        "green_flags": [],
    }


def run_trend_forecast(
    llm: BaseLanguageModel,
    papers: list[dict],
    decomposition: list[dict],
    gaps: dict,
    query: str,
) -> dict:
    """
    完整的趋势预测流程：大规模统计检索→时间线分析→方法演化追踪→趋势预测。

    Step 0: 对 query 单独做一轮大规模检索（S2 + arXiv），获取领域级数据
    Step 1: 时间线分析（基于领域级统计 + 本地论文）
    Step 2: 方法演化追踪
    Step 3: 趋势预测

    Returns:
        {
            "trend_stats": {...},   # 领域级统计数据
            "timeline": {...},
            "evolution": {...},
            "trend_forecast": {...},
        }
    """
    # Step 0: 领域级大规模检索（近两年）
    print(f"[Trend] 执行领域级统计检索: {query}")
    trend_stats = fetch_trend_stats(query)
    for year, stat in trend_stats.get("year_stats", {}).items():
        print(f"  {year}年: S2 total={stat.get('total', 0)}, 样本={len(stat.get('sample_papers', []))}篇")

    # Step 1: 时间线分析（传入 trend_stats）
    timeline = analyze_timeline(llm, papers, query, trend_stats=trend_stats)

    # Step 2: 方法演化追踪
    evolution = track_method_evolution(llm, timeline, decomposition, query)

    # Step 3: 趋势预测
    forecast = forecast_trend(llm, timeline, evolution, gaps, query)

    return {
        "trend_stats": trend_stats,
        "timeline": timeline,
        "evolution": evolution,
        "trend_forecast": forecast,
    }
