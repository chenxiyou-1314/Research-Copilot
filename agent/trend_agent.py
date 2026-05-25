"""Trend Forecasting Agent — 时间线分析→方法演化追踪→趋势预测

从"这个方向值不值得做"的决策视角，分析论文的时间分布、技术路线演变、
范式转换信号，预测研究方向的热度/饱和度/潜力。
"""
import json
from langchain_core.language_models import BaseLanguageModel
from agent.prompts import TIMELINE_ANALYSIS, METHOD_EVOLUTION, TREND_FORECAST


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
    # 按年份排序
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


def analyze_timeline(
    llm: BaseLanguageModel,
    papers: list[dict],
    query: str,
) -> dict:
    """
    Step 1: 时间线分析 — 按年份统计主题分布，识别新兴/衰退/稳定趋势。

    Returns:
        {
            "year_distribution": {"2022": {"count": 5, "keywords": [...]}, ...},
            "emerging_topics": [...],
            "declining_topics": [...],
            "steady_topics": [...],
        }
    """
    papers_timeline = _build_papers_timeline(papers)

    if not papers_timeline.strip():
        return {
            "year_distribution": {},
            "emerging_topics": [],
            "declining_topics": [],
            "steady_topics": [],
        }

    prompt = TIMELINE_ANALYSIS.format(
        query=query,
        papers_timeline=papers_timeline[:4000],
    )
    response = llm.invoke(prompt)

    result = _parse_json_response(response.content)
    if isinstance(result, dict):
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
    完整的趋势预测流程：时间线分析→方法演化追踪→趋势预测。

    Returns:
        {
            "timeline": {...},
            "evolution": {...},
            "trend_forecast": {...},
        }
    """
    # Step 1: 时间线分析
    timeline = analyze_timeline(llm, papers, query)

    # Step 2: 方法演化追踪
    evolution = track_method_evolution(llm, timeline, decomposition, query)

    # Step 3: 趋势预测
    forecast = forecast_trend(llm, timeline, evolution, gaps, query)

    return {
        "timeline": timeline,
        "evolution": evolution,
        "trend_forecast": forecast,
    }
