"""Research Profile Graph Agent — 用户研究知识图谱构建

基于用户历史查询和已索引论文，构建个人知识图谱：
核心领域 → 已掌握方法 → 知识盲区 → 高相关未读 → 研究风格画像
"""
import json
from langchain_core.language_models import BaseLanguageModel
from agent.prompts import PROFILE_EXTRACTION


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


def build_profile_graph(
    llm: BaseLanguageModel,
    query_history: list[dict],
    papers: list[dict],
) -> dict:
    """
    构建用户研究知识图谱。

    Args:
        llm: 语言模型
        query_history: 历史查询列表 [{"query": "...", "time": "..."}, ...]
        papers: 已索引论文列表

    Returns:
        {
            "core_domains": [...],
            "mastered_methods": [...],
            "knowledge_blindspots": [...],
            "unread_relevant": [...],
            "research_style": {...},
        }
    """
    # 构建查询历史文本
    query_text = ""
    for qh in query_history[:20]:
        query_text += f"  - {qh.get('query', '')} ({qh.get('time', '')[:10]})\n"
    if not query_text.strip():
        query_text = "  - 暂无历史查询"

    # 构建论文列表文本
    papers_text = ""
    for p in papers[:15]:
        authors_str = ', '.join(p.get("authors", [])[:3])
        papers_text += f"  - [{authors_str}, {p.get('year', '?')}] {p.get('title', '?')}\n"
    if not papers_text.strip():
        papers_text = "  - 暂无已索引论文"

    prompt = PROFILE_EXTRACTION.format(
        query_history=query_text,
        papers_list=papers_text,
    )
    response = llm.invoke(prompt)

    result = _parse_json_response(response.content)
    if isinstance(result, dict):
        return result

    return {
        "core_domains": [],
        "mastered_methods": [],
        "knowledge_blindspots": [],
        "unread_relevant": [],
        "research_style": {},
    }
