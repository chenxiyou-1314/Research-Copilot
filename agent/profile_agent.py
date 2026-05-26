"""Research Profile Graph Agent — 用户研究知识图谱构建

基于用户历史查询、已索引论文、方法解构结果，构建个人知识图谱：
核心领域 → 已掌握方法 → 方法组件关系图 → 知识盲区 → 高相关未读 → 研究风格画像

方法组件关系图：将 Decomposition Agent 拆解的原子组件建模为图结构，
节点 = 方法组件，边 = 相似/关联关系，可视化方法间的联系。
"""
import json
from langchain_core.language_models import BaseLanguageModel
from agent.prompts import PROFILE_EXTRACTION, METHOD_GRAPH_BUILD


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
    decomposition: list[dict] = None,
) -> dict:
    """
    构建用户研究知识图谱。

    Args:
        llm: 语言模型
        query_history: 历史查询列表 [{"query": "...", "time": "..."}, ...]
        papers: 已索引论文列表
        decomposition: 方法解构结果（来自 Decomposition Agent）

    Returns:
        {
            "core_domains": [...],
            "mastered_methods": [...],
            "method_graph": {"nodes": [...], "edges": [...]},
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
    if not isinstance(result, dict):
        result = {}

    # ── 构建方法组件关系图 ──
    method_graph = {"nodes": [], "edges": []}
    if decomposition:
        method_graph = build_method_graph(llm, decomposition)

    return {
        "core_domains": result.get("core_domains", []),
        "mastered_methods": result.get("mastered_methods", []),
        "method_graph": method_graph,
        "knowledge_blindspots": result.get("knowledge_blindspots", []),
        "unread_relevant": result.get("unread_relevant", []),
        "research_style": result.get("research_style", {}),
    }


def build_method_graph(
    llm: BaseLanguageModel,
    decomposition: list[dict],
) -> dict:
    """
    从方法解构结果中构建方法组件关系图。

    节点 = 方法组件（如 ResNet-18、distillation loss、episodic memory）
    边 = 相似/关联关系（如同类组件、互补组件、经常一起使用的组合）
    """
    if not decomposition:
        return {"nodes": [], "edges": []}

    # 格式化解构结果
    decomp_text = ""
    for d in decomposition:
        title = d.get("paper_title", "Unknown")
        year = d.get("paper_year", "?")
        decomp_text += f"\n### {title} ({year})\n"
        components = d.get("components", {})
        for comp_type, comp_detail in components.items():
            if isinstance(comp_detail, dict):
                decomp_text += f"- **{comp_type}**: {comp_detail.get('name', 'N/A')} — {comp_detail.get('detail', '')}\n"
            else:
                decomp_text += f"- **{comp_type}**: {comp_detail}\n"

    prompt = METHOD_GRAPH_BUILD.format(
        decomposition=decomp_text[:4000],
    )
    response = llm.invoke(prompt)

    result = _parse_json_response(response.content)
    if isinstance(result, dict) and "nodes" in result:
        return result

    return {"nodes": [], "edges": []}
