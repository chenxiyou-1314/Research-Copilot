"""Method Decomposition Agent — 方法解构→跨论文重组→可行性验证

将论文中的方法拆解为原子组件（backbone/training_strategy/loss/augmentation/evaluation），
再跨论文重组，生成新方法方案，最后验证可行性。
"""
import json
from langchain_core.language_models import BaseLanguageModel
from agent.prompts import METHOD_DECOMPOSITION, METHOD_RECOMBINATION, RECOMBINATION_VALIDATION


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


def decompose_methods(
    llm: BaseLanguageModel,
    papers_context: str,
    query: str,
) -> list[dict]:
    """
    Step 1: 方法解构 — 将每篇论文的方法拆解为5个原子组件。

    Returns:
        [
            {
                "paper_title": "AANets",
                "paper_year": 2020,
                "components": {
                    "backbone": {"name": "ResNet-18", "detail": "...", ...},
                    "training_strategy": {...},
                    "loss_function": {...},
                    "data_augmentation": {...},
                    "evaluation_protocol": {...},
                }
            },
            ...
        ]
    """
    prompt = METHOD_DECOMPOSITION.format(
        query=query,
        papers_context=papers_context[:4000],  # 限制长度防超token
    )
    response = llm.invoke(prompt)

    result = _parse_json_response(response.content)
    if isinstance(result, list):
        return result[:8]  # 最多8篇论文的解构

    return []


def recombine_methods(
    llm: BaseLanguageModel,
    decomposition: list[dict],
    gaps: dict,
    query: str,
) -> list[dict]:
    """
    Step 2: 方法重组 — 从不同论文中选取组件重新组合，生成新方法方案。

    Returns:
        [
            {
                "name": "ProtoCIL-Prompt",
                "components": {
                    "backbone": "来自iVoro的Voronoi原型backbone",
                    "training_strategy": "来自AANets的stable-plastic双路聚合",
                    ...
                },
                "source_papers": ["iVoro", "AANets"],
                "motivation": "...",
                "expected_synergy": "...",
                "target_gap": "方法学空白#1: 细粒度场景下原型表示不足",
            },
            ...
        ]
    """
    # 将解构结果格式化为可读文本
    decomp_text = ""
    for d in decomposition:
        decomp_text += f"\n### {d.get('paper_title', 'Unknown')} ({d.get('paper_year', '?')})\n"
        components = d.get("components", {})
        for comp_name, comp_detail in components.items():
            if isinstance(comp_detail, dict):
                decomp_text += f"- **{comp_name}**: {comp_detail.get('name', 'N/A')} — {comp_detail.get('detail', '')}\n"
            else:
                decomp_text += f"- **{comp_name}**: {comp_detail}\n"

    gaps_text = json.dumps(gaps, ensure_ascii=False, indent=2) if gaps else "暂无Gap分析"

    prompt = METHOD_RECOMBINATION.format(
        query=query,
        decomposition=decomp_text[:3000],
        gaps=gaps_text[:1500],
    )
    response = llm.invoke(prompt)

    result = _parse_json_response(response.content)
    if isinstance(result, list):
        return result[:4]  # 最多4个重组方案

    return []


def validate_recombinations(
    llm: BaseLanguageModel,
    recombinations: list[dict],
    query: str,
) -> list[dict]:
    """
    Step 3: 可行性验证 — 评估每个重组方案的技术可行性、兼容性、风险。

    Returns:
        [
            {
                "name": "ProtoCIL-Prompt",
                "compatibility_score": 4,
                "implementation_difficulty": "中等",
                "risk_factors": ["风险1", "风险2"],
                "mitigation": "...",
                "overall_feasibility": "高",
                "quick_start": "...",
                "potential_pitfall": "...",
            },
            ...
        ]
    """
    recomb_text = ""
    for r in recombinations:
        recomb_text += f"\n### {r.get('name', 'Unknown')}\n"
        for comp_name, comp_detail in r.get("components", {}).items():
            recomb_text += f"- **{comp_name}**: {comp_detail}\n"
        recomb_text += f"- **来源论文**: {', '.join(r.get('source_papers', []))}\n"
        recomb_text += f"- **动机**: {r.get('motivation', '')}\n"
        recomb_text += f"- **预期协同**: {r.get('expected_synergy', '')}\n"
        recomb_text += f"- **目标Gap**: {r.get('target_gap', '')}\n"

    prompt = RECOMBINATION_VALIDATION.format(
        query=query,
        recombination=recomb_text[:3000],
    )
    response = llm.invoke(prompt)

    result = _parse_json_response(response.content)
    if isinstance(result, list):
        # 将验证结果与原始方案合并
        validated = []
        for i, v in enumerate(result):
            merged = {
                "name": v.get("name", ""),
                "compatibility_score": v.get("compatibility_score", 3),
                "implementation_difficulty": v.get("implementation_difficulty", "未知"),
                "risk_factors": v.get("risk_factors", []),
                "mitigation": v.get("mitigation", ""),
                "overall_feasibility": v.get("overall_feasibility", "中"),
                "quick_start": v.get("quick_start", ""),
                "potential_pitfall": v.get("potential_pitfall", ""),
                # 保留原始方案信息
                "original_recombination": recombinations[i] if i < len(recombinations) else {},
            }
            validated.append(merged)
        return validated

    return []


def run_decomposition(
    llm: BaseLanguageModel,
    papers_context: str,
    gaps: dict,
    query: str,
) -> dict:
    """
    完整的方法解构与重组流程：解构→重组→验证。

    Returns:
        {
            "decomposition": [...],          # 方法解构结果
            "recombinations": [...],         # 重组方案
            "validated_recombinations": [...], # 验证后的方案
        }
    """
    # Step 1: 方法解构
    decomposition = decompose_methods(llm, papers_context, query)

    if not decomposition:
        return {
            "decomposition": [],
            "recombinations": [],
            "validated_recombinations": [],
        }

    # Step 2: 方法重组
    recombinations = recombine_methods(llm, decomposition, gaps, query)

    if not recombinations:
        return {
            "decomposition": decomposition,
            "recombinations": [],
            "validated_recombinations": [],
        }

    # Step 3: 可行性验证
    validated = validate_recombinations(llm, recombinations, query)

    return {
        "decomposition": decomposition,
        "recombinations": recombinations,
        "validated_recombinations": validated,
    }
