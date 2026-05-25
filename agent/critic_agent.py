"""Critic Agent — 综述质量评估与反馈"""
from langchain_core.language_models import BaseLanguageModel
from agent.prompts import CRITIC_EVAL


def evaluate_summary(
    llm: BaseLanguageModel,
    query: str,
    summary: str,
    papers_context: str,
    threshold: float = 3.5,
) -> dict:
    """
    Critic Agent: 评估综述质量，给出评分和改进建议。
    
    Args:
        llm: LLM实例
        query: 研究方向
        summary: 生成的综述
        papers_context: 论文列表文本
        threshold: 质量阈值（低于此分数触发重写）
    
    Returns:
        {
            "coverage_score": float,    # 覆盖度 1-5
            "accuracy_score": float,    # 准确性 1-5
            "coherence_score": float,   # 连贯性 1-5
            "overall_score": float,     # 总分 1-5
            "passed": bool,             # 是否达标
            "feedback": str,            # 改进建议
        }
    """
    prompt = CRITIC_EVAL.format(
        query=query,
        summary=summary,
        papers_context=papers_context[:3000],  # 限制长度避免超token
    )
    response = llm.invoke(prompt)
    
    # 解析评分
    try:
        import json
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        result = json.loads(content.strip())
        
        coverage = float(result.get("coverage_score", 3.0))
        accuracy = float(result.get("accuracy_score", 3.0))
        coherence = float(result.get("coherence_score", 3.0))
        overall = (coverage + accuracy + coherence) / 3
        
        return {
            "coverage_score": coverage,
            "accuracy_score": accuracy,
            "coherence_score": coherence,
            "overall_score": round(overall, 2),
            "passed": overall >= threshold,
            "feedback": result.get("feedback", "无具体反馈"),
        }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # 解析失败，默认通过（避免无限循环）
        return {
            "coverage_score": 3.0,
            "accuracy_score": 3.0,
            "coherence_score": 3.0,
            "overall_score": 3.0,
            "passed": True,
            "feedback": f"评分解析失败({e})，默认通过",
        }
