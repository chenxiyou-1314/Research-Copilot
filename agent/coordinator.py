"""Coordinator Agent — 任务规划与子Agent调度"""
from langchain_core.language_models import BaseLanguageModel
from agent.prompts import COORDINATOR_PLAN


def plan_task(llm: BaseLanguageModel, query: str, preferences: str, feedback: str = "") -> dict:
    """
    Coordinator 规划任务：分析用户查询，决定子Agent执行策略。
    
    Returns:
        {
            "search_queries": [...],       # 检索用的改写query列表
            "max_papers": int,             # 检索论文数上限
            "focus_areas": [...],          # 重点关注方向
            "needs_rerun": bool,           # 是否是重跑（Critic不满意）
            "critic_feedback": str,        # Critic的反馈
        }
    """
    prompt = COORDINATOR_PLAN.format(
        query=query,
        preferences=preferences,
        feedback=feedback or "无（首次执行）",
    )
    response = llm.invoke(prompt)
    
    # 解析规划结果
    try:
        import json
        content = response.content
        # 尝试提取JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        plan = json.loads(content.strip())
    except (json.JSONDecodeError, IndexError):
        # 解析失败，用默认规划
        plan = {
            "search_queries": [query],
            "max_papers": 10,
            "focus_areas": [query],
            "needs_rerun": bool(feedback),
            "critic_feedback": feedback,
        }
    
    return plan
