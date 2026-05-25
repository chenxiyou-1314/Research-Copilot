"""Novelty Agent — 研究思路发现：Gap分析→跨域迁移→思路生成→新颖性验证"""
import json
from langchain_core.language_models import BaseLanguageModel
from tools.arxiv_search import search_arxiv
from tools.scholar_search import search_scholar
from agent.prompts import GAP_ANALYSIS, CROSS_DOMAIN_TRANSFER, IDEA_GENERATION, NOVELTY_VERIFICATION


def analyze_gaps(llm: BaseLanguageModel, summary: str, query: str) -> dict:
    """
    Step 1: Gap分析 — 从综述中提取未解决问题、方法局限、数据空白。
    
    Returns:
        {
            "methodological_gaps": [...],    # 方法学空白
            "data_gaps": [...],              # 数据/评估空白
            "theoretical_gaps": [...],       # 理论空白
            "practical_gaps": [...],         # 实践/应用空白
        }
    """
    prompt = GAP_ANALYSIS.format(query=query, summary=summary)
    response = llm.invoke(prompt)
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        gaps = json.loads(content.strip())
    except (json.JSONDecodeError, ValueError, IndexError):
        gaps = {
            "methodological_gaps": ["无法解析，请重试"],
            "data_gaps": [],
            "theoretical_gaps": [],
            "practical_gaps": [],
        }
    
    return gaps


def cross_domain_transfer(
    llm: BaseLanguageModel,
    query: str,
    gaps: dict,
    summary: str,
) -> list[dict]:
    """
    Step 2: 跨域迁移 — 将其他领域的成功思路迁移到当前领域。
    
    Returns:
        [
            {
                "source_domain": "NLP",
                "source_method": "Prompt Tuning",
                "target_application": "细粒度CIL的prompt-guided feature adaptation",
                "why_transferable": "两者都面临预训练-微调的稳定性-可塑性困境",
                "expected_benefit": "避免全量微调，减少灾难性遗忘",
            },
            ...
        ]
    """
    prompt = CROSS_DOMAIN_TRANSFER.format(
        query=query,
        gaps=json.dumps(gaps, ensure_ascii=False, indent=2),
        summary=summary[:2000],  # 限制长度
    )
    response = llm.invoke(prompt)
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        transfers = json.loads(content.strip())
        if isinstance(transfers, list):
            return transfers[:3]  # 最多3个迁移方向
    except (json.JSONDecodeError, ValueError):
        pass
    
    return []


def generate_research_ideas(
    llm: BaseLanguageModel,
    query: str,
    gaps: dict,
    transfers: list[dict],
    summary: str,
) -> list[dict]:
    """
    Step 3: 思路生成 — 提出2-3个具体可执行的研究方向。
    
    Returns:
        [
            {
                "title": "基于Prompt-Guided Feature Adaptation的细粒度CIL",
                "motivation": "现有CIL方法在细粒度场景下特征判别性不足...",
                "technical_route": ["Step1: 设计prompt-guided feature extractor", ...],
                "expected_contribution": "首次将prompt tuning引入细粒度CIL，实现...",
                "feasibility": "高 — 仅需单GPU，CUB-200/Ship数据集可用",
                "risk": "prompt设计可能对特定领域敏感",
            },
            ...
        ]
    """
    prompt = IDEA_GENERATION.format(
        query=query,
        gaps=json.dumps(gaps, ensure_ascii=False, indent=2),
        transfers=json.dumps(transfers, ensure_ascii=False, indent=2),
        summary=summary[:2000],
    )
    response = llm.invoke(prompt)
    
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        ideas = json.loads(content.strip())
        if isinstance(ideas, list):
            return ideas[:3]  # 最多3个思路
    except (json.JSONDecodeError, ValueError):
        pass
    
    return []


def verify_novelty(
    llm: BaseLanguageModel,
    ideas: list[dict],
) -> list[dict]:
    """
    Step 4: 新颖性验证 — 将生成的思路与已有论文对比，确认没有被做过。
    
    Returns:
        [
            {
                "idea_title": "...",
                "is_novel": True,
                "similar_works": ["Paper X 做了类似但不同的事..."],
                "novelty_statement": "与Paper X不同，本思路强调...",
                "confidence": 0.7,
            },
            ...
        ]
    """
    verified_ideas = []
    
    for idea in ideas:
        # 用思路的关键词检索，看有没有已存在的类似工作
        search_query = idea.get("title", "")
        if not search_query:
            continue
        
        # 检索已有论文
        try:
            arxiv_results = search_arxiv(search_query, max_results=5)
            scholar_results = search_scholar(search_query, max_results=5)
        except Exception:
            arxiv_results = []
            scholar_results = []
        
        # 合并检索结果
        existing_works = []
        for p in (arxiv_results + scholar_results)[:5]:
            existing_works.append({
                "title": p.get("title", ""),
                "abstract": p.get("abstract", "")[:200],
                "year": p.get("year", ""),
            })
        
        # 让LLM判断新颖性
        prompt = NOVELTY_VERIFICATION.format(
            idea_title=idea.get("title", ""),
            idea_motivation=idea.get("motivation", ""),
            idea_route=idea.get("technical_route", ""),
            existing_works=json.dumps(existing_works, ensure_ascii=False, indent=2),
        )
        response = llm.invoke(prompt)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
            verified_ideas.append({
                "idea_title": idea.get("title", ""),
                "is_novel": result.get("is_novel", True),
                "similar_works": result.get("similar_works", []),
                "novelty_statement": result.get("novelty_statement", ""),
                "confidence": result.get("confidence", 0.5),
                "original_idea": idea,
            })
        except (json.JSONDecodeError, ValueError):
            verified_ideas.append({
                "idea_title": idea.get("title", ""),
                "is_novel": True,
                "similar_works": [],
                "novelty_statement": "新颖性验证解析失败，请人工确认",
                "confidence": 0.3,
                "original_idea": idea,
            })
    
    return verified_ideas


def run_novelty_analysis(
    llm: BaseLanguageModel,
    summary: str,
    query: str,
) -> dict:
    """
    完整的新思路发现流程：Gap分析→跨域迁移→思路生成→新颖性验证。
    
    Returns:
        {
            "gaps": {...},
            "transfers": [...],
            "ideas": [...],
            "verified_ideas": [...],
        }
    """
    # Step 1: Gap分析
    gaps = analyze_gaps(llm, summary, query)
    
    # Step 2: 跨域迁移
    transfers = cross_domain_transfer(llm, query, gaps, summary)
    
    # Step 3: 思路生成
    ideas = generate_research_ideas(llm, query, gaps, transfers, summary)
    
    # Step 4: 新颖性验证
    verified_ideas = verify_novelty(llm, ideas)
    
    return {
        "gaps": gaps,
        "transfers": transfers,
        "ideas": ideas,
        "verified_ideas": verified_ideas,
    }
