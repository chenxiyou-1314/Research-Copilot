"""Writing Agent — 综述生成、格式化、修订"""
from langchain_core.language_models import BaseLanguageModel
from agent.prompts import LITERATURE_SUMMARY, QA_PROMPT, REVISION_PROMPT


def generate_summary(
    llm: BaseLanguageModel,
    query: str,
    papers_context: str,
    rag_context: str,
) -> str:
    """生成结构化文献综述。"""
    prompt = LITERATURE_SUMMARY.format(
        query=query,
        papers_context=papers_context,
        rag_context=rag_context or "（暂无RAG补充上下文）",
    )
    response = llm.invoke(prompt)
    return response.content


def generate_qa_answer(
    llm: BaseLanguageModel,
    question: str,
    rag_context: str,
) -> str:
    """基于论文库回答问题。"""
    prompt = QA_PROMPT.format(
        question=question,
        rag_context=rag_context or "（暂无相关论文上下文）",
    )
    response = llm.invoke(prompt)
    return response.content


def revise_summary(
    llm: BaseLanguageModel,
    original_summary: str,
    critic_feedback: str,
    query: str,
    rag_context: str,
) -> str:
    """
    根据Critic反馈修订综述。
    
    Args:
        llm: LLM实例
        original_summary: 原始综述
        critic_feedback: Critic的改进建议
        query: 研究方向
        rag_context: RAG上下文
    
    Returns:
        修订后的综述
    """
    prompt = REVISION_PROMPT.format(
        original_summary=original_summary,
        critic_feedback=critic_feedback,
        query=query,
        rag_context=rag_context or "（暂无补充上下文）",
    )
    response = llm.invoke(prompt)
    return response.content
