"""综述 / QA 生成工具"""
from langchain_core.language_models import BaseLanguageModel

from agent.prompts import LITERATURE_SUMMARY, QA_PROMPT


def generate_summary(
    llm: BaseLanguageModel,
    query: str,
    papers_context: str,
    rag_context: str,
) -> str:
    """
    生成结构化文献综述。
    
    Args:
        llm: LangChain LLM 实例
        query: 研究方向
        papers_context: 论文列表文本
        rag_context: RAG 检索的补充上下文
    
    Returns:
        综述文本
    """
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
    """
    基于论文库回答问题。
    
    Args:
        llm: LangChain LLM 实例
        question: 用户问题
        rag_context: RAG 检索的上下文
    
    Returns:
        回答文本
    """
    prompt = QA_PROMPT.format(
        question=question,
        rag_context=rag_context or "（暂无相关论文上下文）",
    )
    response = llm.invoke(prompt)
    return response.content
