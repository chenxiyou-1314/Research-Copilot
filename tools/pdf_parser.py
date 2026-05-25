"""PDF 下载 + 解析 + 切分工具"""
import os
import httpx
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config


def download_pdf(paper: dict) -> Optional[str]:
    """
    下载论文 PDF 到本地。
    
    Returns:
        本地 PDF 路径，失败返回 None
    """
    pdf_url = paper.get("pdf_url", "")
    if not pdf_url:
        return None
    
    paper_id = paper["paper_id"].replace("/", "_")
    pdf_path = os.path.join(config.PDF_DOWNLOAD_DIR, f"{paper_id}.pdf")
    
    if os.path.exists(pdf_path):
        return pdf_path
    
    try:
        resp = httpx.get(pdf_url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(resp.content)
        return pdf_path
    except httpx.HTTPError as e:
        print(f"[PDF] 下载失败 {paper['title']}: {e}")
        return None


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从 PDF 提取文本内容。
    优先使用 pdfminer，失败则降级到纯文本读取。
    """
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        if text.strip():
            return text
    except Exception as e:
        print(f"[PDF] pdfminer 解析失败: {e}")
    
    # 降级: 尝试 unstructured
    try:
        from unstructured.partition.auto import partition
        elements = partition(filename=pdf_path)
        text = "\n".join(str(el) for el in elements)
        if text.strip():
            return text
    except Exception as e:
        print(f"[PDF] unstructured 解析失败: {e}")
    
    return ""


def chunk_text(text: str, metadata: dict = None) -> list[dict]:
    """
    将长文本切分为适合向量化的 chunk。
    
    Args:
        text: 原始文本
        metadata: 附加元数据（paper_id, title 等）
    
    Returns:
        chunk 列表，每个 chunk 包含 content + metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[metadata or {}],
    )
    
    return [
        {
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
        if chunk.page_content.strip()
    ]


def parse_papers(paper: dict) -> list[dict]:
    """
    完整流程: 下载 PDF → 提取文本 → 切分。
    
    Args:
        paper: 论文元数据字典
    
    Returns:
        chunk 列表
    """
    pdf_path = download_pdf(paper)
    if not pdf_path:
        # PDF 下载失败，用 abstract 代替
        if paper.get("abstract"):
            return chunk_text(
                paper["abstract"],
                metadata={
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "source": "abstract_only",
                },
            )
        return []
    
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        if paper.get("abstract"):
            text = paper["abstract"]
        else:
            return []
    
    return chunk_text(
        text,
        metadata={
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "source": "pdf",
        },
    )
