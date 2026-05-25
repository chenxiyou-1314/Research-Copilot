"""FastAPI 入口 — SSE 流式输出 + REST 接口"""
import json
import asyncio
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.graph import agent
from memory.paper_store import PaperStore
from memory.user_profile import UserProfile
from tools.vector_store import VectorStore
import config

app = FastAPI(title="Research Copilot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

paper_store = PaperStore()
user_profile = UserProfile()
vector_store = VectorStore()


# ── 请求模型 ──
class ResearchRequest(BaseModel):
    query: str
    max_papers: int = 10


class QARequest(BaseModel):
    question: str
    top_k: int = 10


class SearchRequest(BaseModel):
    query: str
    max_results: int = 10


# ── SSE 流式综述生成 ──
@app.post("/research/stream")
async def research_stream(req: ResearchRequest):
    """SSE 流式输出综述生成过程。"""
    
    async def event_generator():
        # 初始化状态
        state = {
            "query": req.query,
        }
        
        # 逐步执行 Agent
        yield f"data: {json.dumps({'step': 'start', 'message': f'开始检索: {req.query}'}, ensure_ascii=False)}\n\n"
        
        try:
            # 运行 Agent（LangGraph 自动编排节点）
            result = await asyncio.to_thread(
                agent.invoke, state
            )
            
            # 流式输出各阶段结果
            if result.get("papers"):
                yield f"data: {json.dumps({'step': 'search', 'count': len(result['papers'])}, ensure_ascii=False)}\n\n"
            
            if result.get("filtered_papers"):
                titles = [p.get("title", "")[:50] for p in result["filtered_papers"][:5]]
                yield f"data: {json.dumps({'step': 'filter', 'count': len(result['filtered_papers']), 'top_papers': titles}, ensure_ascii=False)}\n\n"
            
            if result.get("new_papers_count", 0) > 0:
                yield f"data: {json.dumps({'step': 'index', 'new': result['new_papers_count'], 'total': result.get('total_papers_count', 0)}, ensure_ascii=False)}\n\n"
            
            # 输出最终综述
            summary = result.get("summary", result.get("answer", "生成失败"))
            yield f"data: {json.dumps({'step': 'done', 'result': summary}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'step': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── 论文检索 ──
@app.post("/papers/search")
async def search_papers(req: SearchRequest):
    """检索论文（不生成综述）。"""
    from tools.arxiv_search import search_arxiv
    from tools.scholar_search import search_scholar
    
    arxiv_results = search_arxiv(req.query, max_results=req.max_results)
    scholar_results = search_scholar(req.query, max_results=req.max_results)
    
    # 去重
    seen = set()
    all_papers = []
    for p in arxiv_results + scholar_results:
        if p["paper_id"] not in seen:
            seen.add(p["paper_id"])
            all_papers.append(p)
    
    return {"papers": all_papers, "total": len(all_papers)}


# ── 跨论文 QA ──
@app.post("/papers/qa")
async def papers_qa(req: QARequest):
    """基于论文库的 RAG 问答。"""
    rag_context = vector_store.search(req.question, top_k=req.top_k)
    
    if not rag_context:
        return {"answer": "论文库中暂无相关内容，请先检索并索引论文。", "has_context": False}
    
    from tools.summary_writer import generate_qa_answer
    from agent.graph import _get_llm, llm
    answer = generate_qa_answer(llm, req.question, rag_context)
    
    return {"answer": answer, "has_context": True}


# ── 论文库状态 ──
@app.get("/papers/status")
async def papers_status():
    """查看论文库状态。"""
    return {
        "total_papers": paper_store.count(),
        "indexed_papers": vector_store.count(),
        "user_interests": user_profile.interests,
        "user_venues": user_profile.venues,
    }


# ── 健康检查 ──
@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
