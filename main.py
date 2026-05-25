"""FastAPI 入口 — SSE 流式输出 + REST 接口"""
import json
import asyncio
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
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
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>Research Copilot</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
            .container { max-width: 800px; margin: 0 auto; padding: 60px 20px; }
            h1 { font-size: 2em; margin-bottom: 8px; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .subtitle { color: #94a3b8; margin-bottom: 40px; }
            .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 16px; transition: border-color 0.2s; }
            .card:hover { border-color: #38bdf8; }
            .card h3 { color: #38bdf8; margin-bottom: 8px; }
            .card p { color: #94a3b8; font-size: 14px; }
            .card code { background: #0f172a; padding: 2px 8px; border-radius: 4px; font-size: 13px; color: #a78bfa; }
            .search-box { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; margin-bottom: 30px; }
            .search-box input { width: 100%; padding: 12px 16px; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: #e2e8f0; font-size: 15px; outline: none; }
            .search-box input:focus { border-color: #38bdf8; }
            .search-box button { margin-top: 12px; padding: 10px 24px; border-radius: 8px; border: none; background: linear-gradient(135deg, #38bdf8, #818cf8); color: #0f172a; font-weight: 600; cursor: pointer; font-size: 14px; }
            .search-box button:hover { opacity: 0.9; }
            .result { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-top: 16px; white-space: pre-wrap; font-size: 14px; line-height: 1.8; display: none; max-height: 600px; overflow-y: auto; }
            .status { display: flex; gap: 20px; margin-bottom: 30px; }
            .status-item { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px 20px; flex: 1; text-align: center; }
            .status-item .num { font-size: 24px; font-weight: 700; color: #38bdf8; }
            .status-item .label { font-size: 12px; color: #94a3b8; margin-top: 4px; }
            .loading { color: #38bdf8; display: none; margin-top: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Research Copilot</h1>
            <p class="subtitle">基于 LangGraph 的科研文献智能检索与综述生成 Agent</p>

            <div class="status">
                <div class="status-item"><div class="num" id="paper-count">-</div><div class="label">已索引论文</div></div>
                <div class="status-item"><div class="num" id="indexed-count">-</div><div class="label">向量库</div></div>
            </div>

            <div class="search-box">
                <input type="text" id="query" placeholder="输入研究方向，如：class-incremental learning for fine-grained detection" />
                <button onclick="doSearch()">生成综述</button>
                <div class="loading" id="loading">⏳ 正在检索、解析论文并生成综述，首次运行需要下载PDF和构建向量索引，请耐心等待1-3分钟...</div>
            </div>
            <div class="result" id="result"></div>

            <h2 style="margin-top:20px;margin-bottom:16px;font-size:1.2em;">API 接口</h2>
            <div class="card">
                <h3>POST /research/stream</h3>
                <p>流式综述生成（SSE）</p>
                <code>curl -N -X POST http://localhost:8000/research/stream -H "Content-Type: application/json" -d '{"query":"你的研究方向","max_papers":5}'</code>
            </div>
            <div class="card">
                <h3>POST /papers/search</h3>
                <p>论文检索（不生成综述）</p>
                <code>curl -X POST http://localhost:8000/papers/search -H "Content-Type: application/json" -d '{"query":"你的研究方向","max_results":10}'</code>
            </div>
            <div class="card">
                <h3>POST /papers/qa</h3>
                <p>跨论文RAG问答</p>
                <code>curl -X POST http://localhost:8000/papers/qa -H "Content-Type: application/json" -d '{"question":"你的问题","top_k":10}'</code>
            </div>
            <div class="card">
                <h3>GET /papers/status</h3>
                <p>论文库状态</p>
                <code>curl http://localhost:8000/papers/status</code>
            </div>
        </div>
        <script>
            fetch('/papers/status').then(r=>r.json()).then(d=>{
                document.getElementById('paper-count').textContent=d.total_papers;
                document.getElementById('indexed-count').textContent=d.indexed_papers;
            });

            async function doSearch(){
                const q=document.getElementById('query').value.trim();
                if(!q)return;
                const resultDiv=document.getElementById('result');
                const loadingDiv=document.getElementById('loading');
                resultDiv.style.display='none';
                resultDiv.textContent='';
                loadingDiv.style.display='block';
                try{
                    const resp=await fetch('/research/stream',{
                        method:'POST',
                        headers:{'Content-Type':'application/json'},
                        body:JSON.stringify({query:q,max_papers:5})
                    });
                    const reader=resp.body.getReader();
                    const decoder=new TextDecoder();
                    let buf='';
                    while(true){
                        const{done,value}=await reader.read();
                        if(done)break;
                        buf+=decoder.decode(value,{stream:true});
                        const lines=buf.split('\\n');
                        buf=lines.pop();
                        for(const line of lines){
                            if(line.startsWith('data: ')){
                                try{
                                    const d=JSON.parse(line.slice(6));
                                    if(d.step==='done'){
                                        resultDiv.textContent=d.result;
                                        resultDiv.style.display='block';
                                    }
                                }catch(e){}
                            }
                        }
                    }
                }catch(e){resultDiv.textContent='Error: '+e.message;resultDiv.style.display='block';}
                loadingDiv.style.display='none';
                fetch('/papers/status').then(r=>r.json()).then(d=>{
                    document.getElementById('paper-count').textContent=d.total_papers;
                    document.getElementById('indexed-count').textContent=d.indexed_papers;
                });
            }
        </script>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
