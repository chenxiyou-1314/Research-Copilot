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
            # Coordinator
            if result.get("search_queries"):
                yield f"data: {json.dumps({'step': 'coordinator', 'queries': result['search_queries']}, ensure_ascii=False)}\n\n"
            
            # Search
            if result.get("papers"):
                yield f"data: {json.dumps({'step': 'search', 'count': len(result['papers'])}, ensure_ascii=False)}\n\n"
            
            if result.get("filtered_papers"):
                titles = [p.get("title", "")[:50] for p in result["filtered_papers"][:5]]
                yield f"data: {json.dumps({'step': 'filter', 'count': len(result['filtered_papers']), 'top_papers': titles}, ensure_ascii=False)}\n\n"
            
            # Analysis
            if result.get("new_papers_count", 0) > 0:
                yield f"data: {json.dumps({'step': 'index', 'new': result['new_papers_count'], 'total': result.get('total_papers_count', 0)}, ensure_ascii=False)}\n\n"
            
            # Critic评分
            if result.get("critic_score"):
                yield f"data: {json.dumps({'step': 'critic', 'overall': result['critic_score'], 'coverage': result.get('critic_coverage', 0), 'accuracy': result.get('critic_accuracy', 0), 'coherence': result.get('critic_coherence', 0), 'passed': result.get('critic_passed', True), 'rerun_count': result.get('rerun_count', 0)}, ensure_ascii=False)}\n\n"
            
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


# ── 论文列表 ──
@app.get("/papers/list")
async def papers_list(keyword: str = "", limit: int = 50):
    """查看已索引论文列表，支持关键词搜索。"""
    if keyword:
        papers = paper_store.search_by_keyword(keyword)
    else:
        papers = paper_store.get_all_papers()
    # 只返回摘要前200字，节省带宽
    result = []
    for p in papers[:limit]:
        result.append({
            "title": p.get("title", ""),
            "authors": p.get("authors", [])[:3],
            "year": p.get("year"),
            "citations": p.get("citations", 0),
            "source": p.get("source", ""),
            "abstract": p.get("abstract", "")[:200] + "...",
            "url": p.get("url", ""),
            "is_indexed": p.get("is_indexed", False),
        })
    return {"total": len(papers), "showing": len(result), "papers": result}


# ── 健康检查 ──
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Copilot</title>
<style>
:root{--bg:#0a0f1a;--surface:#111827;--surface2:#1e293b;--border:#1e3a5f;--primary:#38bdf8;--primary2:#818cf8;--accent:#a78bfa;--success:#34d399;--warn:#fbbf24;--danger:#f87171;--text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;--radius:12px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}::-webkit-scrollbar-thumb:hover{background:var(--primary)}
.nav{position:fixed;top:0;left:0;right:0;z-index:100;background:rgba(10,15,26,0.85);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);padding:0 24px;height:60px;display:flex;align-items:center;justify-content:space-between}
.nav-brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:1.1em}
.nav-brand .icon{width:32px;height:32px;background:linear-gradient(135deg,var(--primary),var(--primary2));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px}
.nav-links{display:flex;gap:4px}
.nav-links button{background:none;border:none;color:var(--text2);padding:8px 16px;border-radius:8px;cursor:pointer;font-size:13px;font-weight:500;transition:all .2s}
.nav-links button:hover,.nav-links button.active{color:var(--primary);background:rgba(56,189,248,0.1)}
.main{max-width:1100px;margin:0 auto;padding:80px 20px 40px}
.tab-content{display:none}.tab-content.active{display:block}
/* Hero */
.hero{text-align:center;padding:40px 0 48px}
.hero h1{font-size:2.8em;font-weight:800;background:linear-gradient(135deg,var(--primary),var(--primary2),var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;letter-spacing:-0.02em}
.hero p{color:var(--text2);font-size:1.05em;max-width:600px;margin:0 auto;line-height:1.6}
/* Stats */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:32px}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;text-align:center;transition:border-color .2s,transform .2s}
.stat-card:hover{border-color:var(--primary);transform:translateY(-2px)}
.stat-num{font-size:2em;font-weight:800;background:linear-gradient(135deg,var(--primary),var(--primary2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.stat-label{font-size:12px;color:var(--text3);margin-top:4px;text-transform:uppercase;letter-spacing:.05em}
/* Cards */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:24px;margin-bottom:16px;transition:border-color .2s}
.card:hover{border-color:rgba(56,189,248,0.3)}
.card h3{color:var(--primary);font-size:1em;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.card h3 .badge{font-size:10px;background:rgba(56,189,248,0.15);color:var(--primary);padding:2px 8px;border-radius:4px;font-weight:600}
/* Search */
.search-section{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:28px;margin-bottom:24px}
.search-row{display:flex;gap:12px;margin-bottom:16px}
.search-input{flex:1;padding:14px 18px;border-radius:10px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:15px;outline:none;transition:border-color .2s}
.search-input:focus{border-color:var(--primary)}
.btn{padding:12px 24px;border-radius:10px;border:none;font-weight:600;cursor:pointer;font-size:14px;transition:all .2s;display:inline-flex;align-items:center;gap:6px}
.btn-primary{background:linear-gradient(135deg,var(--primary),var(--primary2));color:var(--bg)}
.btn-primary:hover{opacity:.9;transform:translateY(-1px)}
.btn-outline{background:transparent;border:1px solid var(--border);color:var(--text2)}
.btn-outline:hover{border-color:var(--primary);color:var(--primary)}
.btn-sm{padding:8px 14px;font-size:12px;border-radius:8px}
/* Progress */
.progress-bar{display:none;margin-top:16px}
.progress-steps{display:flex;gap:4px;margin-bottom:8px}
.progress-step{flex:1;height:4px;border-radius:2px;background:var(--border);transition:background .3s}
.progress-step.done{background:var(--primary)}
.progress-step.active{background:var(--primary);animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.progress-log{font-size:12px;color:var(--text3);max-height:80px;overflow-y:auto}
.progress-log div{padding:2px 0;border-bottom:1px solid rgba(30,58,95,0.3)}
.progress-log .step-coordinator{color:var(--accent)}.progress-log .step-search{color:var(--primary)}.progress-log .step-filter{color:var(--success)}.progress-log .step-index{color:var(--warn)}.progress-log .step-critic{color:#f472b6}.progress-log .step-done{color:var(--success);font-weight:600}
/* Result */
.result-box{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:24px;margin-top:16px;display:none;max-height:700px;overflow-y:auto;line-height:1.9;font-size:14px}
.result-box h1,.result-box h2,.result-box h3{color:var(--primary);margin:20px 0 8px}
.result-box h1{font-size:1.4em}.result-box h2{font-size:1.2em}.result-box h3{font-size:1.05em}
.result-box p{margin:8px 0}
.result-box strong{color:var(--text)}
.result-box table{width:100%;border-collapse:collapse;margin:12px 0}
.result-box th,.result-box td{border:1px solid var(--border);padding:8px 12px;text-align:left;font-size:13px}
.result-box th{background:var(--surface2);color:var(--primary);font-weight:600}
/* Critic */
.critic-box{display:none;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-top:12px}
.critic-scores{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px}
.critic-item{text-align:center}
.critic-item .score{font-size:1.8em;font-weight:800}
.critic-item .score.good{color:var(--success)}.critic-item .score.ok{color:var(--warn)}.critic-item .score.bad{color:var(--danger)}
.critic-item .label{font-size:11px;color:var(--text3);margin-top:2px}
.critic-feedback{font-size:13px;color:var(--text2);background:var(--bg);padding:12px;border-radius:8px;white-space:pre-wrap}
/* Papers */
.paper-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:10px;transition:border-color .2s}
.paper-card:hover{border-color:rgba(56,189,248,0.3)}
.paper-title{color:var(--primary);font-weight:600;font-size:14px;margin-bottom:6px;cursor:pointer}
.paper-title:hover{text-decoration:underline}
.paper-meta{font-size:12px;color:var(--text3);margin-bottom:6px}
.paper-meta span{margin-right:12px}
.paper-abstract{font-size:13px;color:var(--text2);line-height:1.6}
.paper-tag{display:inline-block;font-size:10px;padding:2px 8px;border-radius:4px;margin-right:4px}
.tag-arxiv{background:rgba(56,189,248,0.15);color:var(--primary)}.tag-s2{background:rgba(167,139,250,0.15);color:var(--accent)}.tag-indexed{background:rgba(52,211,153,0.15);color:var(--success)}
/* QA */
.qa-section{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:28px;margin-bottom:24px}
.qa-answer{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-top:16px;display:none;line-height:1.8;font-size:14px}
/* Architecture */
.arch-flow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:20px 0;justify-content:center}
.arch-node{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 18px;text-align:center;min-width:100px;transition:all .2s}
.arch-node:hover{border-color:var(--primary);transform:translateY(-2px)}
.arch-node .name{font-weight:700;font-size:13px;margin-bottom:2px}
.arch-node .desc{font-size:10px;color:var(--text3)}
.arch-node.coordinator{border-color:var(--accent)}.arch-node.coordinator .name{color:var(--accent)}
.arch-node.search{border-color:var(--primary)}.arch-node.search .name{color:var(--primary)}
.arch-node.analysis{border-color:var(--success)}.arch-node.analysis .name{color:var(--success)}
.arch-node.writing{border-color:var(--warn)}.arch-node.writing .name{color:var(--warn)}
.arch-node.critic{border-color:#f472b6}.arch-node.critic .name{color:#f472b6}
.arch-arrow{color:var(--text3);font-size:18px}
.arch-loop{display:flex;align-items:center;gap:6px;margin-top:12px;justify-content:center;font-size:12px;color:var(--text3)}
.arch-loop .loop-arrow{font-size:20px;color:#f472b6}
/* Empty state */
.empty-state{text-align:center;padding:60px 20px;color:var(--text3)}
.empty-state .icon{font-size:48px;margin-bottom:16px}
.empty-state p{font-size:14px}
/* Responsive */
@media(max-width:768px){.stats{grid-template-columns:repeat(2,1fr)}.hero h1{font-size:1.8em}.critic-scores{grid-template-columns:repeat(2,1fr)}.nav-links button span{display:none}}
</style>
</head>
<body>

<nav class="nav">
  <div class="nav-brand"><div class="icon">🔬</div> Research Copilot</div>
  <div class="nav-links">
    <button class="active" onclick="switchTab('generate')">综述生成</button>
    <button onclick="switchTab('papers')">论文库</button>
    <button onclick="switchTab('qa')">问答</button>
    <button onclick="switchTab('architecture')">架构</button>
  </div>
</nav>

<div class="main">

<!-- Tab: Generate -->
<div class="tab-content active" id="tab-generate">
  <div class="hero">
    <h1>Research Copilot</h1>
    <p>基于多Agent协作的科研文献智能检索与综述生成系统，支持意图识别、查询改写、RAG增强、自反思修订</p>
  </div>

  <div class="stats">
    <div class="stat-card"><div class="stat-num" id="s-papers">-</div><div class="stat-label">已索引论文</div></div>
    <div class="stat-card"><div class="stat-num" id="s-indexed">-</div><div class="stat-label">向量库</div></div>
    <div class="stat-card"><div class="stat-num" id="s-interests">-</div><div class="stat-label">研究方向</div></div>
    <div class="stat-card"><div class="stat-num" id="s-queries">-</div><div class="stat-label">历史查询</div></div>
  </div>

  <div class="search-section">
    <h3 style="color:var(--text);margin-bottom:16px">🎯 生成文献综述</h3>
    <div class="search-row">
      <input class="search-input" id="gen-query" placeholder="输入研究方向，如：class-incremental learning for fine-grained detection" />
      <select class="search-input" id="gen-count" style="flex:0 0 100px">
        <option value="5">5篇</option>
        <option value="10" selected>10篇</option>
        <option value="15">15篇</option>
      </select>
      <button class="btn btn-primary" onclick="doGenerate()">生成综述</button>
    </div>
    <div class="progress-bar" id="gen-progress">
      <div class="progress-steps" id="progress-steps"></div>
      <div class="progress-log" id="progress-log"></div>
    </div>
  </div>

  <div class="critic-box" id="critic-box">
    <h3 style="color:#f472b6;margin-bottom:12px">🔍 Critic 评估</h3>
    <div class="critic-scores" id="critic-scores"></div>
    <div class="critic-feedback" id="critic-feedback"></div>
  </div>

  <div class="result-box" id="gen-result"></div>
</div>

<!-- Tab: Papers -->
<div class="tab-content" id="tab-papers">
  <h2 style="margin-bottom:20px">📚 论文库</h2>
  <div class="search-row" style="margin-bottom:20px">
    <input class="search-input" id="paper-search" placeholder="搜索论文标题或摘要..." oninput="searchPapers()" />
    <button class="btn btn-outline btn-sm" onclick="loadPapers()">刷新</button>
  </div>
  <div id="papers-list"></div>
</div>

<!-- Tab: QA -->
<div class="tab-content" id="tab-qa">
  <h2 style="margin-bottom:20px">💬 跨论文问答</h2>
  <div class="qa-section">
    <p style="color:var(--text2);font-size:13px;margin-bottom:16px">基于已索引论文的RAG检索，回答科研问题。请确保论文库中已有相关论文。</p>
    <div class="search-row">
      <input class="search-input" id="qa-question" placeholder="提问，如：CIL方法在细粒度场景下的主要挑战是什么？" />
      <button class="btn btn-primary" onclick="doQA()">提问</button>
    </div>
    <div class="qa-answer" id="qa-answer"></div>
  </div>
</div>

<!-- Tab: Architecture -->
<div class="tab-content" id="tab-architecture">
  <h2 style="margin-bottom:8px">🏗️ 多Agent架构</h2>
  <p style="color:var(--text2);font-size:14px;margin-bottom:24px">基于LangGraph构建的多Agent协作系统，5个专业Agent各司其职，Critic实现自我反思与修订闭环。</p>

  <div class="arch-flow">
    <div class="arch-node coordinator"><div class="name">Coordinator</div><div class="desc">规划·调度</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node search"><div class="name">Search</div><div class="desc">检索·改写</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node analysis"><div class="name">Analysis</div><div class="desc">解析·RAG</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node writing"><div class="name">Writing</div><div class="desc">生成·修订</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node critic"><div class="name">Critic</div><div class="desc">评估·反馈</div></div>
  </div>
  <div class="arch-loop">
    <span class="loop-arrow">↩</span> 未达标时反馈回Coordinator，重新检索补充论文并修订综述（最多2轮）
  </div>

  <div style="margin-top:32px">
    <div class="card" style="margin-bottom:12px"><h3>Coordinator Agent <span class="badge">规划</span></h3><p style="color:var(--text2);font-size:13px;line-height:1.7">分析用户查询意图，生成多角度检索策略（查询改写），确定重点关注方向。Critic反馈不通过时，结合反馈调整检索策略重新规划。</p></div>
    <div class="card" style="margin-bottom:12px"><h3>Search Agent <span class="badge">检索</span></h3><p style="color:var(--text2);font-size:13px;line-height:1.7">执行查询改写→多源检索（arXiv + Semantic Scholar）→去重合并→引用数补全→按引用/年份筛选。支持Coordinator提供的多query并行检索。</p></div>
    <div class="card" style="margin-bottom:12px"><h3>Analysis Agent <span class="badge">分析</span></h3><p style="color:var(--text2);font-size:13px;line-height:1.7">PDF下载→文本提取→切分→Embedding→FAISS增量索引→RAG向量检索。增量索引机制避免全量重算，论文库从0到N索引耗时保持线性增长。</p></div>
    <div class="card" style="margin-bottom:12px"><h3>Writing Agent <span class="badge">生成</span></h3><p style="color:var(--text2);font-size:13px;line-height:1.7">RAG增强综述生成：基于检索上下文+论文元数据生成结构化文献综述（背景→方法→挑战→趋势）。支持根据Critic反馈修订模式。</p></div>
    <div class="card" style="margin-bottom:12px"><h3>Critic Agent <span class="badge">评估</span></h3><p style="color:var(--text2);font-size:13px;line-height:1.7">三维度评估综述质量：覆盖度（是否遗漏重要方法）、准确性（引用和方法描述是否正确）、连贯性（逻辑结构是否清晰）。低于阈值触发Coordinator重跑。</p></div>
  </div>

  <h3 style="margin:24px 0 12px;color:var(--text)">技术栈</h3>
  <div style="display:flex;flex-wrap:wrap;gap:8px">
    <span style="background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:8px;font-size:12px">LangChain</span>
    <span style="background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:8px;font-size:12px">LangGraph</span>
    <span style="background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:8px;font-size:12px">DeepSeek V4</span>
    <span style="background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:8px;font-size:12px">BGE-M3</span>
    <span style="background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:8px;font-size:12px">FAISS</span>
    <span style="background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:8px;font-size:12px">FastAPI</span>
    <span style="background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:8px;font-size:12px">SSE</span>
    <span style="background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:8px;font-size:12px">Docker</span>
    <span style="background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:8px;font-size:12px">Redis</span>
    <span style="background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:8px;font-size:12px">arXiv API</span>
    <span style="background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:8px;font-size:12px">Semantic Scholar</span>
  </div>
</div>

</div>

<script>
const stepNames={start:'🚀 启动',coordinator:'🧭 Coordinator规划',search:'🔍 论文检索',filter:'📋 论文筛选',index:'💾 索引构建',critic:'🔍 Critic评估',done:'✅ 完成'};
let completedSteps=[];

function switchTab(t){document.querySelectorAll('.tab-content').forEach(e=>e.classList.remove('active'));document.querySelectorAll('.nav-links button').forEach(e=>e.classList.remove('active'));document.getElementById('tab-'+t).classList.add('active');event.target.classList.add('active');if(t==='papers')loadPapers();if(t==='generate')refreshStats();}

function refreshStats(){fetch('/papers/status').then(r=>r.json()).then(d=>{document.getElementById('s-papers').textContent=d.total_papers;document.getElementById('s-indexed').textContent=d.indexed_papers;document.getElementById('s-interests').textContent=d.user_interests.length;document.getElementById('s-queries').textContent=d.user_interests.length;}).catch(()=>{});}
refreshStats();

function updateProgress(step,data){
  if(!stepNames[step])return;
  completedSteps.push(step);
  const stepsEl=document.getElementById('progress-steps');
  const logEl=document.getElementById('progress-log');
  stepsEl.innerHTML=Object.keys(stepNames).map(s=>'<div class="progress-step '+(completedSteps.includes(s)?'done':'')+' '+(s===step?'active':'')+'"></div>').join('');
  let msg=stepNames[step]||step;
  if(step==='coordinator'&&data.queries)msg+=' — 改写查询: '+data.queries.join(' | ');
  if(step==='search'&&data.count)msg+=' — 检索到 '+data.count+' 篇';
  if(step==='filter'&&data.count)msg+=' — 筛选后 '+data.count+' 篇';
  if(step==='index'&&data.new)msg+=' — 新索引 '+data.new+' 篇';
  if(step==='critic'&&data.overall)msg+=' — 总分 '+data.overall+'/5 ('+(data.passed?'通过':'未通过，将重跑')+')';
  logEl.innerHTML+='<div class="step-'+step+'">'+msg+'</div>';
  logEl.scrollTop=logEl.scrollHeight;
}

async function doGenerate(){
  const q=document.getElementById('gen-query').value.trim();if(!q)return;
  const n=parseInt(document.getElementById('gen-count').value);
  const resultEl=document.getElementById('gen-result');const criticEl=document.getElementById('critic-box');const progressEl=document.getElementById('gen-progress');
  resultEl.style.display='none';criticEl.style.display='none';progressEl.style.display='block';
  completedSteps=[];
  document.getElementById('progress-steps').innerHTML='';
  document.getElementById('progress-log').innerHTML='';
  try{
    const resp=await fetch('/research/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,max_papers:n})});
    const reader=resp.body.getReader();const decoder=new TextDecoder();let buf='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\\n');buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        try{
          const d=JSON.parse(line.slice(6));
          updateProgress(d.step,d);
          if(d.step==='critic'){
            criticEl.style.display='block';
            const scoreClass=v=>v>=4?'good':v>=3?'ok':'bad';
            document.getElementById('critic-scores').innerHTML=
              '<div class="critic-item"><div class="score '+scoreClass(d.coverage)+'">'+d.coverage+'</div><div class="label">覆盖度</div></div>'+
              '<div class="critic-item"><div class="score '+scoreClass(d.accuracy)+'">'+d.accuracy+'</div><div class="label">准确性</div></div>'+
              '<div class="critic-item"><div class="score '+scoreClass(d.coherence)+'">'+d.coherence+'</div><div class="label">连贯性</div></div>'+
              '<div class="critic-item"><div class="score '+scoreClass(d.overall)+'">'+d.overall+'</div><div class="label">总分</div></div>';
          }
          if(d.step==='done'){
            resultEl.innerHTML=renderMarkdown(d.result);resultEl.style.display='block';
          }
        }catch(e){}
      }
    }
  }catch(e){resultEl.textContent='Error: '+e.message;resultEl.style.display='block';}
  progressEl.querySelector('.progress-steps').querySelectorAll('.progress-step').forEach(s=>{s.classList.remove('active');s.classList.add('done');});
  refreshStats();
}

function renderMarkdown(text){
  if(!text)return'';
  let h=text.replace(/^### (.+)$/gm,'<h3>$1</h3>').replace(/^## (.+)$/gm,'<h2>$1</h2>').replace(/^# (.+)$/gm,'<h1>$1</h1>');
  h=h.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>');
  h=h.replace(/\\[(.+?)\\]\\((.+?)\\)/g,'<a href="$2" target="_blank" style="color:var(--primary)">$1</a>');
  h=h.replace(/\\n/g,'<br>');
  return h;
}

async function loadPapers(){
  const kw=document.getElementById('paper-search').value.trim();
  const url='/papers/list'+(kw?'?keyword='+encodeURIComponent(kw):'');
  const resp=await fetch(url);const data=await resp.json();
  const el=document.getElementById('papers-list');
  if(!data.papers||data.papers.length===0){el.innerHTML='<div class="empty-state"><div class="icon">📭</div><p>论文库为空，先去生成综述索引一些论文吧</p></div>';return;}
  el.innerHTML=data.papers.map(p=>'<div class="paper-card"><div class="paper-title" onclick="window.open(\\''+p.url+'\\',\\'_blank\\')">'+p.title+'</div><div class="paper-meta"><span>👥 '+(p.authors||[]).join(', ')+'</span><span>📅 '+(p.year||'-')+'</span><span>📊 '+(p.citations||0)+' 引用</span></div><div class="paper-abstract">'+p.abstract+'</div><div style="margin-top:8px"><span class="paper-tag '+(p.source==='arxiv'?'tag-arxiv':'tag-s2')+'">'+p.source+'</span>'+(p.is_indexed?'<span class="paper-tag tag-indexed">已索引</span>':'')+'</div></div>').join('');
}

async function doQA(){
  const q=document.getElementById('qa-question').value.trim();if(!q)return;
  const el=document.getElementById('qa-answer');el.style.display='block';el.innerHTML='<span style="color:var(--text3)">⏳ 正在检索论文库并生成回答...</span>';
  try{
    const resp=await fetch('/papers/qa',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,top_k:10})});
    const data=await resp.json();
    el.innerHTML=renderMarkdown(data.answer);
  }catch(e){el.textContent='Error: '+e.message;}
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
