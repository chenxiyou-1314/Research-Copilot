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


class DecomposeRequest(BaseModel):
    query: str


class TrendRequest(BaseModel):
    query: str


class ProfileRequest(BaseModel):
    query: str = ""


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
            
            # Novelty分析
            if result.get("verified_ideas"):
                novelty_data = {
                    "step": "novelty",
                    "gaps": result.get("gaps", {}),
                    "transfers_count": len(result.get("transfers", [])),
                    "ideas_count": len(result.get("ideas", [])),
                    "verified": [],
                }
                for vi in result["verified_ideas"]:
                    novelty_data["verified"].append({
                        "title": vi.get("idea_title", ""),
                        "is_novel": vi.get("is_novel", True),
                        "confidence": vi.get("confidence", 0),
                        "novelty_statement": vi.get("novelty_statement", ""),
                        "similar_works": vi.get("similar_works", []),
                    })
                yield f"data: {json.dumps(novelty_data, ensure_ascii=False)}\n\n"
            
            # Method Decomposition
            if result.get("decomposition"):
                decomp_data = {
                    "step": "decomposition",
                    "papers_decomposed": len(result["decomposition"]),
                    "decomposition": result["decomposition"],
                    "recombinations_count": len(result.get("recombinations", [])),
                    "validated": [],
                }
                for vr in result.get("validated_recombinations", []):
                    decomp_data["validated"].append({
                        "name": vr.get("name", ""),
                        "compatibility_score": vr.get("compatibility_score", 3),
                        "implementation_difficulty": vr.get("implementation_difficulty", ""),
                        "risk_factors": vr.get("risk_factors", []),
                        "mitigation": vr.get("mitigation", ""),
                        "overall_feasibility": vr.get("overall_feasibility", ""),
                        "quick_start": vr.get("quick_start", ""),
                        "potential_pitfall": vr.get("potential_pitfall", ""),
                        "source_papers": vr.get("original_recombination", {}).get("source_papers", []),
                        "components": vr.get("original_recombination", {}).get("components", {}),
                        "motivation": vr.get("original_recombination", {}).get("motivation", ""),
                        "expected_synergy": vr.get("original_recombination", {}).get("expected_synergy", ""),
                    })
                yield f"data: {json.dumps(decomp_data, ensure_ascii=False)}\n\n"
            
            # Trend Forecasting
            if result.get("trend_forecast"):
                trend_data = {
                    "step": "trend",
                    "trend_stats": result.get("trend_stats", {}),
                    "timeline": result.get("timeline", {}),
                    "evolution": result.get("evolution", {}),
                    "forecast": result["trend_forecast"],
                }
                yield f"data: {json.dumps(trend_data, ensure_ascii=False)}\n\n"
            
            # Research Profile Graph
            if result.get("profile_graph"):
                profile_data = {
                    "step": "profile",
                    "profile_graph": result["profile_graph"],
                }
                yield f"data: {json.dumps(profile_data, ensure_ascii=False)}\n\n"
            
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


# ── 方法解构与重组 ──
@app.post("/decomposition")
async def method_decomposition(req: DecomposeRequest):
    """独立调用方法解构与重组（需要论文库中已有论文）。"""
    from agent.graph import _get_llm, llm
    from agent.decomposition_agent import run_decomposition
    
    papers = paper_store.get_all_papers()
    if not papers:
        return {"error": "论文库为空，请先生成综述索引论文", "decomposition": [], "recombinations": [], "validated_recombinations": []}
    
    # 构建论文上下文
    papers_context = ""
    for p in papers[:8]:
        authors_str = ', '.join(p.get('authors', [])[:3])
        papers_context += f"\n[{authors_str}, {p.get('year')}] {p.get('title')}\n  摘要: {p.get('abstract', '')[:300]}\n"
    
    result = run_decomposition(llm, papers_context, {}, req.query)
    
    return {
        "decomposition": result["decomposition"],
        "recombinations": result["recombinations"],
        "validated_recombinations": result["validated_recombinations"],
    }


# ── 趋势预测 ──
@app.post("/trend")
async def trend_forecast(req: TrendRequest):
    """独立调用趋势预测（会自动做大规模统计检索，无需依赖本地论文库）。"""
    from agent.graph import _get_llm, llm
    from agent.trend_agent import run_trend_forecast
    
    # 趋势预测现在自行做大规模检索，本地论文仅作补充
    papers = paper_store.get_all_papers() or []
    result = run_trend_forecast(llm, papers, [], {}, req.query)
    
    return {
        "trend_stats": result.get("trend_stats", {}),
        "timeline": result["timeline"],
        "evolution": result["evolution"],
        "trend_forecast": result["trend_forecast"],
    }


# ── 研究知识图谱 ──
@app.post("/profile")
async def research_profile(req: ProfileRequest):
    """独立获取研究知识图谱。"""
    from agent.graph import _get_llm, llm
    from agent.profile_agent import build_profile_graph
    
    papers = paper_store.get_all_papers()
    query_history = user_profile._profile.get("query_history", [])
    
    if not papers and not query_history:
        return {"error": "暂无历史数据，请先生成综述或检索论文", "profile_graph": {}}
    
    result = build_profile_graph(llm, query_history, papers)
    
    return {"profile_graph": result}


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
:root{--bg:#0a0f1a;--surface:#111827;--surface2:#1e293b;--border:#1e3a5f;--primary:#38bdf8;--primary2:#818cf8;--accent:#a78bfa;--success:#34d399;--warn:#fbbf24;--danger:#f87171;--text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;--radius:12px;--nav-bg:rgba(10,15,26,0.85);--canvas-grid:rgba(56,189,248,0.1);--canvas-axis:rgba(56,189,248,0.15);--canvas-fill:rgba(56,189,248,0.15);--canvas-stroke:#38bdf8;--canvas-point-stroke:#0a0f1a}
[data-theme="light"]{--bg:#f8fafc;--surface:#ffffff;--surface2:#f1f5f9;--border:#cbd5e1;--primary:#0284c7;--primary2:#6366f1;--accent:#7c3aed;--success:#059669;--warn:#d97706;--danger:#dc2626;--text:#0f172a;--text2:#475569;--text3:#94a3b8;--nav-bg:rgba(248,250,252,0.85);--canvas-grid:rgba(2,132,199,0.08);--canvas-axis:rgba(2,132,199,0.12);--canvas-fill:rgba(2,132,199,0.12);--canvas-stroke:#0284c7;--canvas-point-stroke:#ffffff}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;transition:background .3s,color .3s}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}::-webkit-scrollbar-thumb:hover{background:var(--primary)}
.nav{position:fixed;top:0;left:0;right:0;z-index:100;background:var(--nav-bg);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);padding:0 24px;height:60px;display:flex;align-items:center;justify-content:space-between}
.nav-brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:1.1em}
.nav-brand .icon{width:32px;height:32px;background:linear-gradient(135deg,var(--primary),var(--primary2));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px}
.nav-right{display:flex;align-items:center;gap:4px}
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
.progress-log .step-coordinator{color:var(--accent)}.progress-log .step-search{color:var(--primary)}.progress-log .step-filter{color:var(--success)}.progress-log .step-index{color:var(--warn)}.progress-log .step-critic{color:#f472b6}.progress-log .step-novelty{color:var(--accent)}.progress-log .step-decomposition{color:var(--success)}.progress-log .step-trend{color:var(--warn)}.progress-log .step-profile{color:var(--primary2)}.progress-log .step-done{color:var(--success);font-weight:600}
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
/* Trend standalone */
.theme-toggle{background:none;border:1px solid var(--border);color:var(--text2);width:36px;height:36px;border-radius:10px;cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;transition:all .2s;margin-left:8px}
.theme-toggle:hover{border-color:var(--primary);color:var(--primary)}
.trend-phase-badge{display:inline-flex;align-items:center;gap:8px;padding:10px 24px;border-radius:12px;font-size:1.3em;font-weight:800;border:2px solid}
.trend-scores-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
.trend-score-card{background:var(--bg);padding:14px;border-radius:10px;text-align:center}
.trend-score-card .num{font-size:1.8em;font-weight:800}
.trend-score-card .label{font-size:11px;color:var(--text3);margin-top:4px}
.trend-score-bar{height:6px;background:var(--border);border-radius:3px;margin-top:6px}
.trend-score-bar div{height:100%;border-radius:3px;transition:width .5s}
/* Profile visualization */
.profile-canvas-wrap{display:flex;justify-content:center;padding:20px 0}
.profile-canvas-wrap canvas{border-radius:12px}
.mastery-row{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid rgba(30,58,95,0.3)}
.mastery-row:last-child{border-bottom:none}
.mastery-label{font-size:13px;color:var(--text);min-width:120px;font-weight:600}
.mastery-bar-track{flex:1;height:8px;background:var(--border);border-radius:4px;overflow:hidden}
.mastery-bar-fill{height:100%;border-radius:4px;transition:width .6s ease}
.mastery-depth{font-size:11px;padding:2px 8px;border-radius:4px;min-width:80px;text-align:center}
.mastery-domain-tag{font-size:10px;color:var(--text3);min-width:60px}
.legend-row{display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin-top:12px}
.legend-item{font-size:11px;color:var(--text3);display:flex;align-items:center;gap:4px}
.legend-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
/* Responsive */
@media(max-width:768px){.stats{grid-template-columns:repeat(2,1fr)}.hero h1{font-size:1.8em}.critic-scores{grid-template-columns:repeat(2,1fr)}.trend-scores-grid{grid-template-columns:repeat(2,1fr)}.nav-links button span{display:none}}
</style>
</head>
<body>

<nav class="nav">
  <div class="nav-brand"><div class="icon">🔬</div> Research Copilot</div>
  <div class="nav-right">
    <div class="nav-links">
      <button class="active" onclick="switchTab('generate')">综述生成</button>
      <button onclick="switchTab('papers')">论文库</button>
      <button onclick="switchTab('qa')">问答</button>
      <button onclick="switchTab('trend')">趋势预测</button>
      <button onclick="switchTab('profile')">知识图谱</button>
      <button onclick="switchTab('architecture')">架构</button>
    </div>
    <button class="theme-toggle" id="theme-btn" onclick="toggleTheme()" title="切换亮色/暗色主题">🌙</button>
  </div>
</nav>

<div class="main">

<!-- Tab: Generate -->
<div class="tab-content active" id="tab-generate">
  <div class="hero">
    <h1>Research Copilot</h1>
    <p>基于多Agent协作的科研文献智能检索与综述生成系统，覆盖"读论文→造方法→判方向→懂自己"的完整科研决策链路</p>
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

  <div class="card" id="novelty-box" style="display:none;margin-top:12px;border-color:var(--accent)">
    <h3 style="color:var(--accent);margin-bottom:16px">💡 研究思路发现 <span class="badge">Novelty Agent</span></h3>
    <div id="novelty-gaps" style="margin-bottom:16px"></div>
    <div id="novelty-transfers" style="margin-bottom:16px"></div>
    <div id="novelty-ideas"></div>
  </div>

  <div class="card" id="decomp-box" style="display:none;margin-top:12px;border-color:var(--success)">
    <h3 style="color:var(--success);margin-bottom:16px">🧬 方法解构与重组 <span class="badge">Decomposition Agent</span></h3>
    <div id="decomp-matrix" style="margin-bottom:16px"></div>
    <div id="decomp-recombinations"></div>
  </div>

  <div class="card" id="trend-box" style="display:none;margin-top:12px;border-color:var(--warn)">
    <h3 style="color:var(--warn);margin-bottom:16px">📈 趋势预测 <span class="badge">Trend Agent</span></h3>
    <div id="trend-timeline" style="margin-bottom:16px"></div>
    <div id="trend-evolution" style="margin-bottom:16px"></div>
    <div id="trend-forecast"></div>
  </div>

  <div class="card" id="profile-box" style="display:none;margin-top:12px;border-color:var(--primary2)">
    <h3 style="color:var(--primary2);margin-bottom:16px">🗺️ 研究知识图谱 <span class="badge">Profile Agent</span></h3>
    <div id="profile-domains" style="margin-bottom:16px"></div>
    <div id="profile-methods" style="margin-bottom:16px"></div>
    <div id="profile-blindspots" style="margin-bottom:16px"></div>
    <div id="profile-unread" style="margin-bottom:16px"></div>
    <div id="profile-style"></div>
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

<!-- Tab: Trend -->
<div class="tab-content" id="tab-trend">
  <h2 style="margin-bottom:8px">📈 趋势预测</h2>
  <p style="color:var(--text2);font-size:14px;margin-bottom:20px">输入研究方向，分析热度/饱和度/潜力/门槛四维评分，判断方向值不值得做。需要论文库中已有论文。</p>

  <div class="search-section">
    <div class="search-row">
      <input class="search-input" id="trend-query" placeholder="输入研究方向，如：vision-language model adaptation" />
      <button class="btn btn-primary" onclick="doTrendStandalone()">分析趋势</button>
    </div>
  </div>

  <div id="trend-standalone-result" style="display:none">
    <div id="trend-sa-phase" style="margin-bottom:16px"></div>
    <div id="trend-sa-scores" class="trend-scores-grid"></div>
    <div id="trend-sa-reasoning" style="margin-bottom:16px"></div>
    <div id="trend-sa-timeline" style="margin-bottom:16px"></div>
    <div id="trend-sa-evolution" style="margin-bottom:16px"></div>
    <div id="trend-sa-forecast" style="margin-bottom:16px"></div>
    <div id="trend-sa-advice" style="margin-bottom:16px"></div>
    <div id="trend-sa-flags"></div>
  </div>

  <div id="trend-standalone-empty" class="empty-state">
    <div class="icon">📊</div>
    <p>输入研究方向，获取四维趋势评分与投入建议</p>
  </div>
</div>

<!-- Tab: Profile -->
<div class="tab-content" id="tab-profile">
  <h2 style="margin-bottom:8px">🗺️ 研究知识图谱</h2>
  <p style="color:var(--text2);font-size:14px;margin-bottom:20px">基于历史查询和已索引论文，构建你的个人知识图谱——核心领域、已掌握方法、知识盲区、研究风格画像。</p>

  <div style="text-align:center;margin-bottom:20px">
    <button class="btn btn-primary" onclick="doProfileStandalone()">🔄 生成知识图谱</button>
  </div>

  <div id="profile-standalone-result" style="display:none">
    <!-- Radar Chart -->
    <div class="card" style="margin-bottom:16px">
      <h3>🎯 领域掌握度雷达图</h3>
      <div class="profile-canvas-wrap"><canvas id="profile-radar" width="420" height="420"></canvas></div>
      <div class="legend-row" id="radar-legend"></div>
    </div>

    <!-- Mastery Bars -->
    <div class="card" style="margin-bottom:16px">
      <h3>🔧 方法掌握度</h3>
      <div id="profile-sa-mastery"></div>
    </div>

    <!-- Blindspots -->
    <div class="card" style="margin-bottom:16px">
      <h3>🕳️ 知识盲区 <span class="badge">需要补强</span></h3>
      <div id="profile-sa-blindspots"></div>
    </div>

    <!-- Unread -->
    <div class="card" style="margin-bottom:16px">
      <h3>📚 高相关但未覆盖 <span class="badge">建议探索</span></h3>
      <div id="profile-sa-unread"></div>
    </div>

    <!-- Research Style -->
    <div class="card" style="margin-bottom:16px">
      <h3>👤 研究风格画像</h3>
      <div id="profile-sa-style"></div>
    </div>
  </div>

  <div id="profile-standalone-empty" class="empty-state">
    <div class="icon">🗺️</div>
    <p>点击上方按钮，生成你的研究知识图谱</p>
  </div>
</div>

<!-- Tab: Architecture -->
<div class="tab-content" id="tab-architecture">
  <h2 style="margin-bottom:8px">🏗️ 多Agent架构</h2>
  <p style="color:var(--text2);font-size:14px;margin-bottom:24px">基于LangGraph构建的多Agent协作系统，9个专业Agent各司其职：Critic实现自我反思与修订闭环，Novelty发现研究空白，Decomposition重组方法，Trend预测趋势，Profile构建知识图谱——覆盖"读论文→造方法→判方向→懂自己"的完整科研决策链路。</p>

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
    <div class="arch-arrow">→</div>
    <div class="arch-node" style="border-color:var(--accent)"><div class="name" style="color:var(--accent)">Novelty</div><div class="desc">思路·发现</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node" style="border-color:var(--success)"><div class="name" style="color:var(--success)">Decomposition</div><div class="desc">解构·重组</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node" style="border-color:var(--warn)"><div class="name" style="color:var(--warn)">Trend</div><div class="desc">趋势·预测</div></div>
    <div class="arch-arrow">→</div>
    <div class="arch-node" style="border-color:var(--primary2)"><div class="name" style="color:var(--primary2)">Profile</div><div class="desc">图谱·画像</div></div>
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
    <div class="card" style="margin-bottom:12px"><h3>Novelty Agent <span class="badge">发现</span></h3><p style="color:var(--text2);font-size:13px;line-height:1.7">研究思路发现引擎：Gap分析（方法学/数据/理论/实践四维度空白提取）→跨域迁移（从其他领域寻找可迁移思路）→思路生成（2-3个具体可执行研究方向含技术路线）→新颖性验证（检索已有论文确认无人做过）。</p></div>
    <div class="card" style="margin-bottom:12px"><h3>Decomposition Agent <span class="badge">解构</span></h3><p style="color:var(--text2);font-size:13px;line-height:1.7">方法解构与重组引擎：将每篇论文的方法拆解为5个原子组件（backbone/training_strategy/loss_function/data_augmentation/evaluation_protocol）→构建跨论文组件矩阵→跨论文方法重组（从不同论文中选取组件组合新方案）→可行性验证（兼容性评分/实施难度/风险评估/快速验证方案）。从"读论文"到"造方法"的跃迁。</p></div>
    <div class="card" style="margin-bottom:12px"><h3>Trend Agent <span class="badge">预测</span></h3><p style="color:var(--text2);font-size:13px;line-height:1.7">研究趋势预测引擎：时间线分析（按年份统计主题分布，识别新兴/衰退/稳定趋势）→方法演化追踪（追踪技术路线演变路径，识别范式转换点）→趋势预测（热度/饱和度/潜力/门槛四维评分 + 短中长期预测 + 投入建议 + 红绿旗信号）。从"这个方向有什么"到"这个方向值不值得做"的决策升级。</p></div>
    <div class="card" style="margin-bottom:12px"><h3>Profile Agent <span class="badge">画像</span></h3><p style="color:var(--text2);font-size:13px;line-height:1.7">研究知识图谱构建引擎：基于用户历史查询和已索引论文，构建个人知识图谱——标注核心研究领域与掌握程度、已掌握方法及深度、知识盲区与建议搜索、高相关但未覆盖方向、研究风格画像（理论/工程导向、深入/广泛、前沿/经典）。每次检索后动态更新，越用越懂你。</p></div>
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
const stepNames={start:'🚀 启动',coordinator:'🧭 Coordinator规划',search:'🔍 论文检索',filter:'📋 论文筛选',index:'💾 索引构建',critic:'🔍 Critic评估',novelty:'💡 新思路发现',decomposition:'🧬 方法解构与重组',trend:'📈 趋势预测',profile:'🗺️ 知识图谱',done:'✅ 完成'};
let completedSteps=[];

// ── Theme ──
function getTheme(){return localStorage.getItem('rc-theme')||'dark';}
function applyTheme(t){
  document.documentElement.setAttribute('data-theme',t);
  const btn=document.getElementById('theme-btn');
  if(btn)btn.textContent=t==='dark'?'🌙':'☀️';
  localStorage.setItem('rc-theme',t);
}
function toggleTheme(){applyTheme(getTheme()==='dark'?'light':'dark');}
applyTheme(getTheme());

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
  if(step==='novelty'&&data.ideas_count)msg+=' — 发现 '+data.ideas_count+' 个新思路';
  if(step==='decomposition'&&data.papers_decomposed)msg+=' — 解构 '+data.papers_decomposed+' 篇论文，生成 '+data.recombinations_count+' 个重组方案';
  if(step==='trend'&&data.forecast)msg+=' — 方向阶段: '+(data.forecast.overall_phase||'分析中');
  if(step==='profile'&&data.profile_graph)msg+=' — 构建知识图谱';
  logEl.innerHTML+='<div class="step-'+step+'">'+msg+'</div>';
  logEl.scrollTop=logEl.scrollHeight;
}

async function doGenerate(){
  const q=document.getElementById('gen-query').value.trim();if(!q)return;
  const n=parseInt(document.getElementById('gen-count').value);
  const resultEl=document.getElementById('gen-result');const criticEl=document.getElementById('critic-box');const progressEl=document.getElementById('gen-progress');
  resultEl.style.display='none';criticEl.style.display='none';progressEl.style.display='block';
  document.getElementById('novelty-box').style.display='none';
  document.getElementById('decomp-box').style.display='none';
  document.getElementById('trend-box').style.display='none';
  document.getElementById('profile-box').style.display='none';
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
          if(d.step==='novelty'){
            const nb=document.getElementById('novelty-box');nb.style.display='block';
            // Gaps
            const gaps=d.gaps||{};
            let gapsHtml='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">📊 Gap分析</h4><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">';
            const gapLabels={methodological_gaps:'方法学空白',data_gaps:'数据空白',theoretical_gaps:'理论空白',practical_gaps:'实践空白'};
            for(const[k,v] of Object.entries(gapLabels)){
              const items=(gaps[k]||[]).slice(0,2);
              gapsHtml+='<div style="background:var(--bg);padding:10px;border-radius:8px"><div style="font-size:11px;color:var(--text3);margin-bottom:4px">'+v+'</div>'+items.map(i=>'<div style="font-size:12px;color:var(--text2);margin:3px 0;padding-left:8px;border-left:2px solid var(--accent)">'+i+'</div>').join('')+'</div>';
            }
            gapsHtml+='</div>';
            document.getElementById('novelty-gaps').innerHTML=gapsHtml;
            // Verified ideas
            const verified=d.verified||[];
            let ideasHtml='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">🎯 研究思路（已验证新颖性）</h4>';
            verified.forEach((v,i)=>{
              const novColor=v.is_novel?'var(--success)':'var(--warn)';
              const novText=v.is_novel?'✅ 新颖':'⚠️ 存在类似工作';
              ideasHtml+='<div style="background:var(--bg);padding:16px;border-radius:8px;margin-bottom:10px;border-left:3px solid '+novColor+'">';
              ideasHtml+='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px"><strong style="font-size:14px;color:var(--text)">'+(i+1)+'. '+v.title+'</strong><span style="font-size:11px;padding:2px 8px;border-radius:4px;background:'+(v.is_novel?'rgba(52,211,153,0.15)':'rgba(251,191,36,0.15)')+';color:'+novColor+'">'+novText+' (置信度:'+(v.confidence*100).toFixed(0)+'%)</span></div>';
              if(v.novelty_statement)ideasHtml+='<div style="font-size:12px;color:var(--accent);margin-bottom:6px">💡 '+v.novelty_statement+'</div>';
              if(v.similar_works&&v.similar_works.length>0)ideasHtml+='<div style="font-size:11px;color:var(--text3)">相关已有工作: '+v.similar_works.join(' | ')+'</div>';
              ideasHtml+='</div>';
            });
            document.getElementById('novelty-ideas').innerHTML=ideasHtml;
          }
          if(d.step==='decomposition'){
            const db=document.getElementById('decomp-box');db.style.display='block';
            // Component matrix
            const papers=d.decomposition||[];
            const compTypes=['backbone','training_strategy','loss_function','data_augmentation','evaluation_protocol'];
            const compLabels={backbone:'🏗️ Backbone',training_strategy:'🎯 训练策略',loss_function:'📊 损失函数',data_augmentation:'🎨 数据增强',evaluation_protocol:'📏 评估协议'};
            let matrixHtml='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">📊 方法组件矩阵</h4><div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:11px;min-width:600px">';
            matrixHtml+='<tr><th style="padding:8px;border:1px solid var(--border);background:var(--surface2);color:var(--primary);text-align:left;min-width:80px">组件</th>';
            papers.slice(0,6).forEach(p=>{matrixHtml+='<th style="padding:8px;border:1px solid var(--border);background:var(--surface2);color:var(--text);text-align:left;min-width:120px">'+(p.paper_title||'?')+'<br><span style="color:var(--text3);font-size:10px">'+(p.paper_year||'')+'</span></th>';});
            matrixHtml+='</tr>';
            compTypes.forEach(ct=>{
              matrixHtml+='<tr><td style="padding:6px 8px;border:1px solid var(--border);color:var(--accent);font-weight:600;white-space:nowrap">'+compLabels[ct]+'</td>';
              papers.slice(0,6).forEach(p=>{
                const comp=(p.components||{})[ct]||{};
                const name=typeof comp==='object'?comp.name||'N/A':comp;
                matrixHtml+='<td style="padding:6px 8px;border:1px solid var(--border);color:var(--text2)">'+name+'</td>';
              });
              matrixHtml+='</tr>';
            });
            matrixHtml+='</table></div>';
            document.getElementById('decomp-matrix').innerHTML=matrixHtml;
            // Validated recombinations
            const validated=d.validated||[];
            let recombHtml='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">🔬 方法重组方案（已验证可行性）</h4>';
            validated.forEach((v,i)=>{
              const feasColor=v.overall_feasibility==='高'?'var(--success)':v.overall_feasibility==='中'?'var(--warn)':'var(--danger)';
              const compatStars='★'.repeat(v.compatibility_score)+'☆'.repeat(5-v.compatibility_score);
              recombHtml+='<div style="background:var(--bg);padding:16px;border-radius:8px;margin-bottom:10px;border-left:3px solid '+feasColor+'">';
              recombHtml+='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px"><strong style="font-size:14px;color:var(--text)">'+(i+1)+'. '+v.name+'</strong><div style="display:flex;gap:6px"><span style="font-size:11px;padding:2px 8px;border-radius:4px;background:rgba(52,211,153,0.15);color:'+feasColor+'">可行性: '+v.overall_feasibility+'</span><span style="font-size:11px;padding:2px 8px;border-radius:4px;background:rgba(56,189,248,0.15);color:var(--primary)">兼容性: '+compatStars+'</span></div></div>';
              if(v.motivation)recombHtml+='<div style="font-size:12px;color:var(--text2);margin-bottom:8px">📌 '+v.motivation+'</div>';
              // Component pills
              if(v.components&&Object.keys(v.components).length>0){
                recombHtml+='<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px">';
                const cLabels={backbone:'🏗',training_strategy:'🎯',loss_function:'📊',data_augmentation:'🎨',evaluation_protocol:'📏'};
                for(const[ck,cv] of Object.entries(v.components)){recombHtml+='<span style="font-size:10px;padding:3px 8px;border-radius:4px;background:rgba(167,139,250,0.1);color:var(--accent);border:1px solid rgba(167,139,250,0.2)">'+(cLabels[ck]||'')+' '+cv+'</span>';}
                recombHtml+='</div>';
              }
              if(v.expected_synergy)recombHtml+='<div style="font-size:11px;color:var(--primary);margin-bottom:6px">⚡ 协同效应: '+v.expected_synergy+'</div>';
              if(v.quick_start)recombHtml+='<div style="font-size:11px;color:var(--success);margin-bottom:6px;padding:8px;background:rgba(52,211,153,0.05);border-radius:4px">🚀 快速验证: '+v.quick_start+'</div>';
              if(v.risk_factors&&v.risk_factors.length>0)recombHtml+='<div style="font-size:11px;color:var(--warn)">⚠️ 风险: '+v.risk_factors.join(' | ')+'</div>';
              if(v.potential_pitfall)recombHtml+='<div style="font-size:11px;color:var(--danger);margin-top:4px">🪤 最可能失败: '+v.potential_pitfall+'</div>';
              recombHtml+='</div>';
            });
            document.getElementById('decomp-recombinations').innerHTML=recombHtml;
          }
          if(d.step==='trend'){
            const tb=document.getElementById('trend-box');tb.style.display='block';
            const fc=d.forecast||{};
            const tl=d.timeline||{};
            const ev=d.evolution||{};
            // Phase badge
            const phase=fc.overall_phase||'分析中';
            const phaseColors={'上升期':'var(--success)','平台期':'var(--warn)','饱和期':'var(--danger)','衰退期':'var(--danger)'};
            const phaseColor=phaseColors[phase]||'var(--primary)';
            // Score cards
            const scores=fc.direction_score||{};
            let forecastHtml='<div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;flex-wrap:wrap">';
            forecastHtml+='<div style="background:var(--bg);padding:12px 20px;border-radius:10px;border:2px solid '+phaseColor+'"><div style="font-size:11px;color:var(--text3);margin-bottom:4px">发展阶段</div><div style="font-size:1.5em;font-weight:800;color:'+phaseColor+'">'+phase+'</div></div>';
            const scoreItems=[{k:'heat',l:'研究热度',icon:'🔥'},{k:'saturation',l:'饱和度',icon:'📦'},{k:'potential',l:'潜力',icon:'🚀'},{k:'entry_barrier',l:'入门门槛',icon:'🚧'}];
            scoreItems.forEach(s=>{
              const v=scores[s.k]||3;
              const barW=v*20;
              const sColor=v>=4?'var(--success)':v>=3?'var(--warn)':'var(--danger)';
              if(s.k==='saturation'||s.k==='entry_barrier'){/* 反向：低更好 */}
              forecastHtml+='<div style="background:var(--bg);padding:10px 16px;border-radius:10px;min-width:100px"><div style="font-size:11px;color:var(--text3);margin-bottom:4px">'+s.icon+' '+s.l+'</div><div style="font-size:1.3em;font-weight:800;color:var(--text)">'+v+'/5</div><div style="height:4px;background:var(--border);border-radius:2px;margin-top:4px"><div style="height:4px;width:'+barW+'%;background:'+sColor+';border-radius:2px"></div></div></div>';
            });
            forecastHtml+='</div>';
            // Phase reasoning
            if(fc.phase_reasoning)forecastHtml+='<div style="background:var(--bg);padding:14px;border-radius:8px;margin-bottom:16px;font-size:13px;color:var(--text2);line-height:1.7;border-left:3px solid '+phaseColor+'">'+fc.phase_reasoning+'</div>';
            // Timeline mini chart (领域级统计)
            const yd=tl.year_distribution||{};
            const years=Object.keys(yd).sort();
            if(years.length>0){
              const maxTotal=Math.max(...years.map(y=>(yd[y]||{}).total_in_field||(yd[y]||{}).count||0),1);
              forecastHtml+='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">📅 论文时间线（领域级统计）</h4><div style="display:flex;align-items:flex-end;gap:6px;height:80px;margin-bottom:16px;padding:0 4px">';
              years.forEach(y=>{
                const fieldTotal=(yd[y]||{}).total_in_field||0;
                const localCnt=(yd[y]||{}).count||0;
                const displayTotal=fieldTotal||localCnt;
                const h=Math.max(8,Math.round(displayTotal/maxTotal*70));
                const kws=((yd[y]||{}).keywords||[]).slice(0,2).join(', ');
                const label=fieldTotal?fieldTotal.toLocaleString():localCnt;
                const subLabel=fieldTotal?'领域总量':'本地';
                forecastHtml+='<div style="flex:1;text-align:center" title="'+kws+'"><div style="height:'+h+'px;background:linear-gradient(180deg,var(--primary),var(--primary2));border-radius:4px 4px 0 0;margin:0 auto;width:80%"></div><div style="font-size:10px;color:var(--text3);margin-top:4px">'+y+'</div><div style="font-size:9px;color:var(--primary)">'+label+'篇</div><div style="font-size:8px;color:var(--text3)">'+subLabel+'</div></div>';
              });
              forecastHtml+='</div>';
            }
            // Evolution paths
            const ePaths=ev.evolution_paths||[];
            if(ePaths.length>0){
              forecastHtml+='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">🔄 方法演化路径</h4>';
              ePaths.forEach(path=>{
                forecastHtml+='<div style="background:var(--bg);padding:14px;border-radius:8px;margin-bottom:10px">';
                forecastHtml+='<div style="font-size:13px;font-weight:600;color:var(--primary);margin-bottom:8px">'+path.path_name+(path.paradigm_shift?' <span style="font-size:10px;background:rgba(248,113,113,0.15);color:var(--danger);padding:2px 6px;border-radius:3px">范式转换</span>':'')+'</div>';
                const stages=path.stages||[];
                stages.forEach((st,si)=>{
                  forecastHtml+='<div style="display:flex;align-items:baseline;gap:8px;margin:6px 0;padding-left:8px;border-left:2px solid var(--accent)">';
                  forecastHtml+='<span style="font-size:11px;color:var(--text3);min-width:80px">'+st.time_range+'</span>';
                  forecastHtml+='<span style="font-size:12px;color:var(--text)">'+st.core_method+'</span>';
                  if(st.key_improvement)forecastHtml+='<span style="font-size:11px;color:var(--success)">→ '+st.key_improvement+'</span>';
                  if(st.key_limitation)forecastHtml+='<span style="font-size:11px;color:var(--danger)">('+st.key_limitation+')</span>';
                  forecastHtml+='</div>';
                });
                if(path.shift_reason)forecastHtml+='<div style="font-size:11px;color:var(--accent);margin-top:4px;padding-left:8px">💡 '+path.shift_reason+'</div>';
                forecastHtml+='</div>';
              });
            }
            // Forecast timeline
            const fcs=fc.forecast||[];
            if(fcs.length>0){
              forecastHtml+='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">🔮 趋势预测</h4><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px;margin-bottom:16px">';
              fcs.forEach(f=>{
                const confPct=Math.round((f.confidence||0.5)*100);
                forecastHtml+='<div style="background:var(--bg);padding:12px;border-radius:8px"><div style="font-size:11px;color:var(--primary);margin-bottom:4px">'+f.time_horizon+' <span style="color:var(--text3)">置信度:'+confPct+'%</span></div><div style="font-size:12px;color:var(--text2);line-height:1.6">'+f.prediction+'</div></div>';
              });
              forecastHtml+='</div>';
            }
            // Investment advice
            const adv=fc.investment_advice||{};
            if(Object.keys(adv).length>0){
              forecastHtml+='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">💡 投入建议</h4><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px">';
              const advMap={for_beginner:{l:'🎓 新手建议',color:'var(--primary)'},for_advanced:{l:'🔬 进阶建议',color:'var(--accent)'},low_hanging_fruit:{l:'🍎 低垂果实',color:'var(--success)'},high_risk_high_reward:{l:'🎰 高风险高回报',color:'var(--danger)'}};
              for(const[ak,al] of Object.entries(advMap)){
                if(adv[ak])forecastHtml+='<div style="background:var(--bg);padding:12px;border-radius:8px;border-left:3px solid '+al.color+'"><div style="font-size:11px;color:'+al.color+';margin-bottom:4px">'+al.l+'</div><div style="font-size:12px;color:var(--text2)">'+adv[ak]+'</div></div>';
              }
              forecastHtml+='</div>';
            }
            // Red/Green flags
            const rf=fc.red_flags||[];
            const gf=fc.green_flags||[];
            if(rf.length||gf.length){
              forecastHtml+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">';
              if(gf.length){forecastHtml+='<div style="background:rgba(52,211,153,0.05);padding:12px;border-radius:8px;border:1px solid rgba(52,211,153,0.2)"><div style="font-size:12px;color:var(--success);margin-bottom:6px">✅ 积极信号</div>'+gf.map(g=>'<div style="font-size:11px;color:var(--text2);margin:3px 0;padding-left:8px;border-left:2px solid var(--success)">'+g+'</div>').join('')+'</div>';}
              if(rf.length){forecastHtml+='<div style="background:rgba(248,113,113,0.05);padding:12px;border-radius:8px;border:1px solid rgba(248,113,113,0.2)"><div style="font-size:12px;color:var(--danger);margin-bottom:6px">⚠️ 风险信号</div>'+rf.map(r=>'<div style="font-size:11px;color:var(--text2);margin:3px 0;padding-left:8px;border-left:2px solid var(--danger)">'+r+'</div>').join('')+'</div>';}
              forecastHtml+='</div>';
            }
            document.getElementById('trend-forecast').innerHTML=forecastHtml;
          }
          if(d.step==='profile'){
            const pb=document.getElementById('profile-box');pb.style.display='block';
            const pg=d.profile_graph||{};
            // Core domains
            const domains=pg.core_domains||[];
            if(domains.length>0){
              let domHtml='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">🎯 核心研究领域</h4><div style="display:flex;flex-wrap:wrap;gap:8px">';
              domains.forEach(dm=>{
                const masteryColors={'精通':'var(--success)','熟悉':'var(--primary)','了解':'var(--text3)'};
                const mc=masteryColors[dm.mastery]||'var(--text3)';
                domHtml+='<div style="background:var(--bg);padding:14px;border-radius:10px;min-width:160px;flex:1;border-left:3px solid '+mc+'">';
                domHtml+='<div style="font-size:14px;font-weight:700;color:var(--text);margin-bottom:4px">'+dm.name+'</div>';
                domHtml+='<div style="display:flex;gap:6px;margin-bottom:6px"><span style="font-size:10px;padding:2px 6px;border-radius:3px;background:rgba(56,189,248,0.1);color:'+mc+'">'+dm.mastery+'</span><span style="font-size:10px;color:var(--text3)">'+(dm.papers_count||0)+'篇论文</span></div>';
                if(dm.keywords)domHtml+='<div style="font-size:11px;color:var(--text2)">'+dm.keywords.join(' · ')+'</div>';
                domHtml+='</div>';
              });
              domHtml+='</div>';
              document.getElementById('profile-domains').innerHTML=domHtml;
            }
            // Mastered methods
            const methods=pg.mastered_methods||[];
            if(methods.length>0){
              let methHtml='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">🔧 已掌握方法</h4><div style="display:flex;flex-wrap:wrap;gap:6px">';
              methods.forEach(m=>{
                const depthColors={'深入理解':'var(--success)','了解原理':'var(--primary)','仅知道存在':'var(--text3)'};
                const dc=depthColors[m.depth]||'var(--text3)';
                const title=m.from_papers?'来自: '+m.from_papers.join(', '):'';
                methHtml+='<span style="font-size:11px;padding:6px 12px;border-radius:6px;background:var(--bg);border:1px solid '+dc+';color:var(--text);cursor:default" title="'+title+'"><span style="color:'+dc+'">●</span> '+m.method+' <span style="color:var(--text3);font-size:10px">'+m.depth+'</span></span>';
              });
              methHtml+='</div>';
              document.getElementById('profile-methods').innerHTML=methHtml;
            }
            // Knowledge blindspots
            const blindspots=pg.knowledge_blindspots||[];
            if(blindspots.length>0){
              let bsHtml='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">🕳️ 知识盲区</h4>';
              blindspots.forEach(bs=>{
                const impColors={'高':'var(--danger)','中':'var(--warn)','低':'var(--text3)'};
                const ic=impColors[bs.importance]||'var(--text3)';
                bsHtml+='<div style="background:var(--bg);padding:12px;border-radius:8px;margin-bottom:8px;border-left:3px solid '+ic+'">';
                bsHtml+='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px"><strong style="font-size:13px;color:var(--text)">'+bs.area+'</strong><span style="font-size:10px;padding:2px 6px;border-radius:3px;background:rgba(248,113,113,0.1);color:'+ic+'">重要性: '+bs.importance+'</span></div>';
                if(bs.reason)bsHtml+='<div style="font-size:11px;color:var(--text2);margin-bottom:4px">'+bs.reason+'</div>';
                if(bs.suggested_queries&&bs.suggested_queries.length>0)bsHtml+='<div style="font-size:10px;color:var(--primary)">🔍 建议搜索: '+bs.suggested_queries.join(' | ')+'</div>';
                bsHtml+='</div>';
              });
              document.getElementById('profile-blindspots').innerHTML=bsHtml;
            }
            // Unread relevant
            const unread=pg.unread_relevant||[];
            if(unread.length>0){
              let urHtml='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">📚 高相关但未覆盖</h4>';
              unread.forEach(u=>{
                urHtml+='<div style="background:var(--bg);padding:12px;border-radius:8px;margin-bottom:8px;border-left:3px solid var(--primary2)">';
                urHtml+='<div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px">'+u.area+'</div>';
                if(u.reason)urHtml+='<div style="font-size:11px;color:var(--text2);margin-bottom:4px">'+u.reason+'</div>';
                if(u.suggested_search)urHtml+='<div style="font-size:10px;color:var(--primary2)">🔍 检索: '+u.suggested_search+'</div>';
                urHtml+='</div>';
              });
              document.getElementById('profile-unread').innerHTML=urHtml;
            }
            // Research style
            const style=pg.research_style||{};
            if(Object.keys(style).length>0){
              let stHtml='<h4 style="color:var(--text);font-size:13px;margin-bottom:8px">👤 研究风格画像</h4>';
              stHtml+='<div style="background:var(--bg);padding:16px;border-radius:10px;border:1px solid var(--border)">';
              if(style.description)stHtml+='<div style="font-size:14px;color:var(--text);margin-bottom:10px;line-height:1.6">'+style.description+'</div>';
              stHtml+='<div style="display:flex;flex-wrap:wrap;gap:8px">';
              const styleMap={preference:'偏好',depth:'深度',trend:'趋势'};
              for(const[sk,sv] of Object.entries(styleMap)){
                if(style[sk])stHtml+='<span style="font-size:11px;padding:4px 10px;border-radius:6px;background:rgba(129,140,248,0.1);color:var(--primary2);border:1px solid rgba(129,140,248,0.2)">'+sv+': '+style[sk]+'</span>';
              }
              stHtml+='</div></div>';
              document.getElementById('profile-style').innerHTML=stHtml;
            }
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

// ── Trend Standalone ──
async function doTrendStandalone(){
  const q=document.getElementById('trend-query').value.trim();if(!q)return;
  const resultEl=document.getElementById('trend-standalone-result');
  const emptyEl=document.getElementById('trend-standalone-empty');
  resultEl.style.display='none';emptyEl.style.display='none';
  // Show loading
  document.getElementById('trend-sa-phase').innerHTML='<div style="text-align:center;padding:40px;color:var(--text3)">⏳ 正在检索领域数据并分析趋势，请稍候...</div>';
  resultEl.style.display='block';
  try{
    const resp=await fetch('/trend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})});
    const data=await resp.json();
    if(data.error){document.getElementById('trend-sa-phase').innerHTML='<div class="empty-state"><div class="icon">📭</div><p>'+data.error+'</p></div>';return;}
    const fc=data.trend_forecast||{};
    const tl=data.timeline||{};
    const ev=data.evolution||{};
    // Phase badge
    const phase=fc.overall_phase||'分析中';
    const phaseColors={'上升期':'var(--success)','平台期':'var(--warn)','饱和期':'var(--danger)','衰退期':'var(--danger)'};
    const pc=phaseColors[phase]||'var(--primary)';
    document.getElementById('trend-sa-phase').innerHTML='<div style="text-align:center"><span class="trend-phase-badge" style="border-color:'+pc+';color:'+pc+'">'+phase+'</span></div>';
    // Score cards
    const scores=fc.direction_score||{};
    const scoreItems=[{k:'heat',l:'🔥 研究热度'},{k:'saturation',l:'📦 饱和度'},{k:'potential',l:'🚀 潜力'},{k:'entry_barrier',l:'🚧 入门门槛'}];
    let scoresHtml='';
    scoreItems.forEach(s=>{
      const v=scores[s.k]||3;
      const sColor=v>=4?'var(--success)':v>=3?'var(--warn)':'var(--danger)';
      scoresHtml+='<div class="trend-score-card"><div class="num" style="color:'+sColor+'">'+v+'/5</div><div class="label">'+s.l+'</div><div class="trend-score-bar"><div style="width:'+v*20+'%;background:'+sColor+'"></div></div></div>';
    });
    document.getElementById('trend-sa-scores').innerHTML=scoresHtml;
    // Reasoning
    if(fc.phase_reasoning)document.getElementById('trend-sa-reasoning').innerHTML='<div class="card"><div style="font-size:13px;color:var(--text2);line-height:1.7;border-left:3px solid '+pc+';padding-left:12px">'+fc.phase_reasoning+'</div></div>';
    else document.getElementById('trend-sa-reasoning').innerHTML='';
    // Timeline (领域级统计)
    const yd=tl.year_distribution||{};
    const years=Object.keys(yd).sort();
    if(years.length>0){
      const maxTotal=Math.max(...years.map(y=>(yd[y]||{}).total_in_field||(yd[y]||{}).count||0),1);
      let tlHtml='<div class="card"><h3>📅 论文时间线（领域级统计）</h3><div style="display:flex;align-items:flex-end;gap:8px;height:120px;padding:0 8px">';
      years.forEach(y=>{
        const localCnt=(yd[y]||{}).count||0;
        const fieldTotal=(yd[y]||{}).total_in_field||0;
        const displayTotal=fieldTotal||localCnt;
        const h=Math.max(12,Math.round(displayTotal/maxTotal*100));
        const kws=((yd[y]||{}).keywords||[]).slice(0,2).join(', ');
        const label=fieldTotal?fieldTotal.toLocaleString()+'篇':localCnt+'篇';
        const subLabel=fieldTotal?'(领域总量)':'(本地样本)';
        tlHtml+='<div style="flex:1;text-align:center" title="'+kws+'"><div style="height:'+h+'px;background:linear-gradient(180deg,var(--primary),var(--primary2));border-radius:4px 4px 0 0;margin:0 auto;width:70%"></div><div style="font-size:11px;color:var(--text3);margin-top:4px">'+y+'</div><div style="font-size:11px;font-weight:600;color:var(--primary)">'+label+'</div><div style="font-size:9px;color:var(--text3)">'+subLabel+'</div></div>';
      });
      tlHtml+='</div></div>';
      document.getElementById('trend-sa-timeline').innerHTML=tlHtml;
    }
    // Evolution
    const ePaths=ev.evolution_paths||[];
    if(ePaths.length>0){
      let evHtml='<div class="card"><h3>🔄 方法演化路径</h3>';
      ePaths.forEach(path=>{
        evHtml+='<div style="background:var(--bg);padding:14px;border-radius:8px;margin-bottom:10px">';
        evHtml+='<div style="font-size:13px;font-weight:600;color:var(--primary);margin-bottom:8px">'+path.path_name+(path.paradigm_shift?' <span style="font-size:10px;background:rgba(248,113,113,0.15);color:var(--danger);padding:2px 6px;border-radius:3px">范式转换</span>':'')+'</div>';
        (path.stages||[]).forEach(st=>{
          evHtml+='<div style="display:flex;align-items:baseline;gap:8px;margin:6px 0;padding-left:8px;border-left:2px solid var(--accent)">';
          evHtml+='<span style="font-size:11px;color:var(--text3);min-width:80px">'+st.time_range+'</span>';
          evHtml+='<span style="font-size:12px;color:var(--text)">'+st.core_method+'</span>';
          if(st.key_improvement)evHtml+='<span style="font-size:11px;color:var(--success)">→ '+st.key_improvement+'</span>';
          evHtml+='</div>';
        });
        if(path.shift_reason)evHtml+='<div style="font-size:11px;color:var(--accent);margin-top:4px;padding-left:8px">💡 '+path.shift_reason+'</div>';
        evHtml+='</div>';
      });
      evHtml+='</div>';
      document.getElementById('trend-sa-evolution').innerHTML=evHtml;
    }
    // Forecast timeline
    const fcs=fc.forecast||[];
    if(fcs.length>0){
      let fcHtml='<div class="card"><h3>🔮 趋势预测</h3><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px">';
      fcs.forEach(f=>{
        const confPct=Math.round((f.confidence||0.5)*100);
        fcHtml+='<div style="background:var(--bg);padding:12px;border-radius:8px"><div style="font-size:11px;color:var(--primary);margin-bottom:4px">'+f.time_horizon+' <span style="color:var(--text3)">置信度:'+confPct+'%</span></div><div style="font-size:12px;color:var(--text2);line-height:1.6">'+f.prediction+'</div></div>';
      });
      fcHtml+='</div></div>';
      document.getElementById('trend-sa-forecast').innerHTML=fcHtml;
    }
    // Investment advice
    const adv=fc.investment_advice||{};
    if(Object.keys(adv).length>0){
      let advHtml='<div class="card"><h3>💡 投入建议</h3><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">';
      const advMap={for_beginner:{l:'🎓 新手建议',c:'var(--primary)'},for_advanced:{l:'🔬 进阶建议',c:'var(--accent)'},low_hanging_fruit:{l:'🍎 低垂果实',c:'var(--success)'},high_risk_high_reward:{l:'🎰 高风险高回报',c:'var(--danger)'}};
      for(const[ak,al] of Object.entries(advMap)){
        if(adv[ak])advHtml+='<div style="background:var(--bg);padding:12px;border-radius:8px;border-left:3px solid '+al.c+'"><div style="font-size:11px;color:'+al.c+';margin-bottom:4px">'+al.l+'</div><div style="font-size:12px;color:var(--text2)">'+adv[ak]+'</div></div>';
      }
      advHtml+='</div></div>';
      document.getElementById('trend-sa-advice').innerHTML=advHtml;
    }
    // Red/Green flags
    const rf=fc.red_flags||[];
    const gf=fc.green_flags||[];
    if(rf.length||gf.length){
      let flHtml='<div class="card"><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">';
      if(gf.length)flHtml+='<div style="background:rgba(52,211,153,0.05);padding:12px;border-radius:8px;border:1px solid rgba(52,211,153,0.2)"><div style="font-size:12px;color:var(--success);margin-bottom:6px">✅ 积极信号</div>'+gf.map(g=>'<div style="font-size:11px;color:var(--text2);margin:3px 0;padding-left:8px;border-left:2px solid var(--success)">'+g+'</div>').join('')+'</div>';
      if(rf.length)flHtml+='<div style="background:rgba(248,113,113,0.05);padding:12px;border-radius:8px;border:1px solid rgba(248,113,113,0.2)"><div style="font-size:12px;color:var(--danger);margin-bottom:6px">⚠️ 风险信号</div>'+rf.map(r=>'<div style="font-size:11px;color:var(--text2);margin:3px 0;padding-left:8px;border-left:2px solid var(--danger)">'+r+'</div>').join('')+'</div>';
      flHtml+='</div></div>';
      document.getElementById('trend-sa-flags').innerHTML=flHtml;
    }
  }catch(e){document.getElementById('trend-sa-phase').innerHTML='<div class="empty-state"><div class="icon">❌</div><p>请求失败: '+e.message+'</p></div>';}
}

// ── Profile Standalone ──
async function doProfileStandalone(){
  const resultEl=document.getElementById('profile-standalone-result');
  const emptyEl=document.getElementById('profile-standalone-empty');
  resultEl.style.display='none';emptyEl.style.display='none';
  try{
    const resp=await fetch('/profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:''})});
    const data=await resp.json();
    if(data.error||!data.profile_graph||Object.keys(data.profile_graph).length===0){emptyEl.innerHTML='<div class="icon">📭</div><p>'+(data.error||'暂无数据，请先生成综述或检索论文')+'</p>';emptyEl.style.display='block';return;}
    const pg=data.profile_graph;
    resultEl.style.display='block';
    // Radar chart
    drawRadarChart(pg.core_domains||[]);
    // Mastery bars
    renderMasteryBars(pg.mastered_methods||[]);
    // Blindspots
    renderBlindspots(pg.knowledge_blindspots||[]);
    // Unread
    renderUnread(pg.unread_relevant||[]);
    // Style
    renderStyle(pg.research_style||{});
  }catch(e){emptyEl.innerHTML='<div class="icon">❌</div><p>请求失败: '+e.message+'</p>';emptyEl.style.display='block';}
}

function drawRadarChart(domains){
  const canvas=document.getElementById('profile-radar');
  if(!canvas||!domains.length)return;
  const ctx=canvas.getContext('2d');
  const W=canvas.width,H=canvas.height;
  const cx=W/2,cy=H/2;
  const R=Math.min(W,H)/2-50;
  const n=domains.length;
  const masteryMap={'精通':5,'熟悉':3,'了解':1};
  const isDark=getTheme()==='dark';
  const masteryColorMap={'精通':isDark?'#34d399':'#059669','熟悉':isDark?'#38bdf8':'#0284c7','了解':isDark?'#64748b':'#94a3b8'};

  ctx.clearRect(0,0,W,H);

  // Grid circles
  for(let ring=1;ring<=5;ring++){
    const r=R*ring/5;
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);
    ctx.strokeStyle=isDark?'rgba(56,189,248,0.1)':'rgba(2,132,199,0.08)';ctx.lineWidth=1;ctx.stroke();
  }

  // Axis lines + labels
  const angleStep=Math.PI*2/n;
  domains.forEach((d,i)=>{
    const a=-Math.PI/2+angleStep*i;
    const ex=cx+R*Math.cos(a),ey=cy+R*Math.sin(a);
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ex,ey);
    ctx.strokeStyle=isDark?'rgba(56,189,248,0.15)':'rgba(2,132,199,0.12)';ctx.lineWidth=1;ctx.stroke();
    // Label
    const lx=cx+(R+28)*Math.cos(a),ly=cy+(R+28)*Math.sin(a);
    ctx.fillStyle=masteryColorMap[d.mastery]||(isDark?'#94a3b8':'#475569');
    ctx.font='bold 12px Inter,sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText(d.name,lx,ly-8);
    ctx.fillStyle=isDark?'#64748b':'#94a3b8';ctx.font='10px Inter,sans-serif';
    ctx.fillText(d.mastery+' · '+(d.papers_count||0)+'篇',lx,ly+8);
  });

  // Data polygon
  ctx.beginPath();
  domains.forEach((d,i)=>{
    const a=-Math.PI/2+angleStep*i;
    const v=masteryMap[d.mastery]||1;
    const r=R*v/5;
    const x=cx+r*Math.cos(a),y=cy+r*Math.sin(a);
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  });
  ctx.closePath();
  ctx.fillStyle=isDark?'rgba(56,189,248,0.15)':'rgba(2,132,199,0.12)';ctx.fill();
  ctx.strokeStyle=isDark?'#38bdf8':'#0284c7';ctx.lineWidth=2;ctx.stroke();

  // Data points
  domains.forEach((d,i)=>{
    const a=-Math.PI/2+angleStep*i;
    const v=masteryMap[d.mastery]||1;
    const r=R*v/5;
    const x=cx+r*Math.cos(a),y=cy+r*Math.sin(a);
    ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);
    ctx.fillStyle=masteryColorMap[d.mastery]||(isDark?'#94a3b8':'#475569');ctx.fill();
    ctx.strokeStyle=isDark?'#0a0f1a':'#ffffff';ctx.lineWidth=2;ctx.stroke();
  });

  // Legend
  let legendHtml='';
  const legItems=[{c:masteryColorMap['精通'],l:'精通 (5/5)'},{c:masteryColorMap['熟悉'],l:'熟悉 (3/5)'},{c:masteryColorMap['了解'],l:'了解 (1/5)'}];
  legItems.forEach(li=>{legendHtml+='<span class="legend-item"><span class="legend-dot" style="background:'+li.c+'"></span>'+li.l+'</span>';});
  document.getElementById('radar-legend').innerHTML=legendHtml;
}

function renderMasteryBars(methods){
  if(!methods.length){document.getElementById('profile-sa-mastery').innerHTML='<p style="color:var(--text3);font-size:13px">暂无数据</p>';return;}
  const depthMap={'深入理解':5,'了解原理':3,'仅知道存在':1};
  const isDark=getTheme()==='dark';
  const depthColorMap={'深入理解':isDark?'#34d399':'#059669','了解原理':isDark?'#38bdf8':'#0284c7','仅知道存在':isDark?'#64748b':'#94a3b8'};
  let html='';
  methods.forEach(m=>{
    const v=depthMap[m.depth]||1;
    const c=depthColorMap[m.depth]||'#64748b';
    const pct=v*20;
    const paperInfo=m.from_papers?'来自: '+m.from_papers.join(', '):'';
    html+='<div class="mastery-row">';
    html+='<span class="mastery-label" title="'+paperInfo+'">'+m.method+'</span>';
    html+='<div class="mastery-bar-track"><div class="mastery-bar-fill" style="width:'+pct+'%;background:'+c+'"></div></div>';
    html+='<span class="mastery-depth" style="background:'+c+'22;color:'+c+'">'+m.depth+'</span>';
    if(m.domain)html+='<span class="mastery-domain-tag">'+m.domain+'</span>';
    html+='</div>';
  });
  document.getElementById('profile-sa-mastery').innerHTML=html;
}

function renderBlindspots(blindspots){
  if(!blindspots.length){document.getElementById('profile-sa-blindspots').innerHTML='<p style="color:var(--success);font-size:13px">✅ 暂未发现明显盲区</p>';return;}
  let html='';
  const impColor={'高':'var(--danger)','中':'var(--warn)','低':'var(--text3)'};
  blindspots.forEach(bs=>{
    const ic=impColor[bs.importance]||'var(--text3)';
    html+='<div style="background:var(--bg);padding:14px;border-radius:8px;margin-bottom:10px;border-left:3px solid '+ic+'">';
    html+='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px"><strong style="font-size:14px;color:var(--text)">'+bs.area+'</strong><span style="font-size:11px;padding:2px 8px;border-radius:4px;background:'+ic+'22;color:'+ic+'">重要性: '+bs.importance+'</span></div>';
    if(bs.reason)html+='<div style="font-size:12px;color:var(--text2);margin-bottom:6px">'+bs.reason+'</div>';
    if(bs.suggested_queries&&bs.suggested_queries.length)html+='<div style="font-size:11px;color:var(--primary)">🔍 建议搜索: '+bs.suggested_queries.map(q=>'<span style="padding:2px 6px;background:rgba(56,189,248,0.1);border-radius:3px;margin-right:4px">'+q+'</span>').join('')+'</div>';
    html+='</div>';
  });
  document.getElementById('profile-sa-blindspots').innerHTML=html;
}

function renderUnread(unread){
  if(!unread.length){document.getElementById('profile-sa-unread').innerHTML='<p style="color:var(--text3);font-size:13px">暂无推荐</p>';return;}
  let html='';
  unread.forEach(u=>{
    html+='<div style="background:var(--bg);padding:14px;border-radius:8px;margin-bottom:10px;border-left:3px solid var(--primary2)">';
    html+='<div style="font-size:14px;font-weight:600;color:var(--text);margin-bottom:4px">'+u.area+'</div>';
    if(u.reason)html+='<div style="font-size:12px;color:var(--text2);margin-bottom:6px">'+u.reason+'</div>';
    if(u.suggested_search)html+='<div style="font-size:11px;color:var(--primary2)">🔍 检索: <span style="padding:2px 6px;background:rgba(129,140,248,0.1);border-radius:3px">'+u.suggested_search+'</span></div>';
    html+='</div>';
  });
  document.getElementById('profile-sa-unread').innerHTML=html;
}

function renderStyle(style){
  if(!style||Object.keys(style).length===0){document.getElementById('profile-sa-style').innerHTML='<p style="color:var(--text3);font-size:13px">数据不足，无法生成画像</p>';return;}
  let html='<div style="background:var(--bg);padding:18px;border-radius:10px;border:1px solid var(--border)">';
  if(style.description)html+='<div style="font-size:14px;color:var(--text);margin-bottom:12px;line-height:1.7">'+style.description+'</div>';
  html+='<div style="display:flex;flex-wrap:wrap;gap:8px">';
  const styleMap={preference:'偏好',depth:'深度',trend:'趋势'};
  for(const[sk,sv] of Object.entries(styleMap)){
    if(style[sk])html+='<span style="font-size:12px;padding:6px 14px;border-radius:8px;background:rgba(129,140,248,0.1);color:var(--primary2);border:1px solid rgba(129,140,248,0.2)">'+sv+': '+style[sk]+'</span>';
  }
  html+='</div></div>';
  document.getElementById('profile-sa-style').innerHTML=html;
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
