"""
Research Copilot 评估框架

用法:
    python evaluate.py                    # 运行全部评测
    python evaluate.py --intent           # 只测意图识别
    python evaluate.py --search           # 只测检索相关性
    python evaluate.py --summary          # 只测综述质量（需人工评分）
    python evaluate.py --latency          # 只测端到端延迟
"""
import json
import time
import argparse
from datetime import datetime

# ── 评测数据集 ──

INTENT_TEST_CASES = [
    {"query": "帮我查一下CIL方向的最新论文", "expected": "search"},
    {"query": "class-incremental learning的最新进展", "expected": "search"},
    {"query": "帮我生成一篇关于OOD检测的文献综述", "expected": "summarize"},
    {"query": "综述一下continual learning的方法", "expected": "summarize"},
    {"query": "iVoro和AANets哪个更适合细粒度场景？", "expected": "qa"},
    {"query": "现有方法在CIFAR-100上的性能对比", "expected": "qa"},
    {"query": "刚才的结果里有没有做增量检测的？", "expected": "refine"},
    {"query": "再补充几篇2025年的论文", "expected": "refine"},
    {"query": "细粒度图像分类的增量学习方法", "expected": "search"},
    {"query": "帮我总结RAG和向量检索的区别", "expected": "qa"},
    {"query": "写一篇关于Agent Skill Learning的综述", "expected": "summarize"},
    {"query": "上面那个综述里缺少了ReAct的讨论", "expected": "refine"},
    {"query": "检索多模态大模型的最新工作", "expected": "search"},
    {"query": "FAISS和Chroma的优劣对比", "expected": "qa"},
    {"query": "生成关于灾难性遗忘缓解方法的综述", "expected": "summarize"},
    {"query": "把范围缩小到CVPR和ICCV的论文", "expected": "refine"},
    {"query": "search for latest diffusion model papers", "expected": "search"},
    {"query": "compare LoRA and full fine-tuning", "expected": "qa"},
    {"query": "write a survey on prompt engineering", "expected": "summarize"},
    {"query": "上次漏了Tool Learning方向的论文", "expected": "refine"},
]

SEARCH_TEST_QUERIES = [
    {"query": "class-incremental learning object detection", "expected_keywords": ["incremental", "detection", "learning"]},
    {"query": "fine-grained image classification continual learning", "expected_keywords": ["fine-grained", "classification", "continual"]},
    {"query": "out-of-distribution detection open set recognition", "expected_keywords": ["OOD", "detection", "recognition"]},
    {"query": "RAG retrieval augmented generation survey", "expected_keywords": ["RAG", "retrieval", "generation"]},
    {"query": "agent tool use function calling LLM", "expected_keywords": ["agent", "tool", "LLM"]},
]

SUMMARY_TEST_QUERIES = [
    {"query": "class-incremental learning for fine-grained detection", "focus": "覆盖度、准确性、连贯性"},
    {"query": "OOD detection methods in open-world scenarios", "focus": "方法分类是否清晰"},
    {"query": "RAG技术在大模型中的应用", "focus": "技术路线覆盖"},
]


def evaluate_intent():
    """评测意图识别准确率"""
    from agent.graph import llm
    from agent.prompts import INTENT_CLASSIFICATION
    
    print("\n" + "="*60)
    print("📋 意图识别评测")
    print("="*60)
    
    correct = 0
    total = len(INTENT_TEST_CASES)
    results = []
    
    for i, case in enumerate(INTENT_TEST_CASES):
        prompt = INTENT_CLASSIFICATION.format(query=case["query"])
        response = llm.invoke(prompt)
        predicted = response.content.strip().lower()
        is_correct = predicted == case["expected"]
        if is_correct:
            correct += 1
        
        results.append({
            "query": case["query"],
            "expected": case["expected"],
            "predicted": predicted,
            "correct": is_correct,
        })
        status = "✅" if is_correct else "❌"
        print(f"  {status} [{i+1}/{total}] 期望={case['expected']} 预测={predicted} | {case['query'][:40]}")
    
    accuracy = correct / total * 100
    print(f"\n  意图识别准确率: {correct}/{total} = {accuracy:.1f}%")
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def evaluate_search():
    """评测检索相关性（precision@10）"""
    from agent.search_agent import search_papers
    
    print("\n" + "="*60)
    print("🔍 检索相关性评测")
    print("="*60)
    
    results = []
    
    for i, case in enumerate(SEARCH_TEST_QUERIES):
        print(f"\n  [{i+1}/{len(SEARCH_TEST_QUERIES)}] 查询: {case['query']}")
        
        papers = search_papers(llm=None, query=case["query"], search_queries=[case["query"]], max_results=10)
        
        # 自动评估：检查返回论文标题/摘要是否包含预期关键词
        relevant = 0
        for p in papers[:10]:
            text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
            if any(kw.lower() in text for kw in case["expected_keywords"]):
                relevant += 1
        
        precision = relevant / min(len(papers), 10) * 100 if papers else 0
        results.append({
            "query": case["query"],
            "returned": len(papers),
            "relevant": relevant,
            "precision@10": round(precision, 1),
        })
        print(f"    返回{len(papers)}篇, 相关{relevant}篇, P@10={precision:.1f}%")
    
    avg_precision = sum(r["precision@10"] for r in results) / len(results) if results else 0
    print(f"\n  平均 Precision@10: {avg_precision:.1f}%")
    return {"avg_precision": avg_precision, "results": results}


def evaluate_summary_quality():
    """评测综述质量（需人工评分）"""
    print("\n" + "="*60)
    print("📝 综述质量评测（需人工评分）")
    print("="*60)
    print("\n  请对以下生成的综述进行评分（1-5分）:")
    print("  - 覆盖度: 是否覆盖了该方向的主要方法和趋势")
    print("  - 准确性: 论文引用和方法描述是否正确")
    print("  - 连贯性: 逻辑结构是否清晰，段落衔接是否自然")
    print()
    
    results = []
    for case in SUMMARY_TEST_QUERIES:
        print(f"  评价方向: {case['query']}")
        print(f"  评价重点: {case['focus']}")
        print(f"  请运行: curl -N -X POST http://localhost:8000/research/stream -H \"Content-Type: application/json\" -d '{{\"query\":\"{case['query']}\", \"max_papers\":5}}'")
        print(f"  生成后请手动评分并记录")
        print()
        results.append({"query": case["query"], "focus": case["focus"], "manual_score": "待人工评分"})
    
    return {"note": "综述质量需人工评分", "results": results}


def evaluate_latency():
    """评测端到端延迟"""
    import httpx
    
    print("\n" + "="*60)
    print("⏱️  端到端延迟评测")
    print("="*60)
    
    test_queries = [
        "class-incremental learning",
        "OOD detection survey",
        "RAG for LLM applications",
    ]
    
    results = []
    for query in test_queries:
        start = time.time()
        try:
            resp = httpx.post(
                "http://localhost:8000/research/stream",
                json={"query": query, "max_papers": 5},
                timeout=300,
            )
            elapsed = time.time() - start
            results.append({"query": query, "latency": round(elapsed, 2), "status": "ok"})
            print(f"  ✅ {query}: {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start
            results.append({"query": query, "latency": round(elapsed, 2), "status": f"error: {e}"})
            print(f"  ❌ {query}: {elapsed:.2f}s (error)")
    
    if results:
        latencies = [r["latency"] for r in results if r["status"] == "ok"]
        if latencies:
            p50 = sorted(latencies)[len(latencies)//2]
            avg = sum(latencies) / len(latencies)
            print(f"\n  平均延迟: {avg:.2f}s | P50: {p50:.2f}s")
    
    return {"results": results}


def main():
    parser = argparse.ArgumentParser(description="Research Copilot 评估框架")
    parser.add_argument("--intent", action="store_true", help="只测意图识别")
    parser.add_argument("--search", action="store_true", help="只测检索相关性")
    parser.add_argument("--summary", action="store_true", help="只测综述质量（需人工）")
    parser.add_argument("--latency", action="store_true", help="只测延迟")
    args = parser.parse_args()
    
    run_all = not any([args.intent, args.search, args.summary, args.latency])
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": {},
    }
    
    if run_all or args.intent:
        report["results"]["intent"] = evaluate_intent()
    
    if run_all or args.search:
        report["results"]["search"] = evaluate_search()
    
    if run_all or args.summary:
        report["results"]["summary"] = evaluate_summary_quality()
    
    if run_all or args.latency:
        report["results"]["latency"] = evaluate_latency()
    
    # 保存报告
    report_path = f"./eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📊 评测报告已保存: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
