"""Prompt 模板 — 多Agent架构"""

# ── Coordinator ──
COORDINATOR_PLAN = """你是一个科研Agent系统的协调者，负责规划任务。

用户查询: {query}
用户偏好: {preferences}
上轮Critic反馈: {feedback}

请分析用户查询，输出JSON格式的执行计划:

```json
{{
    "search_queries": ["query1", "query2", "query3"],
    "max_papers": 10,
    "focus_areas": ["方向1", "方向2"],
    "needs_rerun": false,
    "critic_feedback": ""
}}
```

要求:
- search_queries: 2-3个不同角度的检索query，覆盖用户意图的不同方面
- max_papers: 检索论文数上限
- focus_areas: 重点关注的技术方向
- 如果是Critic反馈后的重跑，needs_rerun=true，critic_feedback填入反馈内容

只输出JSON，不要其他内容。"""

# ── Search Agent ──
QUERY_REWRITE = """将以下科研查询改写为2-3个更具体、更适合学术检索的英文关键词组合。

原始查询: {query}

要求:
- 每行一个改写query
- 使用英文关键词（学术检索效果更好）
- 从不同角度覆盖原始意图
- 不加编号，不加引号

例如原始查询"CIL最新进展"可改写为:
class-incremental learning 2024 2025 survey
continual learning catastrophic forgetting recent methods
incremental object detection new framework"""

# ── Analysis Agent ──
PAPER_COMPARISON = """你是一个科研分析助手，请对以下论文进行对比分析。

{papers_info}

对比焦点: {focus}

请生成Markdown格式的对比表格和简要分析:

## 论文对比矩阵

| 维度 | 论文1 | 论文2 | 论文3 |
|------|-------|-------|-------|
| 核心思路 | | | |
| 技术路线 | | | |
| 增量策略 | | | |
| 评估数据集 | | | |
| 主要优势 | | | |
| 主要局限 | | | |

## 简要分析
（2-3段，总结各方法的异同和适用场景）"""

# ── Writing Agent ──
INTENT_CLASSIFICATION = """你是一个科研助手，需要判断用户的意图。

用户输入: {query}

请从以下意图中选择一个:
- search: 用户想搜索/查找某个方向的论文
- qa: 用户对已有论文库提问，需要跨论文检索回答
- summarize: 用户想要生成某个方向的文献综述
- refine: 用户想对上一次结果进行修正/追问

只输出意图关键词，不要其他内容。"""

PAPER_SEARCH_QUERY = """将用户的自然语言查询转换为适合学术检索的关键词。

用户输入: {query}
用户偏好方向: {preferences}

输出格式: 用空格分隔的英文关键词，不要其他内容。
例如: class-incremental learning fine-grained detection"""

LITERATURE_SUMMARY = """你是一个科研文献综述助手。请根据以下论文信息生成结构化的文献综述。

研究方向: {query}
论文列表:
{papers_context}

RAG 召回的补充上下文:
{rag_context}

请按以下结构生成综述:

## 1. 研究背景与动机
（该方向为什么重要，核心问题是什么）

## 2. 主流方法与技术路线
（按技术范式分类，每类引用具体论文）

## 3. 关键挑战与开放问题
（当前方法的局限性）

## 4. 发展趋势
（未来可能的方向）

要求:
- 每个观点必须引用具体论文（标注[作者, 年份]）
- 用连贯段落写作，不要分点编号
- 语言: 中文"""

QA_PROMPT = """你是一个科研文献问答助手。请基于以下论文上下文回答用户问题。

用户问题: {question}

相关论文上下文:
{rag_context}

要求:
- 回答必须基于上述论文内容，不编造
- 引用具体论文支撑观点
- 如果上下文不足以回答，明确说明"""

REVISION_PROMPT = """你是一个科研文献综述的修订助手。根据评审反馈对综述进行修订。

研究方向: {query}

原始综述:
{original_summary}

评审反馈:
{critic_feedback}

补充上下文（可用于补充论据）:
{rag_context}

请根据反馈修订综述，要求:
- 针对反馈中指出的具体问题逐一改进
- 保留原文中正确的部分，不要全部重写
- 补充缺失的论文引用
- 保持结构和语言风格一致
- 语言: 中文"""

# ── Critic Agent ──
CRITIC_EVAL = """你是一个科研文献综述的评审专家。请评估以下综述的质量。

研究方向: {query}

待评估综述:
{summary}

参考论文列表:
{papers_context}

请从三个维度评分（1-5分），并给出具体改进建议。输出JSON格式:

```json
{{
    "coverage_score": 4.0,
    "accuracy_score": 3.5,
    "coherence_score": 4.0,
    "feedback": "具体改进建议：1. ... 2. ... 3. ..."
}}
```

评分标准:
- coverage_score（覆盖度）: 综述是否覆盖了参考论文的主要贡献？是否遗漏重要方法？
- accuracy_score（准确性）: 论文引用是否正确？方法描述是否准确？有无编造内容？
- coherence_score（连贯性）: 逻辑结构是否清晰？段落衔接是否自然？

feedback要求:
- 必须具体，指出哪段哪句有问题
- 给出可操作的改进建议
- 如果覆盖不足，指出遗漏了哪些论文的哪些贡献
"""

# ── 旧版兼容 ──
PAPER_FILTER = """根据以下标准筛选高价值论文:

筛选条件:
- 最低引用数: {min_citations}
- 最早年份: {year_from}
- 用户偏好方向: {preferences}

待筛选论文:
{papers_json}

请返回筛选后的论文ID列表（每行一个），不要其他内容。"""

REFINE_PROMPT = """用户对之前的检索结果提出了修正要求。

原始查询: {original_query}
之前的综述/结果摘要: {previous_result}
用户修正: {refinement}

请根据修正要求，给出调整后的检索关键词和筛选条件建议。"""
