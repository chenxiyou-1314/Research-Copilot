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

# ── Novelty Agent ──
GAP_ANALYSIS = """你是一个资深的科研方向分析师。请从以下文献综述中，系统性地提取该领域的空白和未解决问题。

研究方向: {query}

文献综述:
{summary}

请从四个维度分析Gap，输出JSON格式:

```json
{{
    "methodological_gaps": [
        "现有方法在XX场景下缺乏有效的YY策略",
        "ZZ方法的泛化能力尚未在AA数据集上验证"
    ],
    "data_gaps": [
        "缺乏XX场景的大规模细粒度基准数据集",
        "现有评估协议未考虑YY分布偏移"
    ],
    "theoretical_gaps": [
        "XX现象的理论解释尚未建立",
        "ZZ与AA之间的理论联系有待揭示"
    ],
    "practical_gaps": [
        "XX方法在资源受限场景下的部署方案缺失",
        "ZZ的实时性要求与现有方法的延迟矛盾"
    ]
}}
```

要求:
- 每个维度至少列出2-3个具体gap
- gap必须具体，不能泛泛而谈（不能写"需要更多研究"）
- 每个gap必须可以转化为一个具体的研究问题
- 基于综述内容，不要编造"""

CROSS_DOMAIN_TRANSFER = """你是一个跨领域研究专家。请分析当前方向的空白，并从其他领域的成功方法中寻找可迁移的思路。

研究方向: {query}

当前方向的空白:
{gaps}

综述背景:
{summary}

请提出2-3个跨域迁移方向，输出JSON数组:

```json
[
    {{
        "source_domain": "NLP / CV / RL / 图神经网络 / 数据库 / 软件工程 / ...",
        "source_method": "源领域的具体方法名",
        "target_application": "迁移到当前领域后的具体应用方式",
        "why_transferable": "为什么这个迁移是合理的（两个领域的共性）",
        "expected_benefit": "迁移后预期能解决哪个gap，带来什么改进"
    }}
]
```

要求:
- 必须是真正有迁移价值的思路，不是硬凑的
- why_transferable要讲清楚两个领域在结构/数学/逻辑上的共性
- 优先选择近年热门且在源领域已验证有效的方法
- 不要从综述中已经提到的方法迁移（那些已经做了）"""

IDEA_GENERATION = """你是一个科研创新顾问。请基于Gap分析和跨域迁移思路，生成2-3个具体可执行的研究方向。

研究方向: {query}

Gap分析:
{gaps}

跨域迁移思路:
{transfers}

综述背景:
{summary}

请生成2-3个研究思路，输出JSON数组:

```json
[
    {{
        "title": "简洁的研究方向标题",
        "motivation": "2-3句话，为什么这个方向值得做，解决哪个gap",
        "technical_route": [
            "Step1: 具体的第一步，如数据集构建/模型设计",
            "Step2: 具体的第二步，如训练策略/评估方案",
            "Step3: 具体的第三步，如消融实验/对比分析"
        ],
        "expected_contribution": "预期的主要贡献（1-2句话）",
        "feasibility": "可行性评估：高/中/低 — 算力/数据/时间约束",
        "risk": "主要风险或不确定性"
    }}
]
```

要求:
- 每个思路必须指向一个可发表论文的具体方向
- technical_route要具体到可以开始写代码的程度
- feasibility要考虑现实约束（研究生单GPU场景）
- 优先选择"高可行性+高新颖性"的方向
- 至少有一个思路来自跨域迁移"""

NOVELTY_VERIFICATION = """你是一个论文新颖性审查专家。请判断以下研究思路是否具有新颖性，即是否与已有工作实质性不同。

待验证思路:
- 标题: {idea_title}
- 动机: {idea_motivation}
- 技术路线: {idea_route}

已有相关论文:
{existing_works}

请判断新颖性，输出JSON:

```json
{{
    "is_novel": true,
    "similar_works": [
        "Paper X做了YY，但与本思路在ZZ方面不同"
    ],
    "novelty_statement": "1-2句话总结本思路与已有工作的核心区别",
    "confidence": 0.7
}}
```

判断标准:
- is_novel: 如果没有任何已有工作做过完全相同的事情，就为true
- similar_works: 列出最相关的已有工作，说明相似和不同之处
- novelty_statement: 必须具体指出"与XX不同，本思路强调YY"
- confidence: 0-1，你对新颖性判断的置信度（仅基于标题和摘要，无法100%确认）
- 如果检索不到相关论文，confidence应较低（可能用词不同但已有人做过）"""
