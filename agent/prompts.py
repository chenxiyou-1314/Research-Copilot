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

# ── Method Decomposition Agent ──
METHOD_DECOMPOSITION = """你是一个科研方法解构专家。请将以下论文中的方法拆解为原子组件，每个组件应是一个独立可替换的技术选择。

研究方向: {query}

待解构论文:
{papers_context}

请对每篇论文进行方法解构，输出JSON格式:

```json
[
    {{
        "paper_title": "论文标题缩写",
        "paper_year": 2024,
        "components": {{
            "backbone": {{
                "name": "ResNet-18 / ViT-B/16 / ...",
                "detail": "具体使用方式的1句话说明",
                "is_pretrained": true,
                "parameter_count": "11.7M"
            }},
            "training_strategy": {{
                "name": "fine-tuning / linear probing / prompt tuning / ...",
                "detail": "具体策略的1句话说明",
                "key_hyperparams": "lr=0.01, epochs=100, ..."
            }},
            "loss_function": {{
                "name": "CE + KD / Focal Loss / ...",
                "detail": "损失函数设计的1句话说明",
                "components": ["cross-entropy", "knowledge distillation"]
            }},
            "data_augmentation": {{
                "name": "RandAugment / Mixup / None",
                "detail": "数据增强策略的1句话说明"
            }},
            "evaluation_protocol": {{
                "name": "CIL protocol / FSCIL protocol",
                "detail": "评估设置的1句话说明",
                "datasets": ["CIFAR-100", "ImageNet-Subset"],
                "metrics": ["accuracy", "forgetting measure"]
            }}
        }}
    }}
]
```

要求:
- 每个组件必须是可以独立替换的最小单元
- 如果论文某组件未明确说明，填 "Not specified"
- backbone包括模型架构和特征提取方式
- training_strategy包括训练方式、优化策略、正则化等
- loss_function包括所有损失项的组成
- data_augmentation包括训练时的数据增强
- evaluation_protocol包括数据集、指标、实验设置
- name字段尽量用领域内通用术语"""

METHOD_RECOMBINATION = """你是一个科研方法创新专家。请基于以下论文的方法解构结果，提出跨论文的方法重组方案。

研究方向: {query}

论文方法解构:
{decomposition}

研究空白(Gap分析):
{gaps}

请提出3-4个方法重组方案，每个方案从不同论文中选取组件重新组合，输出JSON数组:

```json
[
    {{
        "name": "方案简称，如 ProtoCIL-Prompt",
        "components": {{
            "backbone": "来自哪篇论文的哪个backbone",
            "training_strategy": "来自哪篇论文的哪个策略",
            "loss_function": "来自哪篇论文的哪个损失",
            "data_augmentation": "来自哪篇论文的哪个增强",
            "evaluation_protocol": "建议的评估协议（不必来自论文）"
        }},
        "source_papers": ["论文1缩写", "论文2缩写"],
        "motivation": "2-3句话，为什么这个组合值得尝试，解决哪个gap",
        "expected_synergy": "组件间预期的协同效应（1-2句话）",
        "target_gap": "该方案主要针对的gap编号或描述"
    }}
]
```

要求:
- 每个方案至少混合2篇不同论文的组件
- 不能简单地复制某篇论文的全部组件
- motivation必须指向一个具体的gap
- expected_synergy要说明为什么这些组件组合在一起比单独使用更好
- 优先选择"高可行性 + 高新颖性"的组合
- 至少一个方案包含跨领域的训练策略或损失函数"""

RECOMBINATION_VALIDATION = """你是一个科研可行性审查专家。请评估以下方法重组方案的技术可行性。

研究方向: {query}

重组方案:
{recombination}

请从以下维度评估每个方案，输出JSON数组:

```json
[
    {{
        "name": "方案名称",
        "compatibility_score": 4,
        "implementation_difficulty": "中等",
        "risk_factors": [
            "风险1: 具体描述",
            "风险2: 具体描述"
        ],
        "mitigation": "降低风险的具体措施",
        "overall_feasibility": "高/中/低",
        "quick_start": "2-3句话，如何快速验证这个方案（最小原型）",
        "potential_pitfall": "最可能失败的地方"
    }}
]
```

评分标准:
- compatibility_score(1-5): 组件之间的兼容性，5=天然兼容，1=严重冲突
- implementation_difficulty: 基于研究生单GPU场景的判断
- risk_factors: 技术层面可能导致失败的具体原因
- mitigation: 针对主要风险的缓解措施
- quick_start: 最小可运行验证方案，越具体越好"""

# ── Trend Forecasting Agent ──
TIMELINE_ANALYSIS = """你是一个科研趋势分析师。请分析以下论文的发表时间分布和主题演化趋势。

研究方向: {query}

论文列表（按年份排列）:
{papers_timeline}

请从以下维度分析时间线趋势，输出JSON格式:

```json
{{
    "year_distribution": {{
        "2022": {{"count": 5, "keywords": ["keyword1", "keyword2"]}},
        "2023": {{"count": 8, "keywords": ["keyword1", "keyword2"]}},
        "2024": {{"count": 12, "keywords": ["keyword1", "keyword2"]}},
        "2025": {{"count": 6, "keywords": ["keyword1", "keyword2"]}}
    }},
    "emerging_topics": [
        {{
            "topic": "主题名称",
            "first_appeared": 2024,
            "growth_rate": "快速增长/稳步增长/萌芽期",
            "representative_papers": ["论文1缩写", "论文2缩写"],
            "description": "1句话说明这个主题在做什么"
        }}
    ],
    "declining_topics": [
        {{
            "topic": "主题名称",
            "peak_year": 2022,
            "trend": "下降/停滞",
            "reason": "可能被XX替代/问题已基本解决/..."
        }}
    ],
    "steady_topics": [
        {{
            "topic": "主题名称",
            "description": "1句话",
            "maturity": "成熟/饱和/仍有空间"
        }}
    ]
}}
```

要求:
- year_distribution: 每年的论文数量和该年最频繁出现的关键词（2-3个）
- emerging_topics: 近1-2年新出现且论文数量增长的主题（2-3个）
- declining_topics: 曾经热门但近期论文减少的主题（1-2个）
- steady_topics: 持续有论文产出但无显著增长的主题（1-2个）
- 所有分析必须基于给定的论文数据，不要编造"""

METHOD_EVOLUTION = """你是一个科研方法演化追踪专家。请分析以下论文中方法的技术路线演变，识别范式转换点。

研究方向: {query}

论文时间线分析:
{timeline_result}

论文方法解构:
{decomposition}

请追踪方法演化路径，输出JSON格式:

```json
{{
    "evolution_paths": [
        {{
            "path_name": "路径名称，如：从知识蒸馏到原型学习",
            "stages": [
                {{
                    "stage": "阶段1",
                    "time_range": "2020-2022",
                    "core_method": "知识蒸馏",
                    "key_limitation": "需要存储旧数据",
                    "representative": "论文缩写"
                }},
                {{
                    "stage": "阶段2",
                    "time_range": "2022-2024",
                    "core_method": "原型/原型网络",
                    "key_improvement": "无需旧数据，用原型代替",
                    "representative": "论文缩写"
                }}
            ],
            "paradigm_shift": true,
            "shift_reason": "从需要旧数据到无需旧数据的范式转变"
        }}
    ],
    "current_paradigm": {{
        "name": "当前主流范式名称",
        "dominant_methods": ["方法1", "方法2"],
        "limitations": ["局限1", "局限2"],
        "saturation_signals": ["信号1: 具体描述"]
    }},
    "next_paradigm_hints": [
        {{
            "hint": "可能的下一个范式方向",
            "evidence": "已有XX论文开始尝试YY，暗示...",
            "readiness": "早期/中期/临近"
        }}
    ]
}}
```

要求:
- evolution_paths: 追踪1-2条主要技术路线的演变路径，每条路径至少2个阶段
- current_paradigm: 当前主流范式的名称、方法、局限和饱和信号
- next_paradigm_hints: 基于已有论文线索预测下一个可能的范式方向
- 如果没有明确的范式转换，current_paradigm的limitations和saturation_signals要写清楚
- readiness判断：早期=仅有1-2篇探索性论文，中期=有初步验证但未成主流，临近=多篇论文朝同一方向"""

TREND_FORECAST = """你是一个科研投资顾问。请基于时间线分析和方法演化追踪，预测该研究方向的趋势。

研究方向: {query}

时间线分析:
{timeline_result}

方法演化追踪:
{evolution_result}

Gap分析:
{gaps}

请给出综合趋势预测，输出JSON格式:

```json
{{
    "direction_score": {{
        "heat": 4,
        "saturation": 2,
        "potential": 5,
        "entry_barrier": 3
    }},
    "overall_phase": "上升期/平台期/饱和期/衰退期",
    "phase_reasoning": "3-4句话，为什么判断是这个阶段",
    "forecast": [
        {{
            "time_horizon": "6个月",
            "prediction": "具体预测：什么方法会变多/什么问题会被关注",
            "confidence": 0.7
        }},
        {{
            "time_horizon": "1-2年",
            "prediction": "具体预测：范式如何变化/新方向如何发展",
            "confidence": 0.5
        }},
        {{
            "time_horizon": "3-5年",
            "prediction": "具体预测：该领域可能如何演进",
            "confidence": 0.3
        }}
    ],
    "investment_advice": {{
        "for_beginner": "给新入方向的研究生的建议（1-2句话）",
        "for_advanced": "给已有基础的researcher的建议（1-2句话）",
        "low_hanging_fruit": "最容易出成果的切入点",
        "high_risk_high_reward": "高风险高回报的方向"
    }},
    "red_flags": [
        "危险信号1: 具体描述为什么这个方向可能不值得做",
    ],
    "green_flags": [
        "积极信号1: 具体描述为什么这个方向值得投入",
    ]
}}
```

评分标准(1-5):
- heat: 当前研究热度（1=冷门, 5=顶会高频）
- saturation: 饱和度（1=大量空间, 5=已拥挤）
- potential: 潜力（1=天花板可见, 5=远未到顶）
- entry_barrier: 入门门槛（1=容易上手, 5=需大量前置知识/算力）

要求:
- forecast的confidence要诚实：短期预测置信度可稍高，长期预测要低
- investment_advice要具体可操作，不要"需要更多研究"这种废话
- red_flags至少1条，green_flags至少2条
- phase_reasoning要有数据支撑（引用时间线分析中的年份分布和趋势）"""
