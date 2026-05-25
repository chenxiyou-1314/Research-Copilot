# 陈灿域

意向岗位：Agent开发实习生

共青团团员 ｜ 13829572603 ｜ chencanyu@mail.gdut.edu.cn

24岁 | 广东广州 | GitHub : https://github.com/chenxiyou-1314

---

## 教育背景

**广东工业大学** | 计算机技术-硕士 | 2024.09-至今

成绩：4.01/5.0（专业前3%），学业一等奖学金

研究方向：大模型应用、多模态技术、深度学习落地

学术成果：老师一作本人二作CCF B类会议 ICME 论文1篇，本人三作CCF B类会议 ICMR 论文1篇；投稿CCF A类会议 ACMMM；申请国家发明专利2项（导师一作，本人二作）；获全国研究生数学建模竞赛三等奖（2024, 2025）

**广东技术师范大学** | 软件工程-本科 | 2020.09-2024.06

成绩：3.96/5（专业前3%），两年"校级三好学生"及校级奖学金

主修课程：计算机系统结构、云计算与大数据技术、软件工程、高级数据库等

---

## 实习经历

**广州格物知新人工智能科技有限公司** | Agent开发工程师 | 2026.01-至今

*生成小组*

核心职责：参与大模型Agent系统研发，负责Agent架构设计、Tool Use机制、RAG模块、工程化落地及知识产权沉淀，支撑视频/图片/PPT全链路AIGC服务稳定迭代。

**Agent架构设计**：基于LangChain搭建多模态内容生成Agent，采用ReAct范式实现意图识别→工具选择→多步骤执行→结果聚合的端到端自动化；设计意图理解与动态路由模块（分类器+LLM-based routing），准确率达95%+；

**Tool Use / Function Calling**：实现Function Calling机制，对接PPT/视频/播客等6类生成工具，Agent根据规划自动选择调用；设计多工具编排策略，支持并行调用与结果聚合；

**RAG与记忆系统**：实现多轮对话记忆系统（长短期记忆+向量检索），支持跨轮次上下文延续与意图修正；基于用户画像的个性化内容生成策略，支持脚本风格、配音等差异化输出；

**工程化落地**：核心接口Docker化部署，SSE流式输出+Redis消息队列高可用架构；重构视频生成流水线，FFmpeg加速+并行合成提升吞吐量；vLLM推理优化：单样本延迟从2.3s降至0.8s；

**大模型微调与对齐**：构建AlignXplore-Plus微调与评测流程，78例测试用例准确率90.2%，标签泄露率从40%降至0%；

核心成果：LangChain Agent实现意图识别→工作流→多步骤执行端到端自动化 | 多模态全链路统一调度 | vLLM推理2.3s→0.8s | 标签泄露40%→0%

---

## 项目经历

**Research Copilot — 科研文献智能检索与综述生成 Agent** | LangGraph + RAG + FastAPI | 2026.05-至今

项目地址：https://github.com/chenxiyou-1314/Research-Copilot

项目描述：基于LangGraph构建科研文献检索与综述生成的多步推理Agent，实现从关键词输入到结构化文献综述输出的端到端自动化。支持意图识别→论文检索→智能筛选→PDF解析向量化→RAG增强综述生成→记忆更新的完整工作流。

个人职责：

- 设计LangGraph多节点DAG工作流，实现意图识别动态路由（检索/QA/综述/追问4种意图），根据意图选择不同执行路径；实现条件边与节点间的状态传递；

- 实现Tool Use机制，封装arXiv API、Semantic Scholar API、PDF解析器、FAISS向量检索等6类工具，Agent根据规划自动选择调用；

- 设计RAG模块：论文PDF切分→Embedding→FAISS索引→跨论文语义检索，支持基于检索上下文的增强综述生成与跨论文问答；

- 实现增量索引机制：新增论文仅对增量部分向量化入索引库，避免全量重算（借鉴CIL思路），论文库从0到50篇索引耗时保持线性增长；

- 实现跨会话记忆系统：持久化已读论文库与用户偏好（研究方向/关注会议），新会话自动加载历史上下文；

- 工程化：FastAPI + SSE流式输出、Docker化部署、Redis消息队列；

项目成果：实现意图识别→多步工具调用→综述生成端到端自动化 | 增量索引避免全量重算 | 跨会话记忆支持连续交互

---

## 个人技能

编程语言：Python，Java，Shell脚本

训练/推理框架：Llama-Factory、Ollama、DeepSpeed、vLLM等

大模型相关：Transformer、GPT、GLM、LLaMA、Qwen等

Agent技术：LangChain、LangGraph、RAG、Tool Use / Function Calling、多智能体、提示工程、ReAct

算法方法：SFT、LoRA、PPO、DPO、GRPO

外语能力：CET-6，具备英文论文写作、撰写技术文档及国际期刊/会议投稿

证书/执照：中级软件设计师，机动车驾驶证C1

---

## 科研成果

[1] Zhouwei Wang, Canyu Chen, Xingming Liao, Hong Li, Nankai Lin, and Xin Chen, "TRIDENT: TEXT-GUIDED REFINEMENT FOR ID AND OOD ENTITIES" ICME 2024(CCF B) 老师一作本人二作

[2] Zeng, M., Liao, X., Chen, C., Lin, N., Wang, Z., Chen, C., & Yang, A. (2024). Chameleon: On the scene diversity and domain variety of ai-generated videos detection. ICMR 2024(CCF B) 第三作者

[3] 王卓薇,陈灿域等，基于 X 射线蒸馏和对象完整框架的多源海洋智能侦察方法.申请号：202510950121.0（导师一作本人二作）

[4] 王卓薇,陈灿域等，一种海上智能侦察仪多尺度小波变换的动态序列推荐机制.申请号：202511279926.3（导师一作本人二作）
