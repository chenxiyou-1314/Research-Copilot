<div align="center">

# 🔬 Research Copilot

**基于 LangGraph 的科研文献智能检索与综述生成 Agent**

从"读论文"到"发现新研究机会"——不只是文献综述工具，而是你的 AI 科研合伙人

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## ✨ 为什么做这个？

现有科研工具只做"帮你读论文"，但科研的核心问题是：**读完之后呢？**

Research Copilot 覆盖完整的科研认知链路：

```
读论文 → 理解方法 → 发现Gap → 重组新方案 → 判断方向值不值得做 → 越用越懂你
```

## 🏗️ 架构总览

11 个 Agent 节点，LangGraph 有向图编排，Self-Reflection 质量闭环：

```
                        ┌─────────┐
                        │  Intent │ 意图识别：检索 / 综述 / 问答 / 追问
                        └────┬────┘
                             │
                        ┌────▼────┐
                        │Coordinator│ 任务规划：检索策略 + 关注方向
                        └────┬────┘
                             │
                        ┌────▼────┐
                        │  Search  │ 多源检索：arXiv + Semantic Scholar
                        └────┬────┘
                             │
                        ┌────▼────┐
                        │ Analysis │ PDF解析 → 向量化 → RAG检索
                        └────┬────┘
                             │
                        ┌────▼────┐
                        │ Writing  │ 综述生成 / QA回答 / Critic修订
                        └────┬────┘
                             │
                        ┌────▼────┐     ┌──────────────────┐
                        │  Critic  │────▶│ 未通过 → 重跑 Coordinator │
                        └────┬────┘     │ (最多 2 次 Self-Refine) │
                             │ 通过      └──────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
   │   Novelty   │   │Decomposition│   │   Trend     │
   │  Gap分析    │──▶│  方法解构    │──▶│  趋势预测    │
   │  跨域迁移   │   │  跨论文重组  │   │  方向评估    │
   │  新思路验证 │   │  可行性验证  │   │  投入建议    │
   └─────────────┘   └─────────────┘   └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │  Profile    │
                                        │  知识图谱    │
                                        │  研究画像    │
                                        └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │   Memory    │
                                        │  持久化存储  │
                                        └─────────────┘
```

## 🎯 核心亮点

### 1. 多 Agent 协作架构
单一职责拆分，每个 Agent 专注一件事，降低幻觉、提升可解释性。Coordinator 编排全局，Critic 闭环质控。

### 2. Self-Reflection 质量闭环
Critic Agent 从**覆盖度 / 准确性 / 连贯性**三个维度评分，未达阈值自动反馈给 Coordinator 重新规划，最多迭代 2 次。

### 3. Novelty Agent — 研究思路发现
不只是总结已有工作，而是主动发现 Gap：
- **四维 Gap 分析**：方法论空白 / 跨域机会 / 数据集局限 / 评估缺陷
- **跨域迁移**：将其他领域的成熟方法迁移到目标领域
- **新颖性验证**：对生成的研究思路做查重式验证，标注置信度

### 4. Method Decomposition Agent — 方法重组
将论文方法拆解为**原子组件**（编码器、损失函数、训练策略等），跨论文重组新方法方案，并验证组件兼容性与实现难度。

### 5. Trend Forecasting Agent — 趋势预测
从论文时间线中提取方法演化路径，**四维评分**评估方向价值：
| 维度 | 含义 |
|------|------|
| 🔥 热度 | 当前社区关注度 |
| 📊 饱和度 | 是否已有大量工作 |
| 💡 潜力 | 未来增长空间 |
| 🚧 门槛 | 入场难度 |

给出投入建议 + 红绿旗标注。

### 6. Research Profile Graph — 越用越懂你
基于历史查询和论文库构建个人知识图谱：核心领域、已掌握方法、知识盲区、高相关未读方向、研究风格画像。

## 📁 项目结构

```
Research-Copilot/
├── agent/                         # Agent 核心
│   ├── graph.py                   # LangGraph 图编排 + 条件路由
│   ├── state.py                   # 全局状态 Schema (ResearchState)
│   ├── prompts.py                 # 所有 Prompt 模板
│   ├── coordinator.py             # Coordinator: 任务规划
│   ├── search_agent.py            # Search: 检索 + 筛选
│   ├── analysis_agent.py          # Analysis: PDF解析 + RAG
│   ├── writing_agent.py           # Writing: 综述/QA/修订
│   ├── critic_agent.py            # Critic: 质量评估 + Self-Refine
│   ├── novelty_agent.py           # Novelty: Gap分析 + 跨域迁移 + 新颖性验证
│   ├── decomposition_agent.py     # Decomposition: 方法解构 + 重组 + 可行性验证
│   ├── trend_agent.py             # Trend: 时间线 + 演化 + 四维趋势预测
│   └── profile_agent.py           # Profile: 个人知识图谱 + 研究画像
├── tools/                         # 工具层
│   ├── arxiv_search.py            # arXiv API 检索
│   ├── scholar_search.py          # Semantic Scholar API 检索
│   ├── pdf_parser.py              # PDF 下载 + 解析 + 切分
│   ├── vector_store.py            # FAISS 向量存储 + 检索
│   └── summary_writer.py          # 综述结构化输出
├── memory/                        # 记忆层
│   ├── paper_store.py             # 论文库持久化
│   └── user_profile.py            # 用户偏好 + 历史查询
├── main.py                        # FastAPI 入口 (SSE + REST)
├── config.py                      # 环境变量配置
├── evaluate.py                    # 评估框架 (意图/检索/综述/延迟)
├── requirements.txt               # 依赖
├── Dockerfile                     # Docker 部署
├── docker-compose.yml             # Docker Compose
└── .env.example                   # 环境变量模板
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- （可选）Docker

### 安装

```bash
# 克隆仓库
git clone https://github.com/chenxiyou-1314/Research-Copilot.git
cd Research-Copilot

# 创建虚拟环境
conda create -n research-copilot python=3.10 -y
conda activate research-copilot

# 安装依赖
pip install -r requirements.txt
```

### 配置

```bash
# 复制环境变量模板
cp .env.example .env
```

编辑 `.env`，填入你的 LLM API Key：

```env
# 使用 OpenAI 兼容 API（如 DeepSeek）
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat

# 或使用本地 Ollama
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=qwen2.5:7b

# Embedding（本地运行，无需 API）
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
```

> 💡 国内用户建议设置 HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`

### 启动

```bash
python main.py
```

浏览器访问 http://localhost:8000 即可使用 Web UI。

### Docker 部署

```bash
docker-compose up -d
```

## 🖥️ Web UI

内置 6 个 Tab 页，覆盖完整使用流程：

| Tab | 功能 | 说明 |
|-----|------|------|
| 综述生成 | 输入研究方向，一键生成文献综述 | SSE 流式输出 + Critic 评分 + Novelty/Decomposition/Trend/Profile 全链路展示 |
| 论文库 | 浏览已索引论文 | 支持关键词搜索，显示来源/引用/索引状态 |
| 问答 | 跨论文 RAG 问答 | 基于向量检索的科研问题回答 |
| 趋势预测 | 独立使用 Trend Agent | 四维评分（热度/饱和度/潜力/门槛）+ 时间线 + 演化路径 + 投入建议 + 红绿旗 |
| 知识图谱 | 独立使用 Profile Agent | 雷达图（领域掌握度）+ 方法掌握度进度条 + 知识盲区 + 高相关未读方向 + 研究风格画像 |
| 架构 | 查看 11-Agent 架构说明 | 每个 Agent 的职责与技术细节 |

**主题切换**：导航栏右侧 🌙/☀️ 按钮可切换暗色/亮色主题，选择会保存到 localStorage。

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/research/stream` | SSE 流式综述生成（完整 Agent 链路） |
| `POST` | `/papers/search` | 论文检索 |
| `POST` | `/papers/qa` | 跨论文 RAG 问答 |
| `GET` | `/papers/list` | 已索引论文列表 |
| `GET` | `/papers/status` | 论文库状态 |
| `POST` | `/decomposition` | Method Decomposition 独立调用 |
| `POST` | `/trend` | Trend Forecasting 独立调用 |
| `POST` | `/profile` | Profile Graph 独立调用 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/` | Web UI |

### 示例

**综述生成（SSE 流式）**

```bash
curl -N http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "class-incremental learning for object detection"}'
```

**论文检索**

```bash
curl http://localhost:8000/papers/search \
  -H "Content-Type: application/json" \
  -d '{"query": "continual learning OOD detection", "max_results": 5}'
```

**跨论文问答**

```bash
curl http://localhost:8000/papers/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "当前CIL方法在细粒度场景下的主要挑战是什么？"}'
```

**趋势预测**

```bash
curl http://localhost:8000/trend \
  -H "Content-Type: application/json" \
  -d '{"query": "vision-language model adaptation"}'
```

## 📊 评估框架

内置评估框架，量化各模块表现：

```bash
python evaluate.py                    # 运行全部评测
python evaluate.py --intent           # 意图识别准确率
python evaluate.py --search           # 检索 Precision@10
python evaluate.py --summary          # 综述质量（覆盖度/准确性/连贯性）
python evaluate.py --latency          # 端到端延迟
```

## 🛠️ 技术栈

| 模块 | 技术 |
|------|------|
| Agent 框架 | LangChain + LangGraph |
| LLM | OpenAI API / DeepSeek / Ollama |
| 向量检索 | FAISS |
| Embedding | BGE-M3 (本地) / text-embedding-3-small (API) |
| PDF 解析 | pdfminer.six + Unstructured |
| 论文源 | arXiv API + Semantic Scholar API |
| 后端 | FastAPI + SSE |
| 部署 | Docker + Docker Compose |

## 🗺️ 设计决策

| 决策 | 理由 |
|------|------|
| 多 Agent 而非单 Agent | 单一职责降低幻觉，面试能讲设计理由 |
| Critic Self-Reflection | 自动质量闭环，减少人工检查成本 |
| Novelty Agent 作为核心差异化 | 现有文献 Agent 只"读论文"，没有"发现新研究机会" |
| Method Decomposition | 贴合"Agent 做结构化推理"叙事，与 Novelty 联动 |
| Trend Forecasting 四维评分 | 从"帮你读"跳到"帮你判断值不值得做"，决策层级提升 |
| Profile Graph | 越用越懂用户，增强系统记忆与个性化 |
| DeepSeek V4-Flash | 项目场景不需要复杂推理，Flash 便宜且够用 |
| BGE-M3 本地 Embedding | 无需 API，CPU 即可运行，保护数据隐私 |

## 📄 License

MIT License

---

<div align="center">

**如果这个项目对你有帮助，欢迎 ⭐ Star**

</div>
