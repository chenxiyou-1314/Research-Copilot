# Research Copilot 🔬

基于 LangGraph 的科研文献智能检索与综述生成 Agent。

## 核心功能

- **论文检索**：输入研究方向关键词，自动从 arXiv / Semantic Scholar 检索相关论文
- **智能筛选**：根据会议级别、引用数、时间窗口过滤高价值论文
- **PDF 解析 + RAG**：下载论文 PDF → 切分 → Embedding → FAISS 向量检索，支持跨论文语义问答
- **综述生成**：自动生成结构化文献综述草稿（研究背景、方法对比、趋势分析）
- **增量更新**：新增论文不会触发全量重算，基于时间戳增量索引（借鉴 CIL 思路）
- **用户记忆**：跨会话持久化已读论文库与用户偏好

## Agent 架构

```
用户输入研究方向
    │
    ▼
[意图识别节点] ──→ 判断：检索 / 追问 / 直接回答
    │
    ▼ (检索)
[论文检索节点] ──→ arXiv API + Semantic Scholar API
    │
    ▼
[论文筛选节点] ──→ 引用数/时间/会议级别过滤
    │
    ▼
[PDF解析节点] ──→ 下载 → 切分 → Embedding → 入库
    │
    ▼
[综述生成节点] ──→ RAG检索 + LLM生成结构化综述
    │
    ▼
[记忆更新节点] ──→ 持久化论文元数据 + 用户偏好
    │
    ▼
输出结果（SSE流式）
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY 或其他 LLM API Key

# 启动服务
python main.py

# 或 Docker 启动
docker-compose up -d
```

## API 接口

### 研究综述生成（SSE 流式）

```bash
curl -N http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "class-incremental learning for object detection", "max_papers": 10}'
```

### 论文检索

```bash
curl http://localhost:8000/papers/search \
  -H "Content-Type: application/json" \
  -d '{"query": "continual learning OOD detection", "max_results": 5}'
```

### 跨论文问答（RAG）

```bash
curl http://localhost:8000/papers/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "当前CIL方法在细粒度场景下的主要挑战是什么？"}'
```

## 技术栈

| 模块 | 技术 |
|------|------|
| Agent 框架 | LangChain + LangGraph |
| LLM | OpenAI GPT-4o-mini / 本地 Ollama |
| 向量检索 | FAISS |
| Embedding | text-embedding-3-small / BGE-M3 |
| PDF 解析 | pdfminer.six + Unstructured |
| 论文源 | arXiv API + Semantic Scholar API |
| 后端 | FastAPI + SSE |
| 部署 | Docker + Redis |
