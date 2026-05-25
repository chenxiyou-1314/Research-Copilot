"""Research Copilot 配置"""
import os
from dotenv import load_dotenv

load_dotenv()


# ── LLM 配置 ──
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai / ollama
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# ── Embedding 配置 ──
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai / local
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")

# ── 论文检索配置 ──
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "10"))
SCHOLAR_MAX_RESULTS = int(os.getenv("SCHOLAR_MAX_RESULTS", "10"))
PAPER_MIN_CITATIONS = int(os.getenv("PAPER_MIN_CITATIONS", "5"))
PAPER_YEAR_FROM = int(os.getenv("PAPER_YEAR_FROM", "2022"))

# ── 向量库配置 ──
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
PAPER_STORE_PATH = os.getenv("PAPER_STORE_PATH", "./data/papers.json")
USER_PROFILE_PATH = os.getenv("USER_PROFILE_PATH", "./data/user_profile.json")

# ── PDF 配置 ──
PDF_DOWNLOAD_DIR = os.getenv("PDF_DOWNLOAD_DIR", "./data/pdfs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── 服务配置 ──
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── 数据目录初始化 ──
for d in [PDF_DOWNLOAD_DIR, os.path.dirname(FAISS_INDEX_PATH) or "./data"]:
    os.makedirs(d, exist_ok=True)
