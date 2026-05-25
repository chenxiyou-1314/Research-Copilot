"""论文元数据持久化存储 — 支持增量更新"""
import os
import json
from typing import Optional
from datetime import datetime

import config


class PaperStore:
    """论文库管理，JSON 持久化，增量添加。"""

    def __init__(self):
        self._path = config.PAPER_STORE_PATH
        self._papers: dict[str, dict] = {}  # paper_id -> paper_meta
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._papers = data.get("papers", {})
                    # 迁移: 兼容旧格式
                    if isinstance(self._papers, list):
                        self._papers = {p["paper_id"]: p for p in self._papers}
                print(f"[PaperStore] 已加载 {len(self._papers)} 篇论文")
            except (json.JSONDecodeError, KeyError):
                self._papers = {}

    def add_paper(self, paper: dict):
        """添加论文，已存在则更新。"""
        paper_id = paper.get("paper_id", "")
        if paper_id and paper_id not in self._papers:
            paper["added_at"] = datetime.now().isoformat()
        self._papers[paper_id] = paper

    def get_paper(self, paper_id: str) -> Optional[dict]:
        return self._papers.get(paper_id)

    def is_indexed(self, paper_id: str) -> bool:
        """论文是否已存在于库中。"""
        return paper_id in self._papers

    def search_by_keyword(self, keyword: str) -> list[dict]:
        """按关键词在标题/摘要中搜索。"""
        keyword = keyword.lower()
        return [
            p for p in self._papers.values()
            if keyword in p.get("title", "").lower()
            or keyword in p.get("abstract", "").lower()
        ]

    def get_recent(self, n: int = 10) -> list[dict]:
        """获取最近添加的 n 篇论文。"""
        sorted_papers = sorted(
            self._papers.values(),
            key=lambda p: p.get("added_at", ""),
            reverse=True,
        )
        return sorted_papers[:n]

    def count(self) -> int:
        return len(self._papers)

    def save(self):
        """持久化到磁盘。"""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"papers": self._papers, "updated_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)

    def get_all_papers(self) -> list[dict]:
        return list(self._papers.values())
