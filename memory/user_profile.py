"""用户偏好管理 — 跨会话持久化"""
import os
import json
from typing import Optional
from datetime import datetime

import config


class UserProfile:
    """用户偏好：研究方向、关注会议、历史查询等。"""

    def __init__(self):
        self._path = config.USER_PROFILE_PATH
        self._profile: dict = {
            "research_interests": [],    # 研究方向关键词
            "preferred_venues": [],      # 偏好会议/期刊
            "query_history": [],         # 历史查询（最近20条）
            "created_at": datetime.now().isoformat(),
        }
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._profile = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._profile, f, ensure_ascii=False, indent=2)

    def update_from_query(self, query: str):
        """从用户查询中提取并更新偏好。"""
        # 记录查询历史
        self._profile["query_history"].append({
            "query": query,
            "time": datetime.now().isoformat(),
        })
        # 只保留最近20条
        self._profile["query_history"] = self._profile["query_history"][-20:]
        self._save()

    def add_interest(self, keyword: str):
        """添加研究方向关键词。"""
        if keyword not in self._profile["research_interests"]:
            self._profile["research_interests"].append(keyword)
            self._save()

    def add_venue(self, venue: str):
        """添加偏好会议。"""
        if venue not in self._profile["preferred_venues"]:
            self._profile["preferred_venues"].append(venue)
            self._save()

    def get_preferences_str(self) -> str:
        """获取偏好描述字符串，用于 Prompt 注入。"""
        interests = ", ".join(self._profile.get("research_interests", []))
        venues = ", ".join(self._profile.get("preferred_venues", []))
        parts = []
        if interests:
            parts.append(f"研究方向: {interests}")
        if venues:
            parts.append(f"偏好会议: {venues}")
        return " | ".join(parts) if parts else "无特定偏好"

    @property
    def interests(self) -> list[str]:
        return self._profile.get("research_interests", [])

    @property
    def venues(self) -> list[str]:
        return self._profile.get("preferred_venues", [])
