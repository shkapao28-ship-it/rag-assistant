# rag_service/retrieval/cache.py

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def normalize_query(text: str) -> str:
    """
    Простая нормализация запроса:
    - lower;
    - трим пробелы;
    - сжать повторные пробелы;
    - заменить ё на е.
    """
    if not text:
        return ""
    text = text.strip().lower()
    text = text.replace("ё", "е")
    parts = text.split()
    return " ".join(parts)


@dataclass(frozen=True)
class CacheKey:
    collection: str
    mode: str
    query_norm: str


@dataclass
class CacheEntry:
    answer: str
    chunks: list[dict[str, Any]]
    created_at: float


class AnswerCache:
    """
    Простейший in-memory кэш ответов:
    - key = (collection, mode, normalized_query)
    - value = (answer, chunks, timestamp)
    - TTL по времени жизни.
    """

    def __init__(self, ttl_seconds: int = 600, max_size: int = 1000) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: Dict[CacheKey, CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> Optional[CacheEntry]:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            if now - entry.created_at > self._ttl:
                # протухло
                del self._store[key]
                return None
            return entry

    def set(self, key: CacheKey, answer: str, chunks: list[dict[str, Any]]) -> None:
        now = time.time()
        with self._lock:
            if len(self._store) >= self._max_size:
                # простейшая стратегия: очистить весь кэш
                self._store.clear()
            self._store[key] = CacheEntry(answer=answer, chunks=chunks, created_at=now)


# Глобальный экземпляр кэша для простоты
answer_cache = AnswerCache(ttl_seconds=600, max_size=1000)