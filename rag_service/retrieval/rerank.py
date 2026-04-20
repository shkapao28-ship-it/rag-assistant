# rag_service/retrieval/rerank.py

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..vector_store.base import QueryResult


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def keyword_overlap_score(query: str, text: str) -> float:
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0
    common = q_tokens & t_tokens
    return len(common) / len(q_tokens)


def similarity_ratio(a: str, b: str) -> float:
    """
    Простая Jaccard-метрика по токенам, для дедупликации похожих чанков.
    """
    a_tokens = set(_tokenize(a))
    b_tokens = set(_tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(inter) / len(union)


def rerank_and_dedup(
    query: str,
    candidates: List[QueryResult],
    max_results: int,
    dedup_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Лёгкий гибридный rerank:
    - semantic_score (по distance),
    - keyword_overlap,
    - без LLM;
    + dedup по текстовой схожести.
    Возвращает список словарей с полями: text, meta, distance, scores.
    """
    if not candidates:
        return []

    scored: List[Dict[str, Any]] = []
    for cand in candidates:
        semantic_score = 1.0 / (1.0 + cand.distance)  # нормировка distance -> [0..1]
        kw_score = keyword_overlap_score(query, cand.text)
        # пока без facet_bonus — можно добавить позже
        final_score = 0.75 * semantic_score + 0.25 * kw_score

        scored.append(
            {
                "chunk_id": cand.chunk_id,
                "document_id": cand.document_id,
                "text": cand.text,
                "metadata": cand.metadata,
                "distance": cand.distance,
                "semantic_score": semantic_score,
                "keyword_score": kw_score,
                "facet_bonus": 0.0,
                "final_score": final_score,
            }
        )

    # сортируем по финальному скору
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # dedup по тексту
    selected: List[Dict[str, Any]] = []
    for candidate in scored:
        text = candidate["text"]
        is_duplicate = False
        for prev in selected:
            sim = similarity_ratio(text, prev["text"])
            if sim >= dedup_threshold:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= max_results:
            break

    return selected