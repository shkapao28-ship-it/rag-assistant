from __future__ import annotations

from typing import Any, Dict, List, Literal

from rag_service.config import settings
from rag_service.retrieval.rerank import rerank_and_dedup
from rag_service.vector_store.base import QueryResult
from rag_service.vector_store.chroma_store import get_vector_store
from rag_service.yandex_client import YandexEmbeddingProvider, YandexLLMProvider

Mode = Literal["precise", "balanced", "recall"]


def _decide_fetch_k(mode: Mode, top_k: int) -> int:
    if mode == "precise":
        return max(top_k * 2, 8)
    if mode == "recall":
        return max(top_k * 3, 18)
    return max(top_k * 2, 12)


def _looks_multi_part(query: str) -> bool:
    q = query.lower()
    separators = [" и ", " или ", ",", ";", "?", " а также ", " при этом "]
    hits = sum(1 for sep in separators if sep in q)
    return hits >= 2 or ("сколько" in q and "можно ли" in q)


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        distance = ch.get("distance", 0.0)
        semantic_score = ch.get("semantic_score", 0.0)
        keyword_score = ch.get("keyword_score", 0.0)
        final_score = ch.get("final_score", 0.0)

        meta = ch.get("metadata", {}) or {}
        section = meta.get("section") or "без секции"
        source = meta.get("source") or ch.get("document_id", "")
        chunk_index = meta.get("chunk_index", "?")

        parts.append(
            f"[Фрагмент {i} | doc={ch.get('document_id', '')} | "
            f"chunk_index={chunk_index} | section={section} | source={source} | "
            f"distance={distance:.4f} | semantic={semantic_score:.4f} | "
            f"keyword={keyword_score:.4f} | final={final_score:.4f}]\n"
            f"{ch.get('text', '')}\n"
        )
    return "\n".join(parts)


def _fallback_answer_from_chunks(query: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "В найденных фрагментах нет достаточной информации, чтобы ответить на этот вопрос."

    snippets: List[str] = []
    for ch in chunks[:3]:
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        short = text[:500].strip()
        snippets.append(short)

    if not snippets:
        return "В найденных фрагментах нет достаточной информации, чтобы ответить на этот вопрос."

    if _looks_multi_part(query):
        lines = ["По найденным фрагментам можно подтвердить следующее:"]
        for s in snippets:
            lines.append(f"- {s}")
        return "\n".join(lines)

    return snippets[0]


def _needs_fallback(answer: str) -> bool:
    if not answer:
        return True

    normalized = answer.strip().lower()
    if not normalized:
        return True

    too_generic_markers = [
        "я не могу ответить",
        "недостаточно информации",
        "в найденных фрагментах нет достаточной информации",
        "не удалось найти",
    ]

    if len(normalized) < 20:
        return True

    if any(marker in normalized for marker in too_generic_markers):
        return True

    return False


def ask(
    query: str,
    collection: str,
    mode: Mode = "balanced",
    top_k: int | None = None,
) -> Dict[str, Any]:
    if top_k is None:
        top_k = settings.default_top_k

    store = get_vector_store()
    embedder = YandexEmbeddingProvider()
    llm = YandexLLMProvider()

    query_vector = embedder.embed_text(query)
    fetch_k = _decide_fetch_k(mode, top_k)

    raw_results: List[QueryResult] = store.query(
        embedding=query_vector,
        collection=collection,
        top_k=fetch_k,
    )

    raw_found = len(raw_results)

    if raw_found == 0:
        return {
            "answer": "В найденных фрагментах нет информации, чтобы ответить на этот вопрос.",
            "chunks": [],
            "shown_chunks_count": 0,
            "context_chunks_preview": [],
            "mode": mode,
            "top_k_used": 0,
            "fetch_k_used": fetch_k,
            "raw_found": 0,
            "reranked_found": 0,
            "shown_chunks": 0,
            "context_chunks_used": 0,
            "latency_ms": 0,
            "debug": {
                "query": query,
                "collection": collection,
                "mode": mode,
                "fetch_k": fetch_k,
                "top_k_requested": top_k,
                "raw_found": 0,
                "reranked_found": 0,
                "shown_chunks": 0,
                "context_limit": settings.max_context_chunks,
                "context_chunks_used": 0,
                "fallback_used": False,
            },
        }

    shown_limit = max(1, top_k)
    context_limit = max(1, min(top_k, settings.max_context_chunks))

    reranked_all = rerank_and_dedup(
        query=query,
        candidates=raw_results,
        max_results=shown_limit,
    )

    reranked_found = len(reranked_all)

    if reranked_found == 0:
        return {
            "answer": "В найденных фрагментах нет достаточной информации, чтобы ответить на этот вопрос.",
            "chunks": [],
            "shown_chunks_count": 0,
            "context_chunks_preview": [],
            "mode": mode,
            "top_k_used": 0,
            "fetch_k_used": fetch_k,
            "raw_found": raw_found,
            "reranked_found": 0,
            "shown_chunks": 0,
            "context_chunks_used": 0,
            "latency_ms": 0,
            "debug": {
                "query": query,
                "collection": collection,
                "mode": mode,
                "fetch_k": fetch_k,
                "top_k_requested": top_k,
                "raw_found": raw_found,
                "reranked_found": 0,
                "shown_chunks": 0,
                "context_limit": context_limit,
                "context_chunks_used": 0,
                "fallback_used": False,
            },
        }

    shown_chunks_raw = reranked_all[:shown_limit]
    context_chunks_raw = reranked_all[:context_limit]

    context_text = _format_context(context_chunks_raw)
    multi_part = _looks_multi_part(query)

    if multi_part:
        answer_style = (
            "Если вопрос состоит из нескольких частей, ответь по пунктам.\n"
            "Каждый пункт включай только если он подтверждается найденными фрагментами.\n"
            "Если подтверждена только часть вопроса, ответь только на подтвержденную часть.\n"
        )
    else:
        answer_style = (
            "Ответь кратко, прямо и по существу, без лишних общих фраз.\n"
        )

    system_prompt = (
        "Ты ассистент, отвечающий строго по предоставленным фрагментам базы знаний.\n"
        "Правила:\n"
        "1) Отвечай только на основе фрагментов.\n"
        "2) Не придумывай факты, цены, условия, сроки или ограничения, которых нет в тексте.\n"
        "3) Если ответ явно есть во фрагментах, дай прямой и точный ответ.\n"
        "4) Если вопрос задан другими словами, используй близкие по смыслу фрагменты.\n"
        "5) Если часть ответа есть, а части не хватает, скажи только подтверждённую часть.\n"
        "6) Только если по фрагментам совсем нельзя ответить, скажи: "
        "«В найденных фрагментах нет достаточной информации, чтобы ответить на этот вопрос».\n"
        "7) Не ссылайся на внешние знания и не добавляй ничего от себя.\n"
    )

    user_prompt = (
        f"Вопрос пользователя:\n{query}\n\n"
        f"Ниже фрагменты базы знаний:\n\n{context_text}\n\n"
        f"{answer_style}"
        "Если в найденных фрагментах есть перечисления, можешь использовать список.\n"
        "Не пересказывай весь контекст, а извлеки только ответ.\n"
        "Если есть несколько подтвержденных аспектов вопроса, отрази их все."
    )

    answer = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=500,
    ).strip()

    fallback_used = False
    if _needs_fallback(answer):
        answer = _fallback_answer_from_chunks(query, context_chunks_raw)
        fallback_used = True

    ui_chunks = [
        {
            "id": ch["chunk_id"],
            "document_id": ch["document_id"],
            "text": ch["text"],
            "score": ch["distance"],
            "meta": {
                "collection": ch.get("metadata", {}).get("collection", collection),
                "file_name": ch.get("metadata", {}).get("source", ""),
                "chunk_index": ch.get("metadata", {}).get("chunk_index", 0),
                "document_hash": ch.get("metadata", {}).get("document_hash", ""),
                "section": ch.get("metadata", {}).get("section", ""),
                "doc_type": ch.get("metadata", {}).get("doc_type", ""),
                "source": ch.get("metadata", {}).get("source", ""),
            },
            "semantic_score": ch.get("semantic_score", 0.0),
            "keyword_score": ch.get("keyword_score", 0.0),
            "final_score": ch.get("final_score", 0.0),
            "used_in_context": idx < len(context_chunks_raw),
        }
        for idx, ch in enumerate(shown_chunks_raw)
    ]

    context_preview = [
        {
            "document_id": ch["document_id"],
            "chunk_id": ch["chunk_id"],
            "section": ch.get("metadata", {}).get("section", ""),
            "chunk_index": ch.get("metadata", {}).get("chunk_index", 0),
            "distance": ch.get("distance", 0.0),
            "semantic_score": ch.get("semantic_score", 0.0),
            "keyword_score": ch.get("keyword_score", 0.0),
            "final_score": ch.get("final_score", 0.0),
        }
        for ch in context_chunks_raw
    ]

    return {
        "answer": answer,
        "chunks": ui_chunks,
        "shown_chunks_count": len(ui_chunks),
        "context_chunks_preview": context_preview,
        "mode": mode,
        "top_k_used": len(ui_chunks),
        "fetch_k_used": fetch_k,
        "raw_found": raw_found,
        "reranked_found": reranked_found,
        "shown_chunks": len(ui_chunks),
        "context_chunks_used": len(context_chunks_raw),
        "latency_ms": 0,
        "debug": {
            "query": query,
            "collection": collection,
            "mode": mode,
            "fetch_k": fetch_k,
            "top_k_requested": top_k,
            "raw_found": raw_found,
            "reranked_found": reranked_found,
            "shown_chunks": len(ui_chunks),
            "context_limit": context_limit,
            "context_chunks_used": len(context_chunks_raw),
            "fallback_used": fallback_used,
            "multi_part_detected": multi_part,
        },
    }