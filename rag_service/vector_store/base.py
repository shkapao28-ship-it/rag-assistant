from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence


@dataclass
class ChunkRecord:
    """Единица хранения в векторной БД (один чанк документа)."""

    chunk_id: str
    document_id: str
    document_hash: str
    text: str
    chunk_index: int
    chunk_total: int
    section: str | None
    doc_type: str
    collection: str
    source: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Результат поиска по векторной БД (один найденный чанк)."""

    chunk_id: str
    document_id: str
    text: str
    distance: float
    metadata: Dict[str, Any]


class VectorStore(Protocol):
    """Интерфейс для векторного хранилища (Chroma / Supabase / др.)."""

    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        ...

    def query(
        self,
        embedding: Sequence[float],
        collection: str,
        top_k: int,
        where: Dict[str, Any] | None = None,
    ) -> List[QueryResult]:
        ...

    def delete_by_document(self, document_id: str) -> int:
        ...

    def get_document_chunk_ids(self, document_id: str) -> List[str]:
        ...

    def has_document(self, document_id: str, document_hash: str) -> bool:
        ...

    def stats(self) -> Dict[str, Any]:
        ...

    def list_documents(self) -> List[Dict[str, Any]]:
        ...