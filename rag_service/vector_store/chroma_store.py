from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag_service.config import settings
from rag_service.vector_store.base import ChunkRecord, QueryResult, VectorStore

COLLECTION_NAME = "rag_collection_v2"
EMBEDDING_DIM = 256


class ChromaStore(VectorStore):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embedding_dim": EMBEDDING_DIM},
        )

    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        if not chunks:
            return

        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        for ch in chunks:
            if not ch.embedding:
                raise ValueError(f"Chunk {ch.chunk_id} has no embedding")

            if EMBEDDING_DIM and len(ch.embedding) != EMBEDDING_DIM:
                raise ValueError(
                    f"Chunk {ch.chunk_id} has invalid embedding dim: "
                    f"{len(ch.embedding)} != {EMBEDDING_DIM}"
                )

            ids.append(ch.chunk_id)
            documents.append(ch.text)
            embeddings.append(ch.embedding)

            meta: Dict[str, Any] = {
                "document_id": ch.document_id,
                "collection": ch.collection,
                "chunk_index": ch.chunk_index,
                "chunk_total": ch.chunk_total,
                "document_hash": ch.document_hash,
                "section": ch.section,
                "doc_type": ch.doc_type,
                "source": ch.source,
            }

            # Дополнительные метаданные пользователя/пайплайна
            if ch.metadata:
                meta.update(ch.metadata)

            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        embedding: Sequence[float],
        collection: str,
        top_k: int,
        where: Dict[str, Any] | None = None,
    ) -> List[QueryResult]:
        if EMBEDDING_DIM and len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Query embedding has invalid dim: {len(embedding)} != {EMBEDDING_DIM}"
            )

        base_where: Dict[str, Any] = {"collection": collection}
        if where:
            where_filter: Dict[str, Any] = {"$and": [base_where, where]}
        else:
            where_filter = base_where

        result = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where=where_filter,
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: List[QueryResult] = []
        for i, chunk_id in enumerate(ids):
            meta = metas[i] or {}
            hits.append(
                QueryResult(
                    chunk_id=str(chunk_id),
                    document_id=meta.get("document_id", ""),
                    text=docs[i] if i < len(docs) else "",
                    distance=float(distances[i]) if i < len(distances) else 0.0,
                    metadata=meta,
                )
            )

        return hits

    def delete_by_document(self, document_id: str) -> int:
        existing_ids = self.get_document_chunk_ids(document_id)
        if not existing_ids:
            return 0

        self.collection.delete(ids=existing_ids)
        return len(existing_ids)

    # Совместимость с текущим app.py
    def delete_document(self, document_id: str) -> None:
        self.delete_by_document(document_id)

    def get_document_chunk_ids(self, document_id: str) -> List[str]:
        rows = self.collection.get(
            where={"document_id": document_id},
        )
        ids = rows.get("ids", [])
        return [str(x) for x in ids]

    def has_document(self, document_id: str, document_hash: str) -> bool:
        rows = self.collection.get(
            where={
                "$and": [
                    {"document_id": document_id},
                    {"document_hash": document_hash},
                ]
            },
            limit=1,
        )
        ids = rows.get("ids", [])
        return bool(ids)

    def list_documents(self) -> List[Dict[str, Any]]:
        rows = self.collection.get(include=["metadatas"])
        ids = rows.get("ids", [])
        metadatas = rows.get("metadatas", [])

        grouped: Dict[str, Dict[str, Any]] = {}

        for i, _chunk_id in enumerate(ids):
            meta = metadatas[i] or {}
            document_id = meta.get("document_id", "")
            if not document_id:
                continue

            if document_id not in grouped:
                grouped[document_id] = {
                    "collection": meta.get("collection", ""),
                    "file_name": meta.get("source", ""),
                    "document_id": document_id,
                    "chunks_count": 0,
                }

            grouped[document_id]["chunks_count"] += 1

        return sorted(
            grouped.values(),
            key=lambda x: (x["collection"], x["file_name"], x["document_id"]),
        )

    # Совместимость с текущим app.py / admin.html
    def get_documents_overview(self) -> List[Dict[str, Any]]:
        return self.list_documents()

    def stats(self) -> Dict[str, Any]:
        rows = self.collection.get(include=["metadatas"])
        ids = rows.get("ids", [])
        metadatas = rows.get("metadatas", [])

        document_ids = set()
        for meta in metadatas:
            meta = meta or {}
            if meta.get("document_id"):
                document_ids.add(meta["document_id"])

        return {
            "db_documents": len(document_ids),
            "db_chunks": len(ids),
            "db_path": str(self.path),
            "collection_name": COLLECTION_NAME,
            "embedding_dim": EMBEDDING_DIM,
        }


_store_instance: Optional[ChromaStore] = None


def get_vector_store() -> ChromaStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaStore(path=settings.chroma_persist_dir)
    return _store_instance