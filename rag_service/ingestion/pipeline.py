# rag_service/ingestion/pipeline.py

from __future__ import annotations

import shutil
import traceback
from pathlib import Path
from typing import List

from ..config import settings
from ..vector_store.chroma_store import get_vector_store
from ..yandex_client import YandexEmbeddingProvider
from .chunking import compute_document_hash, make_chunks_for_file
from .readers import SUPPORTED_EXTENSIONS, read_file


def list_input_files(collection: str | None = None) -> List[Path]:
    """
    Сканирует input/ и возвращает список файлов для индексации.
    Если указана collection, то ищем только в input/<collection>/.
    """
    base = settings.input_dir
    files: List[Path] = []

    if collection:
        target = base / collection
        if not target.exists():
            return []

        for p in target.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)

        return files

    if not base.exists():
        return []

    for sub in base.iterdir():
        if sub.is_dir():
            for p in sub.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(p)
        elif sub.is_file() and sub.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(sub)

    return files


def infer_collection_from_path(path: Path) -> str:
    """
    Если файл лежит в input/<collection>/..., считаем collection=<collection>,
    иначе используем "other".
    """
    base = settings.input_dir.resolve()

    try:
        rel = path.resolve().relative_to(base)
    except ValueError:
        return "other"

    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]

    return "other"


def make_document_id(path: Path, collection: str) -> str:
    return f"{collection}:{path.name}"


def run_ingestion(collection: str | None = None) -> None:
    """
    Основной ingestion pipeline:
    - находит файлы;
    - читает документ;
    - считает hash;
    - chunking;
    - embeddings;
    - upsert в vector store;
    - перенос в processed/ или failed/.
    """
    files = list_input_files(collection)
    if not files:
        print("[INGEST] No files found for ingestion.")
        return

    print(f"[INGEST] Found {len(files)} file(s) to ingest.")

    store = get_vector_store()
    embedder = YandexEmbeddingProvider()

    for path in files:
        coll = infer_collection_from_path(path)
        doc_type = coll
        document_id = make_document_id(path, coll)
        source_path = str(path.resolve())

        print(f"[INGEST] Processing file: {path} (collection={coll})")

        try:
            content_bytes = path.read_bytes()
            document_hash = compute_document_hash(content_bytes)

            if store.has_document(document_id=document_id, document_hash=document_hash):
                print(f"[INGEST] Skipping unchanged document: {path}")
                _move_to_processed(path, coll)
                continue

            text = read_file(path)
            if not text or not text.strip():
                print(f"[INGEST] Empty text after reading, skipping: {path}")
                _move_to_failed(path, coll)
                continue

            chunk_records = make_chunks_for_file(
                file_path=path,
                text=text,
                collection=coll,
                doc_type=doc_type,
                document_id=document_id,
                document_hash=document_hash,
            )

            if not chunk_records:
                print(f"[INGEST] No chunks produced, skipping: {path}")
                _move_to_failed(path, coll)
                continue

            for chunk in chunk_records:
                if getattr(chunk, "metadata", None) is None:
                    chunk.metadata = {}
                chunk.metadata["source_path"] = source_path
                chunk.metadata["collection"] = coll
                chunk.metadata["document_id"] = document_id
                chunk.metadata["document_hash"] = document_hash
                chunk.metadata["file_name"] = path.name

            texts = [chunk.text for chunk in chunk_records]
            embeddings = embedder.embed_texts(texts)

            if len(embeddings) != len(chunk_records):
                raise RuntimeError(
                    f"Embedding count mismatch: chunks={len(chunk_records)}, embeddings={len(embeddings)}"
                )

            for chunk, embedding in zip(chunk_records, embeddings):
                chunk.embedding = embedding

            store.upsert_chunks(chunk_records)
            print(f"[INGEST] Ingested {len(chunk_records)} chunks for {path.name}")

            _move_to_processed(path, coll)

        except Exception as exc:
            print(f"[INGEST] Error processing {path}: {exc}")
            print(traceback.format_exc())
            _move_to_failed(path, coll)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_unique_target(dst_dir: Path, file_name: str) -> Path:
    target = dst_dir / file_name
    if not target.exists():
        return target

    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    counter = 1

    while True:
        candidate = dst_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _move_to_processed(path: Path, collection: str) -> None:
    target_dir = settings.processed_dir / collection
    _ensure_dir(target_dir)
    target = _build_unique_target(target_dir, path.name)
    shutil.move(str(path), str(target))


def _move_to_failed(path: Path, collection: str) -> None:
    target_dir = settings.failed_dir / collection
    _ensure_dir(target_dir)
    target = _build_unique_target(target_dir, path.name)
    shutil.move(str(path), str(target))