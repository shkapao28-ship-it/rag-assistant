from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from rag_service.config import ensure_project_dirs, settings
from rag_service.ingestion.chunking import compute_document_hash, make_chunks_for_file
from rag_service.ingestion.readers import read_file_as_text
from rag_service.retrieval.search import ask
from rag_service.vector_store.chroma_store import get_vector_store
from rag_service.yandex_client import YandexEmbeddingProvider

BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "rag_service" / "templates"),
    static_folder=str(BASE_DIR / "rag_service" / "static"),
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

ensure_project_dirs()

INPUT_DIR = settings.input_dir
PROCESSED_DIR = settings.processed_dir
FAILED_DIR = settings.failed_dir

ALLOWED_COLLECTIONS = list(settings.collections)

QUERY_LOG_FILE = settings.logs_dir / "query_log.jsonl"
ANSWER_CACHE_FILE = settings.logs_dir / "answer_cache.json"
INDEX_VERSION_FILE = settings.logs_dir / "index_version.txt"


def _ensure_runtime_files() -> None:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    if not QUERY_LOG_FILE.exists():
        QUERY_LOG_FILE.write_text("", encoding="utf-8")
    if not ANSWER_CACHE_FILE.exists():
        ANSWER_CACHE_FILE.write_text("{}", encoding="utf-8")
    if not INDEX_VERSION_FILE.exists():
        INDEX_VERSION_FILE.write_text(
            datetime.now().strftime("%Y%m%d%H%M%S"), encoding="utf-8"
        )


_ensure_runtime_files()


def get_index_version() -> str:
    if not INDEX_VERSION_FILE.exists():
        return "unknown"
    return INDEX_VERSION_FILE.read_text(encoding="utf-8").strip() or "unknown"


def bump_index_version() -> str:
    version = datetime.now().strftime("%Y%m%d%H%M%S")
    INDEX_VERSION_FILE.write_text(version, encoding="utf-8")
    return version


def load_answer_cache() -> Dict[str, Any]:
    try:
        return json.loads(ANSWER_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_answer_cache(data: Dict[str, Any]) -> None:
    ANSWER_CACHE_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_answer_cache() -> None:
    save_answer_cache({})


def make_cache_key(
    query: str, collection: str, mode: str, top_k: int, index_version: str
) -> str:
    raw = f"{collection}|{mode}|{top_k}|{index_version}|{query.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def append_query_log(row: Dict[str, Any]) -> None:
    with QUERY_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_query_log(limit: int = 50) -> List[Dict[str, Any]]:
    if not QUERY_LOG_FILE.exists():
        return []
    lines = QUERY_LOG_FILE.read_text(encoding="utf-8").splitlines()
    rows: List[Dict[str, Any]] = []
    for line in reversed(lines[-limit:]):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def list_files_in_dir(base_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not base_dir.exists():
        return rows

    for collection in settings.collections:
        col_dir = base_dir / collection
        if not col_dir.exists():
            continue
        for file_path in sorted(col_dir.glob("*")):
            if not file_path.is_file():
                continue
            stat = file_path.stat()
            rows.append(
                {
                    "collection": collection,
                    "name": file_path.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "path": str(file_path),
                }
            )
    return rows


def ingest_one_file(file_path: Path, collection: str) -> Dict[str, Any]:
    if collection not in settings.collections:
        raise ValueError(f"Unknown collection: {collection}")

    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    raw_bytes = file_path.read_bytes()
    # Читаем файл через readers.py (поддержка .docx и текстовых форматов)
    text, detected_doc_type = read_file_as_text(file_path)
    text = text.strip()
    if not text:
        raise ValueError(f"File is empty or unsupported: {file_path.name}")

    document_hash = compute_document_hash(raw_bytes)
    document_id = f"{collection}:{file_path.name}"
    # Пока оставляем doc_type равным коллекции (как было); detected_doc_type можно использовать позже
    doc_type = collection

    store = get_vector_store()

    if store.has_document(document_id=document_id, document_hash=document_hash):
        target = PROCESSED_DIR / collection / file_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if file_path.resolve() != target.resolve():
            shutil.move(str(file_path), str(target))
        return {
            "file_name": file_path.name,
            "collection": collection,
            "document_id": document_id,
            "chunks_count": 0,
            "status": "skipped_unchanged",
        }

    chunks = make_chunks_for_file(
        file_path=file_path,
        text=text,
        collection=collection,
        doc_type=doc_type,
        document_id=document_id,
        document_hash=document_hash,
    )

    if not chunks:
        raise ValueError(f"No chunks produced for file: {file_path.name}")

    embedder = YandexEmbeddingProvider()
    vectors = embedder.embed_texts([ch.text for ch in chunks])

    if len(vectors) != len(chunks):
        raise ValueError("Embeddings count does not match chunks count")

    for ch, vec in zip(chunks, vectors):
        ch.embedding = vec

    store.delete_by_document(document_id)
    store.upsert_chunks(chunks)

    target = PROCESSED_DIR / collection / file_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_path.resolve() != target.resolve():
        shutil.move(str(file_path), str(target))

    return {
        "file_name": file_path.name,
        "collection": collection,
        "document_id": document_id,
        "chunks_count": len(chunks),
        "status": "indexed",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result: Dict[str, Any] | None = None
    diagnostics: Dict[str, Any] | None = None

    query = ""
    collection = settings.collections[0] if settings.collections else "faq"
    mode = "balanced"
    top_k = settings.default_top_k

    if request.method == "POST":
        query = (request.form.get("query") or "").strip()
        collection = (request.form.get("collection") or collection).strip()
        mode = (request.form.get("mode") or mode).strip()
        try:
            top_k = int(request.form.get("top_k") or settings.default_top_k)
        except ValueError:
            top_k = settings.default_top_k

        if collection not in settings.collections:
            flash("Неизвестная коллекция", "error")
            return redirect(url_for("index"))

        if query:
            index_version = get_index_version()
            cache = load_answer_cache()
            cache_key = make_cache_key(query, collection, mode, top_k, index_version)

            if cache_key in cache:
                result = cache[cache_key]
                cache_hit = True
            else:
                result = ask(
                    query=query,
                    collection=collection,
                    mode=mode,
                    top_k=top_k,
                )
                cache[cache_key] = result
                save_answer_cache(cache)
                cache_hit = False

            diagnostics = {
                "collection": collection,
                "mode": mode,
                "top_k": top_k,
                "index_version": index_version,
                "cache_hit": cache_hit,
                "top_k_used": result.get("top_k_used", 0),
                "fetch_k_used": result.get("fetch_k_used", 0),
                "raw_found": result.get("raw_found", 0),
                "reranked_found": result.get("reranked_found", 0),
                "context_chunks_used": result.get("context_chunks_used", 0),
                "latency_ms": result.get("latency_ms", 0),
            }

            append_query_log(
                {
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "collection": collection,
                    "mode": mode,
                    "top_k": top_k,
                    "raw_found": diagnostics["raw_found"],
                    "reranked_found": diagnostics["reranked_found"],
                    "context_chunks_used": diagnostics["context_chunks_used"],
                    "latency_ms": diagnostics["latency_ms"],
                    "cache_hit": cache_hit,
                    "query": query,
                }
            )

    return render_template(
        "index.html",
        query=query,
        collection=collection,
        mode=mode,
        top_k=top_k,
        collections=settings.collections,
        result=result,
        diagnostics=diagnostics,
    )


@app.route("/admin", methods=["GET"])
def admin():
    store = get_vector_store()
    db_stats = store.stats()
    db_docs = store.get_documents_overview()

    input_files = list_files_in_dir(INPUT_DIR)
    processed_files = list_files_in_dir(PROCESSED_DIR)
    failed_files = list_files_in_dir(FAILED_DIR)

    db_path_full = db_stats.get("db_path", "")
    db_path_short = ""
    if db_path_full:
        try:
            db_path_short = str(Path(db_path_full).relative_to(BASE_DIR))
        except ValueError:
            db_path_short = db_path_full

    stats = {
        "index_version": get_index_version(),
        "input_count": len(input_files),
        "processed_count": len(processed_files),
        "failed_count": len(failed_files),
        "db_documents": db_stats.get("db_documents", 0),
        "db_chunks": db_stats.get("db_chunks", 0),
        "db_path": db_path_short,
        "collection_name": db_stats.get("collection_name", ""),
        "embedding_dim": db_stats.get("embedding_dim", 0),
    }

    return render_template(
        "admin.html",
        stats=stats,
        collections=settings.collections,
        input_files=input_files,
        processed_files=processed_files,
        db_docs=db_docs,
        query_log=read_query_log(),
    )


@app.route("/admin/upload", methods=["POST"])
def admin_upload():
    file = request.files.get("file")
    collection = (request.form.get("collection") or "").strip()

    if collection not in settings.collections:
        flash("Неизвестная коллекция", "error")
        return redirect(url_for("admin"))

    if not file or not file.filename:
        flash("Файл не выбран", "error")
        return redirect(url_for("admin"))

    safe_name = secure_filename(file.filename)
    if not safe_name:
        flash("Некорректное имя файла", "error")
        return redirect(url_for("admin"))

    save_path = INPUT_DIR / collection / safe_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(save_path)

    flash(f"Файл загружен: {safe_name}", "success")
    return redirect(url_for("admin"))


@app.route("/admin/ingest-all", methods=["POST"])
def ingest_all():
    indexed = 0
    skipped = 0
    failed = 0

    for collection in settings.collections:
        col_dir = INPUT_DIR / collection
        if not col_dir.exists():
            continue

        for file_path in sorted(col_dir.glob("*")):
            if not file_path.is_file():
                continue

            try:
                res = ingest_one_file(file_path, collection)
                if res["status"] == "indexed":
                    indexed += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                fail_target = FAILED_DIR / collection / file_path.name
                fail_target.parent.mkdir(parents=True, exist_ok=True)
                if file_path.exists():
                    shutil.move(str(file_path), str(fail_target))
                flash(f"Ошибка индексации {file_path.name}: {exc}", "error")

    bump_index_version()
    clear_answer_cache()
    flash(
        f"Ingest завершён. indexed={indexed}, skipped={skipped}, failed={failed}",
        "success",
    )
    return redirect(url_for("admin"))


@app.route("/admin/clear-cache", methods=["POST"])
def admin_clear_cache():
    clear_answer_cache()
    flash("Кэш ответов очищен", "success")
    return redirect(url_for("admin"))


@app.route("/admin/batch", methods=["POST"])
def admin_batch():
    selected = request.form.getlist("selected_files")
    action = (request.form.get("action") or "").strip()

    if not selected:
        flash("Не выбраны файлы в таблице Processed", "error")
        return redirect(url_for("admin"))

    store = get_vector_store()
    success = 0
    errors = 0

    for item in selected:
        try:
            collection, file_name, location = item.split("|", 2)
            document_id = f"{collection}:{file_name}"

            src_base = PROCESSED_DIR if location == "processed" else INPUT_DIR
            src_path = src_base / collection / file_name

            if action == "delete_db":
                store.delete_document(document_id)
                success += 1

            elif action == "reindex":
                target_input = INPUT_DIR / collection / file_name
                target_input.parent.mkdir(parents=True, exist_ok=True)

                if src_path.exists() and src_path.resolve() != target_input.resolve():
                    shutil.copy2(src_path, target_input)

                if target_input.exists():
                    ingest_one_file(target_input, collection)
                    success += 1
                else:
                    raise FileNotFoundError(str(target_input))

            elif action == "delete_processed":
                store.delete_document(document_id)
                if src_path.exists():
                    src_path.unlink()
                success += 1

            else:
                raise ValueError(f"Неизвестное действие: {action}")

        except Exception as exc:
            errors += 1
            flash(f"Ошибка batch для {item}: {exc}", "error")

    if success:
        bump_index_version()
        clear_answer_cache()

    flash(f"Batch завершён. success={success}, errors={errors}", "success")
    return redirect(url_for("admin"))


if __name__ == "__main__":
    app.run(debug=True)