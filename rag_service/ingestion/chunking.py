
    # rag_service/ingestion/chunking.py

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..config import settings
from ..vector_store.base import ChunkRecord


# ---- Вспомогательные структуры ----

@dataclass
class Section:
    title: str | None
    level: int
    text: str


# ---- Hash-и ----

def compute_document_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compute_chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- Основной вход: текст -> чанки ----

def make_chunks_for_file(
    file_path: str | Path,
    text: str,
    collection: str,
    doc_type: str,
    document_id: str,
    document_hash: str,
) -> List[ChunkRecord]:
    """
    Делает hybrid heading-aware chunking для одного файла
    и возвращает список ChunkRecord без эмбеддингов (embedding заполним позже).
    """
    sections = split_into_sections(text)
    raw_chunks: list[str] = []

    for sec in sections:
        sec_chunks = split_section_into_chunks(sec)
        raw_chunks.extend(sec_chunks)

    # Фильтруем по минимальной длине
    min_len = settings.min_chunk_length
    raw_chunks = [c for c in raw_chunks if len(c) >= min_len]

    # Fallback для коротких документов: если текст не пустой, но чанков нет,
    # сохраняем один чанк целиком
    if not raw_chunks and text.strip():
        raw_chunks = [text.strip()]

    chunk_total = len(raw_chunks)
    chunk_records: List[ChunkRecord] = []

    # section-метка для каждого чанка можно брать по простой эвристике:
    # заголовок секции, в которой он был (мы уже встроили в текст, можем позже усложнить)
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_id = make_chunk_id(document_id, idx)
        section_title = extract_section_title_from_chunk(chunk_text)

        chunk_records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                document_hash=document_hash,
                text=chunk_text,
                chunk_index=idx,
                chunk_total=chunk_total,
                section=section_title,
                doc_type=doc_type,
                collection=collection,
                source=str(Path(file_path).name),
                embedding=[],  # заполним на этапе embeddings
            )
        )

    return chunk_records


def make_chunk_id(document_id: str, index: int) -> str:
    # Стабильный ID чанка: doc_id + индекс
    return f"{document_id}_{index}"


# ---- Разбор структуры документа ----

_HEADING_PATTERN = re.compile(r"^(#+\s+|[A-ZА-Я0-9][^a-zа-я\n]{5,}\s*$)")


def split_into_sections(text: str) -> List[Section]:
    """
    Очень упрощённый heading-aware разбор:
    - ищем строки, похожие на заголовки (markdown # или капс);
    - между заголовками собираем секции.
    """
    lines = text.split("\n")
    sections: List[Section] = []

    current_title: str | None = None
    current_level: int = 1
    buf: List[str] = []

    def flush_section() -> None:
        nonlocal buf, current_title, current_level
        content = "\n".join(buf).strip()
        if content:
            sections.append(
                Section(
                    title=current_title,
                    level=current_level,
                    text=content,
                )
            )
        buf = []

    for line in lines:
        if _is_heading(line):
            # закрываем предыдущую секцию
            flush_section()
            # новая секция
            current_title = line.strip("# ").strip()
            current_level = _heading_level(line)
            buf = []
        else:
            buf.append(line)

    # финальная секция
    flush_section()

    # если заголовков не было, будет одна секция с title=None
    if not sections:
        sections.append(Section(title=None, level=1, text=text.strip()))

    return sections


def _is_heading(line: str) -> bool:
    line_stripped = line.strip()
    if not line_stripped:
        return False
    # markdown-style heading
    if line_stripped.startswith("#"):
        return True
    # капсовый / "шапочный" заголовок (много заглавных, мало строчных)
    if _HEADING_PATTERN.match(line_stripped):
        return True
    return False


def _heading_level(line: str) -> int:
    line_stripped = line.strip()
    if line_stripped.startswith("#"):
        return len(line_stripped) - len(line_stripped.lstrip("#"))
    return 1


# ---- Разбиение секций на чанки ----

def split_section_into_chunks(section: Section) -> List[str]:
    """
    Делит текст секции на чанки:
    1) режем по абзацам;
    2) если абзац слишком большой — режем по предложениям;
    3) если всё ещё больше лимита — делаем скользящее окно по символам.
    В начало каждого чанка включаем заголовок секции, если он есть.
    """
    max_size = settings.chunk_size
    overlap = settings.chunk_overlap

    paragraphs = _split_paragraphs(section.text)
    chunks: List[str] = []
    current: List[str] = []

    def flush_current():
        nonlocal current
        if not current:
            return
        chunk_body = "\n\n".join(current).strip()
        if not chunk_body:
            current = []
            return

        if section.title:
            chunk_text = f"{section.title}\n\n{chunk_body}"
        else:
            chunk_text = chunk_body

        if len(chunk_text) > max_size:
            # Дополнительное деление
            chunks.extend(_split_long_text(chunk_text, max_size, overlap))
        else:
            chunks.append(chunk_text)

        current = []

    for para in paragraphs:
        if not para.strip():
            continue
        tentative = "\n\n".join(current + [para]).strip()
        if len(tentative) <= max_size:
            current.append(para)
        else:
            flush_current()
            if len(para) > max_size:
                chunks.extend(_split_long_text(para, max_size, overlap, title=section.title))
            else:
                current = [para]

    flush_current()
    return chunks


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n")]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")


def _split_long_text(
    text: str,
    max_size: int,
    overlap: int,
    title: Optional[str] = None,
) -> List[str]:
    """
    Делит слишком длинный текст:
    - сначала по предложениям,
    - потом, если нужно, скользящим окном по символам.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text)
    chunks: List[str] = []
    current: List[str] = []

    def flush_current():
        nonlocal current
        if not current:
            return
        body = " ".join(current).strip()
        if not body:
            current = []
            return
        if title:
            full = f"{title}\n\n{body}"
        else:
            full = body
        if len(full) > max_size:
            # окончательный fallback: character-based window
            chunks.extend(_split_by_window(full, max_size, overlap))
        else:
            chunks.append(full)
        current = []

    for sent in sentences:
        if not sent.strip():
            continue
        tentative = " ".join(current + [sent]).strip()
        if len(tentative) <= max_size:
            current.append(sent)
        else:
            flush_current()
            current = [sent]

    flush_current()
    return chunks


def _split_by_window(text: str, max_size: int, overlap: int) -> List[str]:
    """
    Делит текст по символам с overlap.
    """
    chunks: List[str] = []
    start = 0
    length = len(text)

    if overlap >= max_size:
        overlap = max_size // 4

    while start < length:
        end = min(start + max_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap

    return chunks


def extract_section_title_from_chunk(chunk_text: str) -> str | None:
    """
    Простая эвристика: если в начале чанка есть строка-заголовок (первая строка без точки),
    считаем её section.
    """
    first_line = chunk_text.split("\n", 1)[0].strip()
    if not first_line:
        return None
    # если строка короткая и без точки — скорее всего заголовок
    if len(first_line) <= 80 and "." not in first_line:
        return first_line
    return None