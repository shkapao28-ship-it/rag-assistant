# rag_service/ingestion/readers.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

try:
    import docx  # python-docx
except ImportError:
    docx = None


TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json"}
DOCX_EXTENSIONS = {".docx"}


def _read_text_generic(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_docx(path: Path) -> str:
    if docx is None:
        # Бэкап, если python-docx не установлен — лучше явная ошибка, чем индексировать XML
        raise RuntimeError(
            "Для чтения .docx требуется пакет python-docx. "
            "Установи его: pip install python-docx"
        )

    doc = docx.Document(str(path))
    paragraphs = []
    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def read_file_as_text(path: str | Path) -> Tuple[str, str]:
    """
    Унифицированный вход: путь -> (text, doc_type).
    doc_type можно использовать дальше, если захочешь отличать FAQ / training и т.п.
    Сейчас возвращаем просто 'faq' или 'docx' по необходимости.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in TEXT_EXTENSIONS:
        text = _read_text_generic(p)
        doc_type = "text"
    elif suffix in DOCX_EXTENSIONS:
        text = _read_docx(p)
        doc_type = "docx"
    else:
        # по умолчанию пробуем как текст, но явно помечаем тип
        text = _read_text_generic(p)
        doc_type = suffix.lstrip(".") or "unknown"

    return text, doc_type