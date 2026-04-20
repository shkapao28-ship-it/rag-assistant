from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -------- Yandex / LLM --------
    yandex_api_key: str = Field(default="", alias="YANDEX_CLOUD_API_KEY")
    yandex_folder_id: str = Field(default="", alias="YANDEX_CLOUD_FOLDER")
    yandex_llm_model: str = Field(default="", alias="YANDEX_MODEL_ID")
    yandex_embedding_model: str = Field(default="", alias="YANDEX_EMBEDDING_MODEL")
    yandex_embed_url: str = Field(
        default="https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
        alias="YANDEX_EMBED_URL",
    )
    yandex_temperature: float = Field(default=0.2, alias="YANDEX_TEMPERATURE")
    yandex_max_tokens: int = Field(default=600, alias="YANDEX_MAX_TOKENS")

    # -------- Retrieval --------
    default_top_k: int = Field(default=4, alias="DEFAULT_TOP_K")
    fetch_k: int = Field(default=12, alias="FETCH_K")
    max_context_chunks: int = Field(default=4, alias="MAX_CONTEXT_CHUNKS")

    # -------- Files / ingestion --------
    input_folder_raw: str = Field(default="./input", alias="INPUT_FOLDER")
    processed_folder_raw: str = Field(default="./processed", alias="PROCESSED_FOLDER")
    failed_folder_raw: str = Field(default="./failed", alias="FAILED_FOLDER")

    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    min_chunk_length: int = Field(default=250, alias="MIN_CHUNK_LENGTH")
    lookahead: int = Field(default=220, alias="LOOKAHEAD")
    request_timeout: int = Field(default=10, alias="REQUEST_TIMEOUT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    doc_type: str = Field(default="faq", alias="DOC_TYPE")
    default_lang: str = Field(default="ru", alias="DEFAULT_LANG")
    default_version: str = Field(default="1.0", alias="DEFAULT_VERSION")
    default_status: str = Field(default="active", alias="DEFAULT_STATUS")
    default_tags: str = Field(default="", alias="DEFAULT_TAGS")
    title_from_filename: bool = Field(default=True, alias="TITLE_FROM_FILENAME")

    # -------- Optional integrations --------
    n8n_webhook: str = Field(default="", alias="N8N_WEBHOOK")
    supabase_host: str = Field(default="", alias="SUPABASE_HOST")
    supabase_port: int = Field(default=5432, alias="SUPABASE_PORT")
    supabase_db: str = Field(default="", alias="SUPABASE_DB")
    supabase_user: str = Field(default="", alias="SUPABASE_USER")
    supabase_password: str = Field(default="", alias="SUPABASE_PASSWORD")

    # -------- App defaults --------
    collections: List[str] = ["faq"]

    @staticmethod
    def _resolve_project_path(raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (BASE_DIR / path).resolve()

    @property
    def input_dir(self) -> Path:
        return self._resolve_project_path(self.input_folder_raw)

    @property
    def processed_dir(self) -> Path:
        return self._resolve_project_path(self.processed_folder_raw)

    @property
    def failed_dir(self) -> Path:
        return self._resolve_project_path(self.failed_folder_raw)

    @property
    def logs_dir(self) -> Path:
        return BASE_DIR / "logs"

    @property
    def chroma_dir(self) -> Path:
        return BASE_DIR / "chroma_data"

    # Совместимость со старым кодом chroma_store.py
    @property
    def chroma_persist_dir(self) -> Path:
        return self.chroma_dir


def ensure_project_dirs() -> None:
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.failed_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    for collection in settings.collections:
        (settings.input_dir / collection).mkdir(parents=True, exist_ok=True)
        (settings.processed_dir / collection).mkdir(parents=True, exist_ok=True)
        (settings.failed_dir / collection).mkdir(parents=True, exist_ok=True)


settings = AppSettings()