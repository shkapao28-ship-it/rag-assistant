from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import requests

from rag_service.config import settings


YANDEX_API_URL_COMPLETIONS = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEX_API_URL_EMBEDDINGS = settings.yandex_embed_url
EXPECTED_EMBEDDING_DIM = 256


class YandexApiError(RuntimeError):
    pass


@dataclass
class YandexEmbeddingProvider:
    api_key: str = settings.yandex_api_key
    folder_id: str = settings.yandex_folder_id
    model: str = settings.yandex_embedding_model
    timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    expected_dim: int = EXPECTED_EMBEDDING_DIM

    def _do_request(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id,
        }

        payload = {
            "modelUri": self.model,
            "text": text,
        }

        resp = requests.post(
            YANDEX_API_URL_EMBEDDINGS,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise YandexApiError(f"Yandex embedding error {resp.status_code}: {resp.text}")

        data = resp.json()

        if "embedding" in data:
            vector = data["embedding"]
        elif isinstance(data, dict) and "result" in data and "embedding" in data["result"]:
            vector = data["result"]["embedding"]
        else:
            raise YandexApiError(f"Unexpected embedding response: {data}")

        if not isinstance(vector, list) or not vector:
            raise YandexApiError(f"Empty or invalid embedding vector: {data}")

        if self.expected_dim and len(vector) != self.expected_dim:
            raise YandexApiError(
                f"Unexpected embedding dimension: got {len(vector)}, expected {self.expected_dim}"
            )

        return vector

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._do_request(text) for text in texts]

    def embed_text(self, text: str) -> List[float]:
        return self._do_request(text)


@dataclass
class YandexLLMProvider:
    api_key: str = settings.yandex_api_key
    folder_id: str = settings.yandex_folder_id
    model: str = settings.yandex_llm_model
    timeout: int = int(os.getenv("REQUEST_TIMEOUT", "40"))

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temp = float(os.getenv("YANDEX_TEMPERATURE", "0.2")) if temperature is None else float(temperature)
        max_toks = int(os.getenv("YANDEX_MAX_TOKENS", "600")) if max_tokens is None else int(max_tokens)

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id,
        }

        payload: Dict[str, Any] = {
            "modelUri": self.model,
            "completionOptions": {
                "stream": False,
                "temperature": temp,
                "maxTokens": max_toks,
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_prompt},
            ],
        }

        resp = requests.post(
            YANDEX_API_URL_COMPLETIONS,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise YandexApiError(f"Yandex LLM error {resp.status_code}: {resp.text}")

        data = resp.json()

        try:
            alternatives = data["result"]["alternatives"]
            first = alternatives[0]
            text = first["message"]["text"]
            return text.strip()
        except Exception as exc:
            raise YandexApiError(f"Unexpected LLM response: {data}") from exc