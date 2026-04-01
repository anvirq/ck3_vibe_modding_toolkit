"""
Custom LlamaIndex embedding class using Qwen3 Embedding via OpenRouter.
OpenRouter exposes an OpenAI-compatible /embeddings endpoint.
"""

import concurrent.futures
from typing import List, Optional
import openai
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_QUERY_INSTRUCTION_ENABLED,
)


class QwenOpenRouterEmbedding(BaseEmbedding):
    """Qwen3 Embedding via OpenRouter (OpenAI-compatible API)."""

    api_key: str = Field(default="")
    api_base: str = Field(default=OPENROUTER_BASE_URL)
    dimensions: Optional[int] = Field(default=EMBEDDING_DIMENSIONS)
    batch_size: int = Field(default=32)
    max_concurrent: int = Field(default=20)
    # Query-side Instruct: text per index (game vs wiki); indexing pipeline leaves default "".
    query_instruction: str = Field(default="")

    # Internal client — excluded from pydantic serialisation
    _client: openai.OpenAI = None  # type: ignore[assignment]

    def __init__(self, **kwargs):
        kwargs.setdefault("model_name", EMBEDDING_MODEL)
        kwargs.setdefault("api_key", OPENROUTER_API_KEY or "")
        if not kwargs["api_key"]:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        super().__init__(**kwargs)
        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    # ── internal helpers ────────────────────────────────────────────────────

    def _format_query_for_embedding(self, query: str) -> str:
        """Qwen3-style task instruction on queries only; indexing stays raw text."""
        if not EMBEDDING_QUERY_INSTRUCTION_ENABLED:
            return query
        instr = (self.query_instruction or "").strip()
        if not instr:
            return query
        return f"Instruct: {instr}\nQuery:{query}"

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        kwargs: dict = {"model": self.model_name, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        # Sort by index to guarantee ordering
        items = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in items]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        if len(batches) <= 1:
            return self._call_api(batches[0]) if batches else []
        results: List[Optional[List[List[float]]]] = [None] * len(batches)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {pool.submit(self._call_api, batch): idx for idx, batch in enumerate(batches)}
            for fut in concurrent.futures.as_completed(futures):
                results[futures[fut]] = fut.result()
        return [emb for batch_result in results for emb in batch_result]

    # ── required LlamaIndex interface ───────────────────────────────────────

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_batch([self._format_query_for_embedding(query)])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)
