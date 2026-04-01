from __future__ import annotations

from typing import Sequence


class CrossEncoderReranker:
    """Cross-encoder reranker for (query, candidate_text) pairs."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: py -m pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        if not results:
            return results
        pairs: Sequence[tuple[str, str]] = [(query, r.get("text", "")) for r in results]
        scores = self._model.predict(list(pairs))
        rescored = []
        for row, score in zip(results, scores):
            item = dict(row)
            item["rerank_score"] = float(score)
            rescored.append(item)
        rescored.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)
        return rescored
