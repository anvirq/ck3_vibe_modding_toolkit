"""
Hybrid retriever: combines vector (ChromaDB cosine) + BM25 (rank-bm25).

Scores are independently normalised to [0, 1], then combined:
    final = alpha * vector_score + (1 - alpha) * bm25_score

Returns a list of parent document dicts (deduplicated) ranked by combined score.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi

from src.config import (
    CHROMA_DIR,
    BM25_DIR,
    PARENTS_DIR,
    GAME_COLLECTION,
    WIKI_COLLECTION,
    EMBEDDING_QUERY_INSTRUCTION_GAME,
    EMBEDDING_QUERY_INSTRUCTION_WIKI,
)
from src.retrieval.embeddings import QwenOpenRouterEmbedding


class HybridSearchIndex:
    """
    Loaded once at server startup; holds all in-memory structures
    needed for hybrid search over one collection.
    """

    def __init__(
        self,
        collection_name: str,
        bm25_path: Path,
        parents_path: Path,
        embed_model: QwenOpenRouterEmbedding,
        chroma_client: chromadb.ClientAPI,
    ):
        self.embed_model = embed_model
        self.collection = chroma_client.get_collection(collection_name)

        # BM25
        with open(bm25_path, "rb") as f:
            payload = pickle.load(f)
        self._bm25: BM25Okapi = payload["bm25"]
        self._bm25_nodes: list[dict] = payload["nodes"]

        # Parent store
        with open(parents_path, "r", encoding="utf-8") as f:
            self._parents: dict[str, dict] = json.load(f)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filter_category: Optional[str] = None,
    ) -> list[dict]:
        """
        Return top_k child chunks ranked by hybrid score.

        Each result dict contains the child chunk text and metadata, plus
        a `parent_id` field the caller can pass to `get_parent()` for full context.

        Parameters
        ----------
        query_text      : natural language query
        top_k           : number of child chunks to return
        alpha           : vector weight (0 = BM25 only, 1 = vector only)
        filter_category : if set, restrict to this file_category (game only)
        """
        candidate_k = top_k * 5  # over-fetch before dedup by parent

        # ── vector retrieval ─────────────────────────────────────────────────
        query_emb = self.embed_model._get_query_embedding(query_text)

        chroma_kwargs: dict = {"query_embeddings": [query_emb], "n_results": candidate_k,
                               "include": ["documents", "metadatas", "distances"]}
        if filter_category:
            chroma_kwargs["where"] = {"file_category": filter_category}

        chroma_result = self.collection.query(**chroma_kwargs)
        vec_ids: list[str] = chroma_result["ids"][0]
        vec_distances: list[float] = chroma_result["distances"][0]
        vec_docs: list[str] = chroma_result["documents"][0]
        vec_metas: list[dict] = chroma_result["metadatas"][0]

        vec_scores_raw = {cid: 1.0 - dist for cid, dist in zip(vec_ids, vec_distances)}
        chroma_nodes = {cid: {"text": doc, "metadata": meta}
                        for cid, doc, meta in zip(vec_ids, vec_docs, vec_metas)}

        # ── BM25 retrieval ───────────────────────────────────────────────────
        tokenized_query = query_text.lower().split()
        bm25_raw_scores = self._bm25.get_scores(tokenized_query)

        bm25_candidates: list[tuple[str, float]] = []
        for node, score in zip(self._bm25_nodes, bm25_raw_scores):
            if filter_category and node["metadata"].get("file_category") != filter_category:
                continue
            bm25_candidates.append((node["child_id"], float(score)))
        bm25_candidates.sort(key=lambda x: x[1], reverse=True)
        bm25_candidates = bm25_candidates[:candidate_k]
        bm25_scores_raw = {cid: score for cid, score in bm25_candidates}

        # BM25 nodes as lookup (text + metadata already in memory)
        bm25_node_map = {n["child_id"]: n for n in self._bm25_nodes}

        # ── normalise + combine ───────────────────────────────────────────────
        def _norm(scores: dict[str, float]) -> dict[str, float]:
            if not scores:
                return {}
            max_s = max(scores.values()) or 1.0
            return {k: v / max_s for k, v in scores.items()}

        vec_scores = _norm(vec_scores_raw)
        bm25_scores = _norm(bm25_scores_raw)

        all_child_ids = set(vec_scores) | set(bm25_scores)
        combined: dict[str, float] = {
            cid: alpha * vec_scores.get(cid, 0.0) + (1 - alpha) * bm25_scores.get(cid, 0.0)
            for cid in all_child_ids
        }

        # ── deduplicate by parent (best child per parent wins) ────────────────
        best_per_parent: dict[str, tuple[str, float]] = {}  # parent_id → (child_id, score)
        for cid, score in combined.items():
            node = chroma_nodes.get(cid) or bm25_node_map.get(cid)
            if not node:
                continue
            pid = node["metadata"].get("parent_id", "")
            if pid not in best_per_parent or score > best_per_parent[pid][1]:
                best_per_parent[pid] = (cid, score)

        # ── rank and build results ────────────────────────────────────────────
        ranked = sorted(best_per_parent.values(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for cid, score in ranked:
            node = chroma_nodes.get(cid) or bm25_node_map.get(cid)
            if not node:
                continue
            results.append({
                "text": node["text"],
                "score": score,
                **node["metadata"],
            })

        return results

    def get_parent(self, parent_id: str) -> Optional[dict]:
        """Return the full parent document for a given parent_id, or None."""
        return self._parents.get(parent_id)


# ── singleton factory ─────────────────────────────────────────────────────────

_game_index: Optional[HybridSearchIndex] = None
_wiki_index: Optional[HybridSearchIndex] = None


def _get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_game_index() -> HybridSearchIndex:
    global _game_index
    if _game_index is None:
        embed_model = QwenOpenRouterEmbedding(
            query_instruction=EMBEDDING_QUERY_INSTRUCTION_GAME,
        )
        _game_index = HybridSearchIndex(
            collection_name=GAME_COLLECTION,
            bm25_path=BM25_DIR / "game.pkl",
            parents_path=PARENTS_DIR / "game.json",
            embed_model=embed_model,
            chroma_client=_get_chroma_client(),
        )
    return _game_index


def get_wiki_index() -> HybridSearchIndex:
    global _wiki_index
    if _wiki_index is None:
        embed_model = QwenOpenRouterEmbedding(
            query_instruction=EMBEDDING_QUERY_INSTRUCTION_WIKI,
        )
        _wiki_index = HybridSearchIndex(
            collection_name=WIKI_COLLECTION,
            bm25_path=BM25_DIR / "wiki.pkl",
            parents_path=PARENTS_DIR / "wiki.json",
            embed_model=embed_model,
            chroma_client=_get_chroma_client(),
        )
    return _wiki_index
