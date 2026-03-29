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
        top_k: int = 3,
        alpha: float = 0.5,
        filter_category: Optional[str] = None,
    ) -> list[dict]:
        """
        Return top_k parent documents relevant to query_text.

        Parameters
        ----------
        query_text      : natural language query
        top_k           : number of parent docs to return
        alpha           : vector weight (0 = BM25 only, 1 = vector only)
        filter_category : if set, restrict to this file_category (game only)
        """
        candidate_k = top_k * 5  # over-fetch before dedup

        # ── vector retrieval ─────────────────────────────────────────────────
        query_emb = self.embed_model._get_query_embedding(query_text)

        chroma_kwargs: dict = {"query_embeddings": [query_emb], "n_results": candidate_k}
        if filter_category:
            chroma_kwargs["where"] = {"file_category": filter_category}

        chroma_result = self.collection.query(**chroma_kwargs)
        vec_ids: list[str] = chroma_result["ids"][0]
        # ChromaDB returns distances (lower = closer for cosine); convert to similarity
        vec_distances: list[float] = chroma_result["distances"][0]
        vec_scores_raw = {cid: 1.0 - dist for cid, dist in zip(vec_ids, vec_distances)}

        # ── BM25 retrieval ───────────────────────────────────────────────────
        tokenized_query = query_text.lower().split()
        bm25_raw_scores = self._bm25.get_scores(tokenized_query)

        # Apply category filter to BM25 results if needed
        bm25_candidates: list[tuple[str, float]] = []
        for node, score in zip(self._bm25_nodes, bm25_raw_scores):
            if filter_category and node["metadata"].get("file_category") != filter_category:
                continue
            bm25_candidates.append((node["child_id"], float(score)))
        bm25_candidates.sort(key=lambda x: x[1], reverse=True)
        bm25_candidates = bm25_candidates[:candidate_k]
        bm25_scores_raw = {cid: score for cid, score in bm25_candidates}

        # ── normalise scores ─────────────────────────────────────────────────
        def _normalise(scores: dict[str, float]) -> dict[str, float]:
            if not scores:
                return {}
            max_s = max(scores.values()) or 1.0
            return {k: v / max_s for k, v in scores.items()}

        vec_scores = _normalise(vec_scores_raw)
        bm25_scores = _normalise(bm25_scores_raw)

        # ── combine ──────────────────────────────────────────────────────────
        all_child_ids = set(vec_scores) | set(bm25_scores)
        combined: dict[str, float] = {
            cid: alpha * vec_scores.get(cid, 0.0) + (1 - alpha) * bm25_scores.get(cid, 0.0)
            for cid in all_child_ids
        }

        # ── map child → parent (deduplicate, keep best child score per parent) ─
        parent_scores: dict[str, float] = {}
        child_to_parent = self._build_child_parent_map(all_child_ids)

        for cid, score in combined.items():
            pid = child_to_parent.get(cid)
            if pid is None:
                continue
            if pid not in parent_scores or score > parent_scores[pid]:
                parent_scores[pid] = score

        # ── rank parents ─────────────────────────────────────────────────────
        ranked_pids = sorted(parent_scores, key=lambda x: parent_scores[x], reverse=True)[:top_k]

        results = []
        for pid in ranked_pids:
            parent = self._parents.get(pid)
            if parent:
                results.append({**parent, "score": parent_scores[pid]})

        return results

    def _build_child_parent_map(self, child_ids: set[str]) -> dict[str, str]:
        """
        Look up parent_id for each child_id.
        First check BM25 node list, then fall back to ChromaDB metadata fetch.
        """
        mapping: dict[str, str] = {}

        # Fast lookup from BM25 nodes (already in memory)
        for node in self._bm25_nodes:
            if node["child_id"] in child_ids:
                mapping[node["child_id"]] = node["metadata"]["parent_id"]

        # Remaining: fetch from ChromaDB
        missing = child_ids - set(mapping)
        if missing:
            result = self.collection.get(ids=list(missing), include=["metadatas"])
            for cid, meta in zip(result["ids"], result["metadatas"]):
                mapping[cid] = meta["parent_id"]

        return mapping


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
