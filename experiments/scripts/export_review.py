"""
Export per-query retrieval results for manual review.

Usage:
    py experiments/scripts/export_review.py experiments/configs/baseline.yaml
    py experiments/scripts/export_review.py experiments/configs/ast_256.yaml --query-subset narrow --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chromadb

from src.config import CHROMA_DIR
from src.ingestion.experiment_config import ExperimentConfig
from src.retrieval.embeddings import QwenOpenRouterEmbedding
from src.retrieval.hybrid_search import HybridSearchIndex

from experiments.scripts.evaluate import load_queries
from experiments.scripts.rerankers import CrossEncoderReranker


def _snippet(text: str, max_lines: int = 3, max_chars: int = 280) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    out = "\n".join(lines[:max_lines])
    if len(out) > max_chars:
        return out[: max_chars - 3] + "..."
    return out


def export_review(cfg: ExperimentConfig, query_subset: str | None, top_k: int,
                  rerank: bool = False, cross_encoder_model: str | None = None) -> Path:
    queries = load_queries(query_subset)
    subset_name = query_subset or "all"
    out_path = cfg.results_dir / f"manual_review_{subset_name}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_game = QwenOpenRouterEmbedding()
    embed_wiki = QwenOpenRouterEmbedding()
    ce_reranker = CrossEncoderReranker(cross_encoder_model) if cross_encoder_model else None

    indexes: dict[str, HybridSearchIndex] = {}
    for label, collection_name, bm25_path, parents_path in [
        ("game", cfg.game_collection, cfg.bm25_game_path, cfg.parents_game_path),
        ("wiki", cfg.wiki_collection, cfg.bm25_wiki_path, cfg.parents_wiki_path),
    ]:
        if not bm25_path.exists():
            continue
        indexes[label] = HybridSearchIndex(
            collection_name=collection_name,
            bm25_path=bm25_path,
            parents_path=parents_path,
            embed_model=embed_game if label == "game" else embed_wiki,
            chroma_client=chroma,
        )

    lines: list[str] = [
        f"# Manual Review - {cfg.name}",
        "",
        f"- query_subset: `{subset_name}`",
        f"- top_k: `{top_k}`",
        "",
    ]

    for q in queries:
        coll = q["collection"]
        lines.extend(
            [
                f"## {q['id']} - {q['query']}",
                "",
                f"- collection: `{coll}`",
                f"- category: `{q.get('category', '-')}`",
                f"- expected: `{', '.join(q.get('expected', []))}`",
                "",
            ]
        )

        index = indexes.get(coll)
        if not index:
            lines.extend(["_Index not found for this collection._", ""])
            continue

        results = index.query(
            query_text=q["query"],
            top_k=top_k,
            alpha=cfg.alpha,
            filter_category=q.get("category"),
            rerank=rerank,
        )
        if ce_reranker:
            results = ce_reranker.rerank(q["query"], results)
        if not results:
            lines.extend(["_No results._", ""])
            continue

        for i, r in enumerate(results, start=1):
            title = r.get("block_name") or r.get("section_title") or "-"
            lines.extend(
                [
                    f"### Rank {i}",
                    "",
                    f"- score: `{r.get('score', 0):.4f}`",
                    f"- file: `{r.get('file_path', '-')}`",
                    f"- title: `{title}`",
                    f"- parent_id: `{r.get('parent_id', '-')}`",
                    "",
                    "```text",
                    _snippet(r.get("text", "")),
                    "```",
                    "",
                ]
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--query-subset", help="Subset name or explicit .txt path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--cross-encoder-model")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    out = export_review(
        cfg,
        args.query_subset,
        args.top_k,
        rerank=args.rerank,
        cross_encoder_model=args.cross_encoder_model,
    )
    print(f"Saved manual review -> {out}")


if __name__ == "__main__":
    main()
