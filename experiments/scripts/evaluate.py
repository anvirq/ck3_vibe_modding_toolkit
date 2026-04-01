"""
Evaluate a single experiment against eval_queries.yaml.

Usage:
    py experiments/scripts/evaluate.py experiments/configs/baseline.yaml
    py experiments/scripts/evaluate.py experiments/configs/baseline.yaml --top-k 5
    py experiments/scripts/evaluate.py experiments/configs/baseline.yaml --verbose
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import yaml
import chromadb

from src.config import CHROMA_DIR
from src.ingestion.experiment_config import ExperimentConfig
from src.retrieval.hybrid_search import HybridSearchIndex
from src.retrieval.embeddings import QwenOpenRouterEmbedding
from experiments.scripts.rerankers import CrossEncoderReranker

EVAL_QUERIES_PATH = Path(__file__).parent.parent / "eval_queries.yaml"
SUBSETS_QUERY_DIR = Path(__file__).parent.parent / "subsets" / "queries"


def resolve_query_subset_arg(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    if p.suffix:
        raise FileNotFoundError(f"Subset file not found: {arg}")
    candidate = SUBSETS_QUERY_DIR / f"{arg}.txt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unknown query subset '{arg}' (expected: {candidate})")


def load_query_ids(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
    return {line for line in lines if line}


def load_queries(query_subset: str | None = None) -> list[dict]:
    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]
    if not query_subset:
        return queries
    subset_path = resolve_query_subset_arg(query_subset)
    ids = load_query_ids(subset_path)
    return [q for q in queries if q["id"] in ids]


def is_hit(result: dict, expected: list[str]) -> bool:
    haystack = (result.get("text", "") + " " + result.get("block_name", "") +
                " " + result.get("section_title", "")).lower()
    return any(e.lower() in haystack for e in expected)


def evaluate(cfg: ExperimentConfig, top_k: int = 5, verbose: bool = False,
             query_subset: str | None = None, rerank: bool = False,
             cross_encoder_model: str | None = None,
             candidate_k: int | None = None) -> dict:
    queries = load_queries(query_subset)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Load both indexes
    embed_game = QwenOpenRouterEmbedding()
    embed_wiki = QwenOpenRouterEmbedding()

    indexes = {}
    ce_reranker = CrossEncoderReranker(cross_encoder_model) if cross_encoder_model else None
    for label, collection_name, bm25_path, parents_path in [
        ("game", cfg.game_collection, cfg.bm25_game_path, cfg.parents_game_path),
        ("wiki", cfg.wiki_collection, cfg.bm25_wiki_path, cfg.parents_wiki_path),
    ]:
        if not bm25_path.exists():
            print(f"[{label}] Not indexed yet - run run_experiment.py first")
            continue
        indexes[label] = HybridSearchIndex(
            collection_name=collection_name,
            bm25_path=bm25_path,
            parents_path=parents_path,
            embed_model=embed_game if label == "game" else embed_wiki,
            chroma_client=chroma,
        )

    results_log = []
    hits_at = {1: 0, 3: 0, 5: 0}
    total = 0

    for q in queries:
        coll = q["collection"]
        if coll not in indexes:
            continue

        index = indexes[coll]
        retrieval_k = candidate_k if candidate_k and candidate_k > top_k else top_k
        results = index.query(
            query_text=q["query"],
            top_k=retrieval_k,
            alpha=cfg.alpha,
            filter_category=q.get("category"),
            rerank=rerank,
        )
        if ce_reranker:
            results = ce_reranker.rerank(q["query"], results)
        results = results[:top_k]

        expected = q["expected"]
        hit1 = any(is_hit(r, expected) for r in results[:1])
        hit3 = any(is_hit(r, expected) for r in results[:3])
        hit5 = any(is_hit(r, expected) for r in results[:5])

        if hit1: hits_at[1] += 1
        if hit3: hits_at[3] += 1
        if hit5: hits_at[5] += 1
        total += 1

        entry = {
            "id": q["id"],
            "query": q["query"],
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "top_result": results[0].get("text", "")[:200] if results else "",
            "top_score": results[0].get("score", 0) if results else 0,
        }
        results_log.append(entry)

        if verbose:
            status = "HIT@1" if hit1 else ("HIT@3" if hit3 else ("HIT@5" if hit5 else "MISS"))
            print(f"[{q['id']}] {status}  score={entry['top_score']:.2f}  {q['query']!r}")
            if not hit1 and results:
                print(f"       top: {results[0].get('text', '')[:120].strip()!r}")

    summary = {
        "experiment": cfg.name,
        "total_queries": total,
        "hit@1": hits_at[1] / total if total else 0,
        "hit@3": hits_at[3] / total if total else 0,
        "hit@5": hits_at[5] / total if total else 0,
    }

    # Save results
    out_path = cfg.results_dir / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "queries": results_log}, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--query-subset",
        help="Subset name (from experiments/subsets/queries/<name>.txt) or explicit .txt path",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply query-aware lexical reranking on retrieved top-k results.",
    )
    parser.add_argument(
        "--cross-encoder-model",
        help="Optional sentence-transformers cross-encoder model for reranking top-k.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        help="If set, retrieve this many candidates before reranking, then keep top-k.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    summary = evaluate(
        cfg,
        top_k=args.top_k,
        verbose=args.verbose,
        query_subset=args.query_subset,
        rerank=args.rerank,
        cross_encoder_model=args.cross_encoder_model,
        candidate_k=args.candidate_k,
    )

    print(f"\n{'='*40}")
    print(f"Experiment : {summary['experiment']}")
    print(f"Queries    : {summary['total_queries']}")
    print(f"Hit@1      : {summary['hit@1']:.0%}")
    print(f"Hit@3      : {summary['hit@3']:.0%}")
    print(f"Hit@5      : {summary['hit@5']:.0%}")
    print(f"Results    -> {cfg.results_dir}/eval_results.json")


if __name__ == "__main__":
    main()
