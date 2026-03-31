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

EVAL_QUERIES_PATH = Path(__file__).parent.parent / "eval_queries.yaml"


def load_queries() -> list[dict]:
    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["queries"]


def is_hit(result: dict, expected: list[str]) -> bool:
    haystack = (result.get("text", "") + " " + result.get("block_name", "") +
                " " + result.get("section_title", "")).lower()
    return any(e.lower() in haystack for e in expected)


def evaluate(cfg: ExperimentConfig, top_k: int = 5, verbose: bool = False) -> dict:
    queries = load_queries()
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Load both indexes
    embed_game = QwenOpenRouterEmbedding()
    embed_wiki = QwenOpenRouterEmbedding()

    indexes = {}
    for label, collection_name, bm25_path, parents_path in [
        ("game", cfg.game_collection, cfg.bm25_game_path, cfg.parents_game_path),
        ("wiki", cfg.wiki_collection, cfg.bm25_wiki_path, cfg.parents_wiki_path),
    ]:
        if not bm25_path.exists():
            print(f"[{label}] Not indexed yet — run run_experiment.py first")
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
        results = index.query(
            query_text=q["query"],
            top_k=top_k,
            alpha=cfg.alpha,
            filter_category=q.get("category"),
        )

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
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    summary = evaluate(cfg, top_k=args.top_k, verbose=args.verbose)

    print(f"\n{'='*40}")
    print(f"Experiment : {summary['experiment']}")
    print(f"Queries    : {summary['total_queries']}")
    print(f"Hit@1      : {summary['hit@1']:.0%}")
    print(f"Hit@3      : {summary['hit@3']:.0%}")
    print(f"Hit@5      : {summary['hit@5']:.0%}")
    print(f"Results    → {cfg.results_dir}/eval_results.json")


if __name__ == "__main__":
    main()
