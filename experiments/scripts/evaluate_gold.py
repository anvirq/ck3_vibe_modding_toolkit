"""
Evaluate experiment using stricter gold rules.

Usage:
    py experiments/scripts/evaluate_gold.py experiments/configs/baseline.yaml
    py experiments/scripts/evaluate_gold.py experiments/configs/ast_256.yaml --query-subset narrow --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chromadb
import yaml

from src.config import CHROMA_DIR
from src.ingestion.experiment_config import ExperimentConfig
from src.retrieval.embeddings import QwenOpenRouterEmbedding
from src.retrieval.hybrid_search import HybridSearchIndex

from experiments.scripts.evaluate import load_queries
from experiments.scripts.rerankers import CrossEncoderReranker

GOLD_PATH = Path(__file__).parent.parent / "gold_answers.yaml"


def load_gold() -> dict[str, dict]:
    with open(GOLD_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f).get("gold", {})


def is_gold_hit_strict(result: dict, rule: dict) -> bool:
    path = result.get("file_path", "").replace("\\", "/").lower()
    hay = (
        result.get("text", "")
        + " "
        + result.get("block_name", "")
        + " "
        + result.get("section_title", "")
    ).lower()

    path_needles = [x.lower().replace("\\", "/") for x in rule.get("file_path_contains", [])]
    if path_needles and not any(needle in path for needle in path_needles):
        return False

    required = [x.lower() for x in rule.get("must_contain_all", [])]
    return all(token in hay for token in required)


def is_gold_hit_soft(result: dict, rule: dict) -> bool:
    """
    Softer match:
    - ignores file path constraint
    - requires ANY token from must_contain_all to appear
    """
    hay = (
        result.get("text", "")
        + " "
        + result.get("block_name", "")
        + " "
        + result.get("section_title", "")
    ).lower()
    required = [x.lower() for x in rule.get("must_contain_all", [])]
    if not required:
        return False
    return any(token in hay for token in required)


def rerank_gold_rule(results: list[dict], rule: dict) -> list[dict]:
    """
    Diagnostic reranker for eval: prioritize candidates that satisfy more gold constraints.
    Keeps original retrieval score as secondary key.
    """
    path_needles = [x.lower().replace("\\", "/") for x in rule.get("file_path_contains", [])]
    required = [x.lower() for x in rule.get("must_contain_all", [])]

    def score_row(r: dict) -> tuple[int, int, float]:
        path = r.get("file_path", "").replace("\\", "/").lower()
        hay = (
            r.get("text", "")
            + " "
            + r.get("block_name", "")
            + " "
            + r.get("section_title", "")
        ).lower()
        path_score = 1 if (not path_needles or any(n in path for n in path_needles)) else 0
        token_hits = sum(1 for tok in required if tok in hay)
        retrieval_score = float(r.get("score", 0.0))
        return (path_score, token_hits, retrieval_score)

    return sorted(results, key=score_row, reverse=True)


def evaluate_gold(
    cfg: ExperimentConfig,
    top_k: int,
    query_subset: str | None,
    verbose: bool,
    mode: str,
    rerank_with_gold_rule_enabled: bool,
    retrieval_rerank: bool,
    cross_encoder_model: str | None,
    candidate_k: int | None,
) -> dict:
    queries = load_queries(query_subset)
    gold = load_gold()
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embed_game = QwenOpenRouterEmbedding()
    embed_wiki = QwenOpenRouterEmbedding()
    indexes: dict[str, HybridSearchIndex] = {}
    ce_reranker = CrossEncoderReranker(cross_encoder_model) if cross_encoder_model else None

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

    total = 0
    hits_at_soft = {1: 0, 3: 0, 5: 0}
    hits_at_strict = {1: 0, 3: 0, 5: 0}
    rows: list[dict] = []

    for q in queries:
        qid = q["id"]
        coll = q["collection"]
        if coll not in indexes or qid not in gold:
            continue

        rule = gold[qid]
        retrieval_k = candidate_k if candidate_k and candidate_k > top_k else top_k
        results = indexes[coll].query(
            query_text=q["query"],
            top_k=retrieval_k,
            alpha=cfg.alpha,
            filter_category=q.get("category"),
            rerank=retrieval_rerank,
        )
        if ce_reranker:
            results = ce_reranker.rerank(q["query"], results)
        results = results[:top_k]
        if rerank_with_gold_rule_enabled:
            results = rerank_gold_rule(results, rule)

        hit1_soft = any(is_gold_hit_soft(r, rule) for r in results[:1])
        hit3_soft = any(is_gold_hit_soft(r, rule) for r in results[:3])
        hit5_soft = any(is_gold_hit_soft(r, rule) for r in results[:5])

        hit1_strict = any(is_gold_hit_strict(r, rule) for r in results[:1])
        hit3_strict = any(is_gold_hit_strict(r, rule) for r in results[:3])
        hit5_strict = any(is_gold_hit_strict(r, rule) for r in results[:5])

        if hit1_soft:
            hits_at_soft[1] += 1
        if hit3_soft:
            hits_at_soft[3] += 1
        if hit5_soft:
            hits_at_soft[5] += 1

        if hit1_strict:
            hits_at_strict[1] += 1
        if hit3_strict:
            hits_at_strict[3] += 1
        if hit5_strict:
            hits_at_strict[5] += 1
        total += 1

        row = {
            "id": qid,
            "query": q["query"],
            "soft_hit@1": hit1_soft,
            "soft_hit@3": hit3_soft,
            "soft_hit@5": hit5_soft,
            "strict_hit@1": hit1_strict,
            "strict_hit@3": hit3_strict,
            "strict_hit@5": hit5_strict,
            "top_score": results[0].get("score", 0) if results else 0,
            "top_file": results[0].get("file_path", "") if results else "",
        }
        rows.append(row)
        if verbose:
            soft_grade = "HIT@1" if hit1_soft else ("HIT@3" if hit3_soft else ("HIT@5" if hit5_soft else "MISS"))
            strict_grade = "HIT@1" if hit1_strict else ("HIT@3" if hit3_strict else ("HIT@5" if hit5_strict else "MISS"))
            print(f"[{qid}] soft={soft_grade} strict={strict_grade} score={row['top_score']:.2f} {q['query']!r}")

    summary_soft = {
        "experiment": cfg.name,
        "metric": "gold_soft",
        "rerank_gold_rule": rerank_with_gold_rule_enabled,
        "total_queries": total,
        "hit@1": hits_at_soft[1] / total if total else 0,
        "hit@3": hits_at_soft[3] / total if total else 0,
        "hit@5": hits_at_soft[5] / total if total else 0,
    }
    summary_strict = {
        "experiment": cfg.name,
        "metric": "gold_strict",
        "rerank_gold_rule": rerank_with_gold_rule_enabled,
        "total_queries": total,
        "hit@1": hits_at_strict[1] / total if total else 0,
        "hit@3": hits_at_strict[3] / total if total else 0,
        "hit@5": hits_at_strict[5] / total if total else 0,
    }

    subset_name = query_subset or "all"
    suffix = mode if mode in {"soft", "strict"} else "both"
    out_path = cfg.results_dir / f"eval_gold_{suffix}_{subset_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary_soft": summary_soft, "summary_strict": summary_strict, "queries": rows},
            f,
            indent=2,
            ensure_ascii=False,
        )

    if mode == "soft":
        return {"selected": summary_soft, "soft": summary_soft, "strict": summary_strict}
    if mode == "strict":
        return {"selected": summary_strict, "soft": summary_soft, "strict": summary_strict}
    return {"selected": summary_strict, "soft": summary_soft, "strict": summary_strict}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-subset", help="Subset name or explicit .txt path")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["soft", "strict", "both"],
        default="both",
        help="Scoring mode: soft, strict, or both (default).",
    )
    parser.add_argument(
        "--rerank-gold-rule",
        action="store_true",
        help="Diagnostic: re-rank retrieved top-k using gold-rule match score before metrics.",
    )
    parser.add_argument(
        "--retrieval-rerank",
        action="store_true",
        help="Apply real query-aware reranking in retrieval before scoring.",
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
    summary = evaluate_gold(
        cfg=cfg,
        top_k=args.top_k,
        query_subset=args.query_subset,
        verbose=args.verbose,
        mode=args.mode,
        rerank_with_gold_rule_enabled=args.rerank_gold_rule,
        retrieval_rerank=args.retrieval_rerank,
        cross_encoder_model=args.cross_encoder_model,
        candidate_k=args.candidate_k,
    )
    print(f"\n{'='*40}")
    print(f"Experiment : {summary['selected']['experiment']}")
    print(f"Queries    : {summary['selected']['total_queries']}")
    print(f"Soft Hit@1 : {summary['soft']['hit@1']:.0%}")
    print(f"Soft Hit@3 : {summary['soft']['hit@3']:.0%}")
    print(f"Soft Hit@5 : {summary['soft']['hit@5']:.0%}")
    print(f"Strict@1   : {summary['strict']['hit@1']:.0%}")
    print(f"Strict@3   : {summary['strict']['hit@3']:.0%}")
    print(f"Strict@5   : {summary['strict']['hit@5']:.0%}")


if __name__ == "__main__":
    main()
