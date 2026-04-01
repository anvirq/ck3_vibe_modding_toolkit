"""
Evaluate the field-feedback query suite (Top1, Top3, noise, actionability, latency).

Usage:
    py experiments/scripts/evaluate_feedback.py experiments/configs/feedback_profile_a.yaml --top-k 5

Requires the experiment index to exist (run_experiment.py for the same config first).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chromadb
import yaml

from src.config import CHROMA_DIR
from src.ingestion.experiment_config import ExperimentConfig
from src.retrieval.embeddings import QwenOpenRouterEmbedding
from src.retrieval.hybrid_search import HybridSearchIndex
from experiments.scripts.rerankers import CrossEncoderReranker

QUERIES_PATH = Path(__file__).parent.parent / "feedback_queries.yaml"
GOLD_PATH = Path(__file__).parent.parent / "feedback_gold.yaml"


def _norm(s: str) -> str:
    return s.replace("\\", "/").lower()


def _path_relevant(fp: str, needles: list[str]) -> bool:
    p = _norm(fp)
    return any(n.lower() in p for n in needles)


def _text_hay(r: dict) -> str:
    return (
        (r.get("text") or "")
        + " "
        + (r.get("block_name") or "")
        + " "
        + (r.get("section_title") or "")
    ).lower()


def _all_in_text(hay: str, parts: list[str]) -> bool:
    return all(p.lower() in hay for p in parts)


def _any_in_text(hay: str, parts: list[str]) -> bool:
    return any(p.lower() in hay for p in parts)


def _actionable(r: dict) -> bool:
    fp = _norm(r.get("file_path", ""))
    if "wiki" in fp or fp.endswith(".md"):
        return bool(r.get("section_title"))
    if not fp:
        return False
    t = r.get("text") or ""
    return bool(r.get("block_name")) or ("=" in t and "{" in t)


def evaluate_profile(
    cfg: ExperimentConfig,
    top_k: int,
    rerank: bool = False,
    cross_encoder_model: str | None = None,
    candidate_k: int | None = None,
) -> dict:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]
    with open(GOLD_PATH, encoding="utf-8") as f:
        gold_map = yaml.safe_load(f).get("gold", {})

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

    rows: list[dict] = []
    scorable_top1 = []
    scorable_top3 = []
    scorable_noise = []
    scorable_action = []

    for q in queries:
        qid = q["id"]
        coll = q["collection"]
        rule = gold_map.get(qid, {})
        scorable = rule.get("scorable", True)

        t0 = time.perf_counter()
        results: list[dict] = []
        if coll in indexes:
            retrieval_k = candidate_k if candidate_k and candidate_k > top_k else top_k
            results = indexes[coll].query(
                query_text=q["query"],
                top_k=retrieval_k,
                alpha=cfg.alpha,
                filter_category=q.get("category"),
                rerank=rerank,
            )
            if ce_reranker:
                results = ce_reranker.rerank(q["query"], results)
            results = results[:top_k]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        top1 = results[0] if results else {}
        top3 = results[:3]
        top5 = results[:5]

        hay1 = _text_hay(top1)

        top1_ok = False
        top3_ok = False
        noise_rate: float | None = None
        fp_pass: bool | None = None

        if rule.get("false_positive_guard"):
            bad = False
            for r in top5:
                fp = r.get("file_path", "")
                if "wiki" in _norm(fp):
                    continue
                if "validate_mod" in _text_hay(r) and ("game" in _norm(fp) or "common" in _norm(fp)):
                    bad = True
                    break
            fp_pass = not bad
            top1_ok = fp_pass
            top3_ok = fp_pass
        elif scorable:
            acc_paths = rule.get("acceptable_path_substrings", [])
            t1_paths = rule.get("top1_path_substrings", acc_paths)

            if rule.get("top1_text_must_contain_all"):
                top1_ok = _all_in_text(hay1, rule["top1_text_must_contain_all"]) and (
                    not t1_paths or _path_relevant(top1.get("file_path", ""), t1_paths)
                )
            elif rule.get("top1_text_must_contain_any"):
                top1_ok = _any_in_text(hay1, rule["top1_text_must_contain_any"]) and (
                    not t1_paths or _path_relevant(top1.get("file_path", ""), t1_paths)
                )
            else:
                top1_ok = bool(top1) and (
                    not t1_paths or _path_relevant(top1.get("file_path", ""), t1_paths)
                )

            t3_any = rule.get("top3_text_must_contain_any")
            if t3_any:
                top3_ok = any(_any_in_text(_text_hay(r), t3_any) for r in top3)
            else:
                top3_ok = any(
                    _path_relevant(r.get("file_path", ""), acc_paths) or _any_in_text(_text_hay(r), rule.get("top1_text_must_contain_any", []))
                    for r in top3
                )

            if not rule.get("skip_noise") and acc_paths:
                rel = 0
                for r in top5:
                    fp = r.get("file_path", "")
                    if _path_relevant(fp, acc_paths) or _any_in_text(_text_hay(r), rule.get("top1_text_must_contain_any", [])):
                        rel += 1
                noise_rate = 1.0 - (rel / max(len(top5), 1))
            elif not rule.get("skip_noise"):
                noise_rate = 1.0 if top5 else 0.0
        else:
            top1_ok = False
            top3_ok = False
            noise_rate = None

        act = _actionable(top1) if top1 else False

        row = {
            "id": qid,
            "query": q["query"],
            "scorable": scorable,
            "latency_ms": round(latency_ms, 1),
            "top1_correct": bool(top1_ok),
            "top3_recall": bool(top3_ok),
            "noise_top5": noise_rate,
            "actionability": bool(act),
            "false_positive_pass": fp_pass,
            "top1_file": (top1.get("file_path", "") if top1 else "")[:120],
        }
        rows.append(row)

        if scorable:
            scorable_top1.append(top1_ok)
            scorable_top3.append(top3_ok)
            if noise_rate is not None:
                scorable_noise.append(noise_rate)
            scorable_action.append(act)

    n = len(scorable_top1)
    summary = {
        "experiment": cfg.name,
        "top_k": top_k,
        "rerank": rerank,
        "cross_encoder_model": cross_encoder_model,
        "candidate_k": candidate_k,
        "scorable_queries": n,
        "top1": sum(scorable_top1) / n if n else 0.0,
        "top3": sum(scorable_top3) / n if n else 0.0,
        "noise_top5_mean": sum(scorable_noise) / len(scorable_noise) if scorable_noise else None,
        "actionability": sum(scorable_action) / n if n else 0.0,
    }

    out = cfg.results_dir / "feedback_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "queries": rows}, f, indent=2, ensure_ascii=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Experiment YAML (e.g. feedback_profile_a.yaml)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--cross-encoder-model")
    parser.add_argument("--candidate-k", type=int)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    summary = evaluate_profile(
        cfg,
        args.top_k,
        rerank=args.rerank,
        cross_encoder_model=args.cross_encoder_model,
        candidate_k=args.candidate_k,
    )

    print(f"\nExperiment     : {summary['experiment']}")
    print(f"Scorable queries: {summary['scorable_queries']}")
    print(f"Top1           : {summary['top1']:.0%}")
    print(f"Top3           : {summary['top3']:.0%}")
    if summary["noise_top5_mean"] is not None:
        print(f"Noise (mean)   : {summary['noise_top5_mean']:.0%}")
    print(f"Actionability  : {summary['actionability']:.0%}")
    print(f"Results -> {cfg.results_dir / 'feedback_eval.json'}")


if __name__ == "__main__":
    main()
