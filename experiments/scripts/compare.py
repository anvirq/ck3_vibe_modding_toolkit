"""
Compare two experiment results side by side.

Usage:
    py experiments/scripts/compare.py experiments/configs/baseline.yaml experiments/configs/bm25_heavy.yaml
    py experiments/scripts/compare.py experiments/configs/baseline.yaml experiments/configs/sentence_128.yaml --verbose
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from src.ingestion.experiment_config import ExperimentConfig


def load_results(cfg: ExperimentConfig) -> dict:
    path = cfg.results_dir / "eval_results.json"
    if not path.exists():
        print(f"[!] No results for '{cfg.name}' - run evaluate.py first: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_a", help="First experiment YAML config")
    parser.add_argument("config_b", help="Second experiment YAML config")
    parser.add_argument("--verbose", action="store_true", help="Show per-query breakdown")
    args = parser.parse_args()

    cfg_a = ExperimentConfig.from_yaml(args.config_a)
    cfg_b = ExperimentConfig.from_yaml(args.config_b)

    data_a = load_results(cfg_a)
    data_b = load_results(cfg_b)

    sum_a = data_a["summary"]
    sum_b = data_b["summary"]

    # Header
    col = 22
    name_a = cfg_a.name[:col].ljust(col)
    name_b = cfg_b.name[:col].ljust(col)

    print(f"\n{'='*60}")
    print(f"{'Metric':<12}  {name_a}  {name_b}  {'Delta':>8}")
    print(f"{'-'*60}")

    for metric in ("hit@1", "hit@3", "hit@5"):
        va = sum_a[metric]
        vb = sum_b[metric]
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<12}  {va:.0%}{'':<{col-4}}  {vb:.0%}{'':<{col-4}}  {sign}{delta:.0%}")

    total_a = sum_a["total_queries"]
    total_b = sum_b["total_queries"]
    print(f"{'queries':<12}  {total_a:<{col}}  {total_b:<{col}}")
    print(f"{'='*60}")

    if not args.verbose:
        return

    # Per-query comparison
    queries_a = {q["id"]: q for q in data_a["queries"]}
    queries_b = {q["id"]: q for q in data_b["queries"]}
    all_ids = sorted(set(queries_a) | set(queries_b))

    print(f"\n{'ID':<6}  {'Query':<40}  {cfg_a.name[:12]:<14}  {cfg_b.name[:12]:<14}  Note")
    print(f"{'-'*100}")

    regressions = []
    improvements = []

    for qid in all_ids:
        qa = queries_a.get(qid)
        qb = queries_b.get(qid)

        if qa is None or qb is None:
            continue

        def grade(q):
            if q["hit@1"]: return "HIT@1"
            if q["hit@3"]: return "HIT@3"
            if q["hit@5"]: return "HIT@5"
            return "MISS "

        ga = grade(qa)
        gb = grade(qb)

        note = ""
        if ga != gb:
            hit_rank = ["HIT@1", "HIT@3", "HIT@5", "MISS "]
            if hit_rank.index(gb) < hit_rank.index(ga):
                note = "<-- B better"
                improvements.append(qid)
            else:
                note = "<-- A better"
                regressions.append(qid)

        query_short = qa["query"][:38]
        print(f"{qid:<6}  {query_short:<40}  {ga:<14}  {gb:<14}  {note}")

    print(f"\nB better on : {improvements if improvements else '(none)'}")
    print(f"A better on : {regressions if regressions else '(none)'}")


if __name__ == "__main__":
    main()
