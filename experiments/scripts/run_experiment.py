"""
Run an indexing experiment with a given YAML config.

Usage:
    py experiments/scripts/run_experiment.py experiments/configs/baseline.yaml
    py experiments/scripts/run_experiment.py experiments/configs/baseline.yaml --subset narrow
    py experiments/scripts/run_experiment.py experiments/configs/narrow_baseline.yaml --game-subset narrow --wiki-subset narrow

Creates a separate ChromaDB collection and stores BM25/parents under
experiments/results/<name>/.
"""

import json
import logging
import pickle
import sys
import time
import argparse
from pathlib import Path

# project root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chromadb
from rank_bm25 import BM25Okapi

from src.config import CHROMA_DIR
from src.ingestion.chunking import chunk_parent
from src.ingestion.clausewitz_parser import parse_game_file
from src.ingestion.experiment_config import ExperimentConfig
from src.ingestion.wiki_parser import parse_wiki_file
from src.retrieval.embeddings import QwenOpenRouterEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SUBSETS_GAME_DIR = Path(__file__).parent.parent / "subsets" / "game"
SUBSETS_WIKI_DIR = Path(__file__).parent.parent / "subsets" / "wiki"


def load_subset_file(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
    return [line for line in lines if line]


def resolve_subset_arg(arg: str, subset_dir: Path, label: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    if p.suffix:
        raise FileNotFoundError(f"{label} subset file not found: {arg}")
    candidate = subset_dir / f"{arg}.txt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unknown {label} subset '{arg}' (expected: {candidate})")


def resolve_named_subset(name: str) -> tuple[Path | None, Path | None]:
    game_path = SUBSETS_GAME_DIR / f"{name}.txt"
    wiki_path = SUBSETS_WIKI_DIR / f"{name}.txt"
    game_found = game_path if game_path.exists() else None
    wiki_found = wiki_path if wiki_path.exists() else None
    if not game_found and not wiki_found:
        raise FileNotFoundError(
            f"Unknown subset '{name}' (checked {game_path} and {wiki_path})"
        )
    return game_found, wiki_found


def build_child_nodes(parents: list[dict], cfg: ExperimentConfig) -> list[dict]:
    child_nodes: list[dict] = []
    child_idx = 0
    for parent in parents:
        for chunk in chunk_parent(
            parent, cfg.chunking_strategy, cfg.chunk_size, cfg.chunk_overlap
        ):
            if not chunk.strip():
                continue
            meta = {
                "parent_id": parent["parent_id"],
                "source_type": parent["source_type"],
                "file_path": parent["file_path"],
            }
            for key in ("file_category", "block_name", "section_title"):
                if key in parent:
                    meta[key] = parent[key]
            child_nodes.append({
                "child_id": f"{parent['parent_id']}::chunk{child_idx}",
                "text": chunk,
                "metadata": meta,
            })
            child_idx += 1
    return child_nodes


def upsert_chroma(collection: chromadb.Collection, nodes: list[dict],
                  embeddings: list[list[float]]) -> None:
    BATCH = 500
    for i in range(0, len(nodes), BATCH):
        batch = nodes[i:i + BATCH]
        collection.upsert(
            ids=[n["child_id"] for n in batch],
            documents=[n["text"] for n in batch],
            metadatas=[n["metadata"] for n in batch],
            embeddings=embeddings[i:i + BATCH],
        )


def save_bm25(nodes: list[dict], path: Path) -> None:
    corpus = [n["text"].lower().split() for n in nodes]
    bm25 = BM25Okapi(corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "nodes": nodes}, f)
    log.info("BM25 saved -> %s (%d docs)", path, len(nodes))


def save_parents(parents: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({p["parent_id"]: p for p in parents}, f, ensure_ascii=False)
    log.info("Parents saved -> %s (%d)", path, len(parents))


def run(cfg: ExperimentConfig) -> None:
    log.info("=== Experiment: %s ===", cfg.name)
    log.info("Strategy=%s  chunk=%d  overlap=%d  alpha=%.2f",
             cfg.chunking_strategy, cfg.chunk_size, cfg.chunk_overlap, cfg.alpha)

    embed = QwenOpenRouterEmbedding()
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    for label, get_files, parser, collection_name, bm25_path, parents_path in [
        (
            "game",
            cfg.game_file_paths,
            parse_game_file,
            cfg.game_collection,
            cfg.bm25_game_path,
            cfg.parents_game_path,
        ),
        (
            "wiki",
            cfg.wiki_file_paths,
            parse_wiki_file,
            cfg.wiki_collection,
            cfg.bm25_wiki_path,
            cfg.parents_wiki_path,
        ),
    ]:
        files = get_files()
        log.info("[%s] %d files", label, len(files))

        parents: list[dict] = []
        for f in files:
            parents.extend(parser(f))
        log.info("[%s] %d parent blocks", label, len(parents))

        child_nodes = build_child_nodes(parents, cfg)
        log.info("[%s] %d child chunks -> embedding...", label, len(child_nodes))

        if not child_nodes:
            log.warning("[%s] 0 child chunks; skipping embeddings and BM25", label)
            save_parents(parents, parents_path)
            continue

        collection = chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Checkpointing
        existing = set(collection.get(include=[])["ids"])
        pending = [n for n in child_nodes if n["child_id"] not in existing]
        log.info("[%s] %d already indexed, %d pending", label, len(existing), len(pending))

        t0 = time.time()
        BATCH = 200
        for i in range(0, len(pending), BATCH):
            batch = pending[i:i + BATCH]
            embs = embed._get_text_embeddings([n["text"] for n in batch])
            upsert_chroma(collection, batch, embs)
            log.info("[%s] %d/%d (%.0fs)", label, min(i + BATCH, len(pending)), len(pending), time.time() - t0)

        save_bm25(child_nodes, bm25_path)
        save_parents(parents, parents_path)

    log.info("Experiment '%s' complete.", cfg.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument(
        "--subset",
        help="Subset name. If found in subsets/game and/or subsets/wiki, applies to those sources automatically.",
    )
    parser.add_argument(
        "--game-subset",
        help="Subset name (from experiments/subsets/game/<name>.txt) or explicit .txt path",
    )
    parser.add_argument(
        "--wiki-subset",
        help="Subset name (from experiments/subsets/wiki/<name>.txt) or explicit .txt path",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.subset:
        game_subset_path, wiki_subset_path = resolve_named_subset(args.subset)
        if game_subset_path:
            cfg.game_files = load_subset_file(game_subset_path)
            log.info("Using game subset: %s (%d files)", game_subset_path, len(cfg.game_files))
        if wiki_subset_path:
            cfg.wiki_files = load_subset_file(wiki_subset_path)
            log.info("Using wiki subset: %s (%d files)", wiki_subset_path, len(cfg.wiki_files))

    if args.game_subset:
        subset_path = resolve_subset_arg(args.game_subset, SUBSETS_GAME_DIR, "game")
        cfg.game_files = load_subset_file(subset_path)
        log.info("Using game subset: %s (%d files)", subset_path, len(cfg.game_files))
    if args.wiki_subset:
        subset_path = resolve_subset_arg(args.wiki_subset, SUBSETS_WIKI_DIR, "wiki")
        cfg.wiki_files = load_subset_file(subset_path)
        log.info("Using wiki subset: %s (%d files)", subset_path, len(cfg.wiki_files))

    run(cfg)
