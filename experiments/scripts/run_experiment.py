"""
Run an indexing experiment with a given YAML config.

Usage:
    py experiments/scripts/run_experiment.py experiments/configs/baseline.yaml

Creates a separate ChromaDB collection and stores BM25/parents under
experiments/results/<name>/.
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path

# project root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chromadb
import tiktoken
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from rank_bm25 import BM25Okapi

from src.config import CHROMA_DIR
from src.ingestion.clausewitz_parser import parse_game_file
from src.ingestion.experiment_config import ExperimentConfig
from src.ingestion.wiki_parser import parse_wiki_file
from src.retrieval.embeddings import QwenOpenRouterEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, cfg: ExperimentConfig) -> list[str]:
    splitter = SentenceSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        tokenizer=_enc.encode,
    )
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    return [n.get_content() for n in nodes]


def build_child_nodes(parents: list[dict], cfg: ExperimentConfig) -> list[dict]:
    child_nodes: list[dict] = []
    child_idx = 0
    for parent in parents:
        for chunk in chunk_text(parent["content"], cfg):
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
    log.info("BM25 saved → %s (%d docs)", path, len(nodes))


def save_parents(parents: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({p["parent_id"]: p for p in parents}, f, ensure_ascii=False)
    log.info("Parents saved → %s (%d)", path, len(parents))


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
        log.info("[%s] %d child chunks → embedding...", label, len(child_nodes))

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
    if len(sys.argv) < 2:
        print("Usage: py experiments/scripts/run_experiment.py <config.yaml>")
        sys.exit(1)
    cfg = ExperimentConfig.from_yaml(sys.argv[1])
    run(cfg)
