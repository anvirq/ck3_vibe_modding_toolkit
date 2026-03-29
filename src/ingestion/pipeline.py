"""
Indexing pipeline.

Run once (or re-run after data updates):
    python -m src.ingestion.pipeline

What it does
------------
1. Walk game files (common/, events/, gui/) and wiki/ markdown files.
2. Parse into parent documents using domain-specific parsers.
3. Split each parent into 256-token child chunks (50-token overlap).
4. Embed child chunks via Qwen3/OpenRouter and store in ChromaDB.
5. Build + persist a BM25 index for each collection.
6. Persist parent texts to storage/parents/ for retrieval post-processing.
"""

import json
import pickle
import time
import logging
from pathlib import Path

import tiktoken
from llama_index.core.node_parser import SentenceSplitter
import chromadb

from src.config import (
    GAME_DATA_DIR,
    WIKI_DATA_DIR,
    STORAGE_DIR,
    CHROMA_DIR,
    BM25_DIR,
    PARENTS_DIR,
    GAME_DIRS_TO_INDEX,
    GAME_FILE_EXTENSIONS,
    GAME_COLLECTION,
    WIKI_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)
from src.ingestion.clausewitz_parser import parse_game_file
from src.ingestion.wiki_parser import parse_wiki_file
from src.retrieval.embeddings import QwenOpenRouterEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# tiktoken tokenizer for chunk size measurement
_enc = tiktoken.get_encoding("cl100k_base")


# ── helpers ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into token-bounded chunks using LlamaIndex SentenceSplitter."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        tokenizer=_enc.encode,
    )
    # SentenceSplitter works on Document/Node; use get_nodes_from_documents
    from llama_index.core import Document
    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])
    return [n.get_content() for n in nodes]


def _collect_game_files() -> list[Path]:
    files = []
    for subdir in GAME_DIRS_TO_INDEX:
        base = GAME_DATA_DIR / subdir
        if not base.exists():
            log.warning("Directory not found: %s", base)
            continue
        for ext in GAME_FILE_EXTENSIONS:
            files.extend(base.rglob(f"*{ext}"))
    return files


def _collect_wiki_files() -> list[Path]:
    if not WIKI_DATA_DIR.exists():
        return []
    return list(WIKI_DATA_DIR.glob("*.md"))


# ── embedding helpers ─────────────────────────────────────────────────────────

def _embed_texts(embed_model: QwenOpenRouterEmbedding, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, logging progress."""
    return embed_model._get_text_embeddings(texts)


# ── ChromaDB upsert ───────────────────────────────────────────────────────────

def _upsert_to_chroma(
    collection: chromadb.Collection,
    child_nodes: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Batch-upsert child chunks into a ChromaDB collection."""
    BATCH = 500
    for i in range(0, len(child_nodes), BATCH):
        batch_nodes = child_nodes[i : i + BATCH]
        batch_embs = embeddings[i : i + BATCH]
        collection.upsert(
            ids=[n["child_id"] for n in batch_nodes],
            documents=[n["text"] for n in batch_nodes],
            metadatas=[n["metadata"] for n in batch_nodes],
            embeddings=batch_embs,
        )


# ── BM25 index ────────────────────────────────────────────────────────────────

def _build_and_save_bm25(child_nodes: list[dict], path: Path) -> None:
    """Build a rank-bm25 index and pickle it together with node metadata."""
    from rank_bm25 import BM25Okapi

    corpus = [n["text"].lower().split() for n in child_nodes]
    bm25 = BM25Okapi(corpus)
    payload = {
        "bm25": bm25,
        "nodes": [{"child_id": n["child_id"], "text": n["text"], "metadata": n["metadata"]}
                  for n in child_nodes],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    log.info("BM25 index saved → %s (%d docs)", path, len(child_nodes))


# ── parent store ──────────────────────────────────────────────────────────────

def _save_parents(parents: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_map = {p["parent_id"]: p for p in parents}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(parent_map, f, ensure_ascii=False)
    log.info("Parent store saved → %s (%d entries)", path, len(parent_map))


# ── main pipeline ─────────────────────────────────────────────────────────────

def _process_collection(
    name: str,
    parents: list[dict],
    chroma_collection: chromadb.Collection,
    embed_model: QwenOpenRouterEmbedding,
    bm25_path: Path,
    parents_path: Path,
) -> None:
    log.info("[%s] %d parent docs → chunking...", name, len(parents))

    child_nodes: list[dict] = []
    child_idx = 0

    for parent in parents:
        chunks = _chunk_text(parent["content"])
        for chunk_text in chunks:
            if not chunk_text.strip():
                continue
            meta = {
                "parent_id": parent["parent_id"],
                "source_type": parent["source_type"],
                "file_path": parent["file_path"],
            }
            # Add optional fields present only in some source types
            for key in ("file_category", "block_name", "section_title"):
                if key in parent:
                    meta[key] = parent[key]

            child_nodes.append(
                {
                    "child_id": f"{parent['parent_id']}::chunk{child_idx}",
                    "text": chunk_text,
                    "metadata": meta,
                }
            )
            child_idx += 1

    # ── skip already-embedded chunks (resume after crash) ────────────────────
    existing = chroma_collection.get(include=[])
    existing_ids = set(existing["ids"])
    pending_nodes = [n for n in child_nodes if n["child_id"] not in existing_ids]
    log.info(
        "[%s] %d child chunks total — %d already embedded, %d pending",
        name, len(child_nodes), len(existing_ids), len(pending_nodes),
    )

    # ── embed + upsert in checkpoint batches ─────────────────────────────────
    CHECKPOINT = 200
    t0 = time.time()
    for i in range(0, len(pending_nodes), CHECKPOINT):
        batch = pending_nodes[i : i + CHECKPOINT]
        texts = [n["text"] for n in batch]
        embeddings = _embed_texts(embed_model, texts)
        _upsert_to_chroma(chroma_collection, batch, embeddings)
        done = min(i + CHECKPOINT, len(pending_nodes))
        log.info("[%s] Progress: %d/%d (%.0fs)", name, done, len(pending_nodes), time.time() - t0)

    log.info("[%s] Building BM25 index...", name)
    _build_and_save_bm25(child_nodes, bm25_path)

    log.info("[%s] Saving parent store...", name)
    _save_parents(parents, parents_path)

    log.info("[%s] Done.", name)


def run() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    embed_model = QwenOpenRouterEmbedding()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # ── GAME FILES ────────────────────────────────────────────────────────────
    log.info("Collecting game files...")
    game_files = _collect_game_files()
    log.info("Found %d game files", len(game_files))

    game_parents: list[dict] = []
    for path in game_files:
        game_parents.extend(parse_game_file(path))
    log.info("Parsed %d game parent blocks", len(game_parents))

    game_collection = chroma_client.get_or_create_collection(
        name=GAME_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    _process_collection(
        name="game",
        parents=game_parents,
        chroma_collection=game_collection,
        embed_model=embed_model,
        bm25_path=BM25_DIR / "game.pkl",
        parents_path=PARENTS_DIR / "game.json",
    )

    # ── WIKI ──────────────────────────────────────────────────────────────────
    log.info("Collecting wiki files...")
    wiki_files = _collect_wiki_files()
    log.info("Found %d wiki files", len(wiki_files))

    wiki_parents: list[dict] = []
    for path in wiki_files:
        wiki_parents.extend(parse_wiki_file(path))
    log.info("Parsed %d wiki parent sections", len(wiki_parents))

    wiki_collection = chroma_client.get_or_create_collection(
        name=WIKI_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    _process_collection(
        name="wiki",
        parents=wiki_parents,
        chroma_collection=wiki_collection,
        embed_model=embed_model,
        bm25_path=BM25_DIR / "wiki.pkl",
        parents_path=PARENTS_DIR / "wiki.json",
    )

    log.info("Pipeline complete.")


if __name__ == "__main__":
    run()
