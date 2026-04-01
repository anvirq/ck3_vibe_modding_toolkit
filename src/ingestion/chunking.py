"""
Shared chunking for the main pipeline and experiment scripts.

- ``ast``: Clausewitz game blocks use tree-sitter-aware chunks; wiki uses SentenceSplitter.
- ``sentence_splitter``: LlamaIndex SentenceSplitter for all parents (legacy baseline).
"""

import logging

import tiktoken
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from src.ingestion.ast_chunker import ast_chunk_game_block

log = logging.getLogger(__name__)
_enc = tiktoken.get_encoding("cl100k_base")


def sentence_chunk(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=_enc.encode,
    )
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    return [n.get_content() for n in nodes]


def chunk_parent(
    parent: dict,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split one parent into child chunk strings.

    AST applies only to ``source_type == "game_file"`` when ``strategy == "ast"``.
    """
    if strategy == "ast" and parent.get("source_type") == "game_file":
        chunks = ast_chunk_game_block(parent["content"], chunk_size=chunk_size)
        if len(chunks) == 1 and len(_enc.encode(chunks[0])) > chunk_size * 1.5:
            log.debug("AST fallback -> SentenceSplitter for %s", parent["parent_id"])
            return sentence_chunk(chunks[0], chunk_size, chunk_overlap)
        return chunks
    return sentence_chunk(parent["content"], chunk_size, chunk_overlap)
