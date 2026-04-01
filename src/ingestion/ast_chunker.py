"""
AST-aware chunker for Paradox Clausewitz/Jomini script blocks.

Uses tree-sitter-paradox to walk the syntax tree and group assignments
into semantically coherent chunks rather than splitting mid-statement.

Install: pip install "tree-sitter~=0.24" tree-sitter-paradox
"""

import logging
from typing import Optional

import tiktoken

log = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")
_parser = None


def _get_parser():
    global _parser
    if _parser is None:
        try:
            import tree_sitter_paradox as tsp
            from tree_sitter import Language, Parser
            _parser = Parser(Language(tsp.language()))
        except ImportError as e:
            raise ImportError(
                "tree-sitter-paradox not installed. "
                "Run: pip install \"tree-sitter~=0.24\" tree-sitter-paradox"
            ) from e
    return _parser


def _tok(text: str) -> int:
    return len(_enc.encode(text))


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _statements(node) -> list:
    """Direct statement children of a map or source_file node."""
    return [c for c in node.children if c.type == "statement"]


def _find_inner_map(stmt_node) -> Optional[object]:
    """
    Return the splittable map node inside a statement.
    Handles both regular assignments (option = { ... })
    and anonymous conditional blocks ({ limit = { ... } ... }).
    """
    for child in stmt_node.children:
        if child.type in ("assignment", "condition_statement", "logical_statement"):
            for grandchild in child.children:
                if grandchild.type == "map":
                    return grandchild
    return None


def _greedy_group(
    statements: list,
    src: bytes,
    chunk_size: int,
    context_prefix: str,
) -> list[str]:
    """
    Greedily pack consecutive statements into chunks of at most chunk_size tokens.
    Oversized single statements are recursively expanded into their map children.
    context_prefix is prepended to every produced chunk.
    """
    chunks: list[str] = []
    bucket: list[str] = []
    bucket_tokens = 0

    def flush():
        if bucket:
            chunks.append(context_prefix + "\n".join(bucket))
            bucket.clear()

    for stmt in statements:
        stmt_text = _text(stmt, src)
        stmt_tok = _tok(stmt_text)

        if stmt_tok > chunk_size:
            # Statement is too big to fit in a chunk on its own.
            # Flush the current bucket first, then try to recurse.
            flush()
            bucket_tokens = 0

            inner_map = _find_inner_map(stmt)
            if inner_map:
                # Build a sub-context: everything up to (and including) the opening `{`
                brace_offset = inner_map.start_byte
                header = src[stmt.start_byte:brace_offset].decode("utf-8", errors="replace").rstrip()
                sub_prefix = context_prefix + header + " {\n"

                sub_stmts = _statements(inner_map)
                if sub_stmts:
                    sub_chunks = _greedy_group(sub_stmts, src, chunk_size, sub_prefix)
                    if sub_chunks:
                        sub_chunks[-1] += "\n}"  # close the block on the last sub-chunk
                    chunks.extend(sub_chunks)
                else:
                    # Empty map — emit the whole statement as-is
                    chunks.append(context_prefix + stmt_text)
            else:
                # Leaf node with no inner map — can't split further, emit as-is
                chunks.append(context_prefix + stmt_text)
        else:
            # Normal statement: add to bucket or flush and start a new one
            if bucket and bucket_tokens + stmt_tok > chunk_size:
                flush()
                bucket_tokens = 0
            bucket.append(stmt_text)
            bucket_tokens += stmt_tok

    flush()
    return chunks


def ast_chunk_game_block(content: str, chunk_size: int = 256) -> list[str]:
    """
    Split a single Clausewitz parent block into AST-aware child chunks.

    - If the block fits within chunk_size, returns [content] unchanged.
    - If tree-sitter-paradox is unavailable or parsing fails, returns [content]
      (the caller can fall back to SentenceSplitter separately).
    - Otherwise walks the map body and groups assignments greedily, recursing
      into oversized sub-blocks.

    Each chunk is prefixed with `# <block_name>` so it is self-contained when
    retrieved in isolation.
    """
    if _tok(content) <= chunk_size:
        return [content]

    try:
        parser = _get_parser()
    except ImportError:
        log.debug("tree-sitter-paradox unavailable — single-chunk fallback")
        return [content]

    try:
        src = content.encode("utf-8")
        tree = parser.parse(src)
        root = tree.root_node

        top = _statements(root)
        if not top:
            return [content]

        # Some declarations like `scripted_effect foo = { ... }` produce a
        # keyword statement ("scripted_effect") followed by the real assignment.
        # Find the first statement that actually contains a map body.
        outer_map = None
        assignment_stmt_idx = 0
        for idx, stmt in enumerate(top):
            m = _find_inner_map(stmt)
            if m is not None:
                outer_map = m
                assignment_stmt_idx = idx
                break

        if outer_map is None:
            return [content]

        inner_stmts = _statements(outer_map)
        if not inner_stmts:
            return [content]

        # Build block name: concatenate any keyword-only statements that precede
        # the assignment, then take the assignment's own identifier.
        # e.g. "scripted_effect birth_9002_name_setting_effect"
        prefix_parts = [_text(top[i], src).strip() for i in range(assignment_stmt_idx)]
        assign_stmt = top[assignment_stmt_idx]
        for child in assign_stmt.children:
            if child.type == "assignment":
                for gc in child.children:
                    if gc.type == "identifier":
                        prefix_parts.append(_text(gc, src))
                        break
                break
        block_name = " ".join(prefix_parts)

        prefix = f"# {block_name}\n" if block_name else ""
        chunks = _greedy_group(inner_stmts, src, chunk_size, context_prefix=prefix)
        return chunks if chunks else [content]

    except Exception as exc:
        log.warning("AST chunking failed (%s) — single-chunk fallback", exc)
        return [content]
