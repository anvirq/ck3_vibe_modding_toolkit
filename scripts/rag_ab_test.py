"""
Run hybrid RAG search from the CLI for A/B testing query-side embedding instructions.

Must be executed from the repository root so `src` imports correctly, e.g.:

  py scripts/rag_ab_test.py --instruct off "how to add a trait"
  py scripts/rag_ab_test.py --compare --queries scripts/sample_queries.txt
  py scripts/rag_ab_test.py --show-content --content-limit 0 "trait modifier"

`EMBEDDING_QUERY_INSTRUCTION_ENABLED` is applied before importing `src` (and in --compare
each subprocess gets a fresh process with the flag set). Does not reindex.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_queries(args: argparse.Namespace) -> list[str]:
    lines: list[str] = []
    if args.queries:
        p = Path(args.queries)
        if not p.is_file():
            raise SystemExit(f"Queries file not found: {p}")
        text = p.read_text(encoding="utf-8")
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    lines.extend(args.query_strings)
    if not lines:
        raise SystemExit("Provide at least one query via positional args or --queries FILE.")
    return lines


def _format_hit(i: int, r: dict, index_kind: str) -> str:
    score = r.get("score", 0.0)
    fp = r.get("file_path", "")
    if index_kind == "wiki":
        title = r.get("section_title", "")
        return f"  [{i}] score={score:.4f}  {fp}  ::  {title}"
    cat = r.get("file_category", "")
    block = r.get("block_name", "")
    return f"  [{i}] score={score:.4f}  {cat}/{block}  ({fp})"


def _print_content_preview(content: str, limit: int) -> None:
    """Same `content` field MCP attaches to each hit (full parent document)."""
    if limit > 0 and len(content) > limit:
        body = content[:limit] + "\n... [truncated; use --content-limit 0 for full parent text]"
    else:
        body = content
    for line in body.splitlines():
        print(f"      {line}")


def _run_searches(args: argparse.Namespace) -> None:
    # Apply before any src import (dotenv in config will not override existing env keys).
    os.environ["EMBEDDING_QUERY_INSTRUCTION_ENABLED"] = (
        "true" if args.instruct == "on" else "false"
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.retrieval.hybrid_search import get_game_index, get_wiki_index

    queries = _load_queries(args)
    instruct_label = args.instruct

    for q in queries:
        print(f"\n>>> query: {q!r}")
        if args.index in ("game", "both"):
            print(f"--- game (instruct={instruct_label}) ---")
            hits = get_game_index().query(
                query_text=q,
                top_k=args.top_k,
                alpha=args.alpha,
                filter_category=args.category,
            )
            if not hits:
                print("  (no results)")
            for i, r in enumerate(hits, 1):
                print(_format_hit(i, r, "game"))
                if args.show_content:
                    print("      --- content (same as MCP tool payload) ---")
                    _print_content_preview(r.get("content") or "", args.content_limit)
        if args.index in ("wiki", "both"):
            print(f"--- wiki (instruct={instruct_label}) ---")
            hits = get_wiki_index().query(
                query_text=q,
                top_k=args.top_k,
                alpha=args.alpha,
            )
            if not hits:
                print("  (no results)")
            for i, r in enumerate(hits, 1):
                print(_format_hit(i, r, "wiki"))
                if args.show_content:
                    print("      --- content (same as MCP tool payload) ---")
                    _print_content_preview(r.get("content") or "", args.content_limit)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CLI hybrid search for CK3 RAG (instruction A/B friendly).",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run twice: instruct off, then instruct on (separate processes).",
    )
    p.add_argument(
        "--instruct",
        choices=("off", "on"),
        default="off",
        help="Query-side instruction flag (ignored when --compare is used). Default: off",
    )
    p.add_argument(
        "--index",
        choices=("game", "wiki", "both"),
        default="both",
        help="Which collection to search.",
    )
    p.add_argument("--queries", "-q", type=Path, metavar="FILE", help="One query per line; # comments ok")
    p.add_argument("--top-k", type=int, default=3, metavar="N")
    p.add_argument("--alpha", type=float, default=0.5, help="Hybrid vector weight [0..1]")
    p.add_argument(
        "--category",
        type=str,
        default=None,
        help="Game only: metadata file_category filter (e.g. events).",
    )
    p.add_argument(
        "--show-content",
        action="store_true",
        help="Print each hit's `content` (full parent doc — what MCP returns, minus score).",
    )
    p.add_argument(
        "--content-limit",
        type=int,
        default=1200,
        metavar="N",
        help="Max characters of content per hit (0 = no truncation). Default: %(default)s",
    )
    p.add_argument(
        "query_strings",
        nargs="*",
        help="Additional queries (same run as --queries lines).",
    )
    return p


def main() -> None:
    os.chdir(ROOT)
    parser = _build_parser()
    args = parser.parse_args()

    if args.compare:
        script = str(Path(__file__).resolve())

        def _child_cmd(mode: str) -> list[str]:
            cmd: list[str] = [
                sys.executable,
                script,
                f"--instruct={mode}",
                "--index",
                args.index,
                "--top-k",
                str(args.top_k),
                "--alpha",
                str(args.alpha),
            ]
            if args.category:
                cmd.extend(["--category", args.category])
            if args.queries:
                cmd.extend(["--queries", str(args.queries)])
            if args.show_content:
                cmd.append("--show-content")
                cmd.extend(["--content-limit", str(args.content_limit)])
            cmd.extend(args.query_strings)
            return cmd

        for mode in ("off", "on"):
            env = os.environ.copy()
            env["EMBEDDING_QUERY_INSTRUCTION_ENABLED"] = "true" if mode == "on" else "false"
            print(f"\n{'=' * 72}")
            print(f" EMBEDDING_QUERY_INSTRUCTION_ENABLED = {env['EMBEDDING_QUERY_INSTRUCTION_ENABLED']}")
            print(f"{'=' * 72}")
            r = subprocess.run(_child_cmd(mode), env=env, cwd=str(ROOT))
            if r.returncode != 0:
                sys.exit(r.returncode)
        return

    _run_searches(args)


if __name__ == "__main__":
    main()
