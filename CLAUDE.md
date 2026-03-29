# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands



```bat
REM Install dependencies
py -m pip install -r requirements.txt

REM Copy and fill in API key
copy .env.example .env

REM Run the indexing pipeline (one-time, slow — embeds all game files via OpenRouter)
py -m src.ingestion.pipeline

REM Start the MCP server
py -m src.mcp.server
REM or, if the fastmcp CLI is on PATH:
REM fastmcp run src/mcp/server.py

REM Quick parser smoke-test (no API key needed)
py -c "from pathlib import Path; from src.ingestion.clausewitz_parser import parse_game_file; from src.ingestion.wiki_parser import parse_wiki_file; print(parse_game_file(Path('data/game/events/birth_events.txt'))[0]['block_name']); print(parse_wiki_file(Path('data/wiki/Event_modding.md'))[0]['section_title'])"
```

For multi-line `py -c` snippets, use a `.py` file or paste into `py` interactively; the one-liner above is enough for a quick check.

## Architecture

### Data flow

```
data/game/{common,events,gui}/  ──► clausewitz_parser  ─┐
data/wiki/*.md                  ──► wiki_parser         ─┤
                                                          ▼
                                              parent docs (block / section)
                                                          │
                                              SentenceSplitter (256 tok / 50 overlap)
                                                          │
                                    ┌─────────────────────┴──────────────────────┐
                                    ▼                                            ▼
                              ChromaDB (cosine)                         BM25Okapi (pickled)
                              storage/chroma/                           storage/bm25/*.pkl
                                    │                                            │
                                    └──────────────── HybridSearchIndex ─────────┘
                                                              │
                                                  parent_id → storage/parents/*.json
                                                              │
                                                       MCP tools (fastmcp)
```

### Parent-child retrieval pattern

The system uses **small-to-big** retrieval. Child chunks (256 tokens) are what gets embedded and searched. When a child matches, its `parent_id` is used to fetch the full parent document from `storage/parents/{game,wiki}.json`. The MCP tools return parent content, not child chunks.

- **Game files**: one top-level Clausewitz block = one parent (e.g., one event `birth.1001 = { ... }`, one decision `commission_artifact_decision = { ... }`)
- **`.info` files**: entire file = one parent (these are Paradox's own syntax documentation)
- **Wiki pages**: one `##` section = one parent

### Key files

| File | Role |
|---|---|
| `src/config.py` | All paths and constants; edit here to change chunk size, collections, which dirs to index |
| `src/ingestion/clausewitz_parser.py` | Character-level parser for `.txt`/`.gui`/`.info` — extracts top-level `name = { ... }` blocks |
| `src/ingestion/wiki_parser.py` | Splits Markdown by `##` headers |
| `src/ingestion/pipeline.py` | Orchestrates parsing → chunking → embedding → ChromaDB upsert → BM25 build → parent JSON save |
| `src/retrieval/embeddings.py` | `QwenOpenRouterEmbedding` — wraps OpenAI client pointed at OpenRouter |
| `src/retrieval/hybrid_search.py` | `HybridSearchIndex` — loaded at server start; holds BM25 + ChromaDB collection + parent store |
| `src/mcp/server.py` | Two MCP tools: `search_wiki` and `search_game_files` |

### Storage layout (created by pipeline)

```
storage/
  chroma/          ChromaDB persistent store (two collections: ck3_game, ck3_wiki)
  bm25/
    game.pkl       BM25Okapi object + node list with metadata
    wiki.pkl
  parents/
    game.json      {parent_id → parent doc dict} — looked up at retrieval time
    wiki.json
```

### Hybrid scoring

Vector and BM25 scores are each normalised to `[0, 1]` then combined:
`final = alpha * vector_score + (1 - alpha) * bm25_score` (default `alpha=0.5`).

Results are deduped by `parent_id` (best child score wins) before the top-k parents are returned.

### Clausewitz parser notes

The parser handles `#` line comments, `"..."` quoted strings, and arbitrarily nested braces. It extracts only blocks that contain `{ ... }` — simple assignments like `namespace = birth` are skipped. Files are read with `utf-8-sig` to strip the BOM that Paradox tools add.

`file_category` is inferred from the directory name (e.g., `common/decisions/foo.txt` → `"decisions"`). It is exposed as a ChromaDB metadata filter in `search_game_files(category=...)`.
