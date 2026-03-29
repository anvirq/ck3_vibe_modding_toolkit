# CK3 modding toolkit

Hybrid search (ChromaDB + BM25) over **Crusader Kings III** script files and optional **modding wiki** Markdown. MCP HTTP server with RAG search + ck3-tiger validation for use in editors and agents.

## Tools

| Tool | Purpose |
|------|---------|
| `search_wiki` | Look up CK3 modding concepts, effects, triggers, file formats |
| `search_game_files` | Find vanilla examples of events, decisions, traits, GUI, etc. |
| `validate_mod` | Run [ck3-tiger](https://github.com/amtep/tiger) validation with filtered output |

## Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key (embeddings: `qwen/qwen3-embedding-8b`)
- [ck3-tiger](https://github.com/amtep/tiger) (for `validate_mod` tool)

## Setup

From a terminal in the repository root (Command Prompt or PowerShell):

```bat
py -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `OPENROUTER_API_KEY`.

## Data layout

The indexer reads from `data/` (paths are defined in `src/config.py`):

| Path | Contents |
|------|----------|
| `data/game/` | Copy any folders from your CK3 install's `game` directory (see below) |
| `data/wiki/` | Optional: Markdown documentation files |

### Game files

Copy folders from your CK3 installation, e.g. Steam:

`…/Steam/steamapps/common/Crusader Kings III/game/`

You can index **any script folders** Common choices:

- `common/` — traits, decisions, modifiers, culture, religion, buildings, etc.
- `events/` — event scripts
- `gui/` — GUI definitions

But you can also add `scripted_effects/`, `scripted_triggers/`, `on_action/`, or any other subdirectory. The pipeline indexes all supported file types (`.txt`, `.info`, `.gui`) it finds under `data/game/`.

Do not commit Paradox's files to Git; only `.gitkeep` placeholders are tracked.

### Wiki documentation

A community-maintained CK3 modding wiki is available at [jesec/ck3-modding-wiki](https://github.com/jesec/ck3-modding-wiki). You can export it as Markdown and place under `data/wiki/`, or use your own notes and documentation.

## Indexing

From the repository root:

```bat
py -m src.ingestion.pipeline
```

This is slow on first run (many API calls). It writes vectors and indexes under `storage/`.

## MCP server

Run the HTTP server (default: `http://127.0.0.1:8000/mcp`):

```bat
py -m src.mcp.server
```

Override bind address via environment variables: `FASTMCP_HOST`, `FASTMCP_PORT`, `FASTMCP_STREAMABLE_HTTP_PATH`.

Connect your MCP client (Claude Desktop, editor plugin, etc.) to the HTTP endpoint.

### Tools reference

**`search_wiki(query, top_k?)`** — Search modding documentation. Use for concepts, syntax rules, available effects/triggers.

**`search_game_files(query, top_k?, category?)`** — Search vanilla game scripts. Use for concrete examples. Optional `category` filter matches subdirectory names (e.g., `events`, `decisions`, `traits`).

**`validate_mod(mod_path, game_path?, min_severity?)`** — Run ck3-tiger validation. Returns grouped, filtered output (errors and warnings by default). Use after every code change.

## CLI search / instruction A/B

From the repo root (requires `storage/` built by the pipeline and a filled `.env`):

```bat
py scripts\rag_ab_test.py --instruct off "your question here"
py scripts\rag_ab_test.py --compare --queries scripts\sample_queries.txt
```

`--compare` runs the same queries twice (instruction off, then on) in separate processes so embedding flags apply cleanly. Use `--show-content` to print the same `content` field the MCP tools return for each hit (`--content-limit 0` for the full parent). See `scripts/rag_ab_test.py --help`.

## License

This repository contains **code only**. Game assets belong to Paradox Interactive; wiki text may be subject to separate licenses—obtain and use those materials according to their terms.
