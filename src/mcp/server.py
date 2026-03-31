"""
CK3 Modding Toolkit — MCP Server

Tools exposed:
    search_wiki(query, top_k?)                    → modding documentation from the wiki
    search_game_files(query, top_k?, category?)   → examples from game source files
    validate_mod(mod_path, game_path?)            → filtered ck3-tiger validation output

Run (HTTP / Streamable HTTP on http://127.0.0.1:8000/mcp by default):
    py -m src.mcp.server
Override bind/path via FASTMCP_HOST, FASTMCP_PORT, FASTMCP_STREAMABLE_HTTP_PATH (see fastmcp settings).

Stdio: call mcp.run(transport="stdio") from a small script or REPL.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional
import fastmcp

from src.config import default_ck3_game_path, resolve_ck3_tiger_executable
from src.retrieval.hybrid_search import get_game_index, get_wiki_index

mcp = fastmcp.FastMCP(
    name="ck3-modding-toolkit",
    instructions=(
        "Use search_wiki to look up CK3 modding concepts, syntax rules, and documentation. "
        "Use search_game_files to find concrete examples from Paradox's own game scripts. "
        "Both search tools return focused 256-token excerpts with a relevance score and parent_id. "
        "Call get_section(parent_id) only when you need the full block or section for additional context. "
        "For exact id/symbol lookup (e.g. 'travel_start_event.1000'), use search_game_files with exact=True. "
        "Use path_prefix to restrict search to a specific directory (e.g. 'travel_events' or 'common/on_action'). "
        "Use validate_mod to run ck3-tiger on the mod and get a filtered, actionable list of errors and warnings. "
        "Always validate after writing or modifying mod files."
    ),
)

# Severity levels ordered from least to most severe
_SEVERITY_ORDER = ["tips", "untidy", "warning", "error", "fatal"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _format_results(results: list[dict]) -> str:
    """Format child chunk results. Each result includes a parent_id for get_section()."""
    if not results:
        return "No results found."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        source = r.get("source_type", "")
        if source == "wiki":
            header = (f"[{i}] score={score:.2f}  Wiki · {r.get('file_path', '')}"
                      f" · {r.get('section_title', '')}")
        else:
            header = (f"[{i}] score={score:.2f}  Game · {r.get('file_category', '')}"
                      f" · {r.get('block_name', '')}  ({r.get('file_path', '')})")

        parts.append(header)
        parts.append(f"parent_id: {r.get('parent_id', '')}")
        parts.append(r.get("text", ""))
        parts.append("")

    return "\n".join(parts).strip()


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_wiki(
    query: str,
    top_k: int = 5,
) -> str:
    """
    Search the CK3 modding wiki for documentation, syntax explanations, and guides.

    Returns focused 256-token excerpts. Each result includes a parent_id —
    call get_section(parent_id) to read the full wiki section if needed.

    Parameters
    ----------
    query   : what to search for (natural language)
    top_k   : number of excerpts to return (default 5)
    """
    index = get_wiki_index()
    results = index.query(query_text=query, top_k=top_k)
    return _format_results(results)


@mcp.tool()
def search_game_files(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
    path_prefix: Optional[str] = None,
    exact: bool = False,
) -> str:
    """
    Search CK3 game source files for concrete script examples.

    Returns focused 256-token excerpts. Each result includes a parent_id —
    call get_section(parent_id) to read the full block (event, decision, etc.) if needed.

    Parameters
    ----------
    query       : what to search for (natural language or script keywords)
    top_k       : number of excerpts to return (default 5)
    category    : restrict to a file category — events, decisions, traits, modifiers,
                  scripted_effects, scripted_triggers, on_action, buildings, gui, …
    path_prefix : restrict to files whose path contains this substring,
                  e.g. "travel_events" or "common/on_action"
    exact       : if True, use BM25-dominant scoring (alpha=0.1) — best for looking up
                  exact ids/symbols like "travel_start_event.1000" or "on_travel_plan_complete"
    """
    index = get_game_index()
    results = index.query(
        query_text=query,
        top_k=top_k,
        alpha=0.1 if exact else 0.5,
        filter_category=category or None,
        path_prefix=path_prefix or None,
    )
    return _format_results(results)


@mcp.tool()
def get_section(parent_id: str, collection: str = "auto") -> str:
    """
    Retrieve the full source text for a parent_id returned by search_wiki or search_game_files.

    Use this when a search excerpt is insufficient and you need the complete
    wiki section or game script block for context.

    Parameters
    ----------
    parent_id  : the parent_id field from a search result
    collection : "wiki", "game", or "auto" (default — inferred from parent_id)
    """
    if collection == "auto":
        collection = "wiki" if "data/wiki" in parent_id or "wiki" in parent_id.split("::")[0] else "game"

    index = get_wiki_index() if collection == "wiki" else get_game_index()
    parent = index.get_parent(parent_id)

    if not parent:
        return f"Not found: {parent_id!r}"

    source = parent.get("source_type", "")
    if source == "wiki":
        header = f"Wiki · {parent.get('file_path', '')} · {parent.get('section_title', '')}"
    else:
        header = (f"Game · {parent.get('file_category', '')} · {parent.get('block_name', '')}"
                  f"  ({parent.get('file_path', '')})")

    return f"{header}\n\n{parent.get('content', '')}"


@mcp.tool()
def validate_mod(
    mod_path: str,
    game_path: Optional[str] = None,
    min_severity: str = "warning",
) -> str:
    """
    Run ck3-tiger on the mod and return a compact, actionable validation report.

    Uses ck3-tiger's JSON output so no progress noise reaches the agent.
    Results are grouped by severity and sorted by file path.

    Parameters
    ----------
    mod_path     : path to the mod's .mod file
    game_path    : path to the CK3 ``game`` directory (default: ``CK3_GAME_PATH`` in ``.env``)
    min_severity : minimum severity to include — tips | untidy | warning | error (default: warning)
    """
    mod_file = Path(mod_path)
    if not mod_file.exists():
        return f"Error: .mod file not found: {mod_path}"

    tiger_exe = resolve_ck3_tiger_executable()
    if not tiger_exe:
        return (
            "Error: ck3-tiger not found. Set CK3_TIGER_EXE to the binary path "
            "or install ck3-tiger on PATH."
        )

    resolved_game = (game_path or "").strip() or default_ck3_game_path()
    if not resolved_game:
        return (
            "Error: CK3 game directory not set. Pass game_path or set CK3_GAME_PATH "
            "in .env (path to the vanilla .../Crusader Kings III/game folder)."
        )

    try:
        result = subprocess.run(
            [tiger_exe, "--json", "--game", resolved_game, str(mod_file)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        return f"Error: ck3-tiger not found at {tiger_exe!r}"
    except subprocess.TimeoutExpired:
        return "Error: ck3-tiger timed out after 120s."

    # ck3-tiger writes JSON to stdout; stderr has human-readable progress
    try:
        reports: list[dict] = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Fallback: return raw stderr trimmed to 3000 chars
        return (result.stderr or result.stdout or "No output from ck3-tiger.")[:3000]

    # Filter by minimum severity
    min_idx = _SEVERITY_ORDER.index(min_severity.lower()) if min_severity.lower() in _SEVERITY_ORDER else 2
    reports = [r for r in reports if _SEVERITY_ORDER.index(r.get("severity", "tips").lower()) >= min_idx]

    if not reports:
        return f"No issues at severity >= {min_severity}."

    # Group by severity (highest first), then by file
    by_severity: dict[str, list[dict]] = {}
    for r in reports:
        sev = r.get("severity", "unknown").lower()
        by_severity.setdefault(sev, []).append(r)

    parts = []
    for sev in reversed(_SEVERITY_ORDER):
        group = by_severity.get(sev)
        if not group:
            continue
        parts.append(f"{sev.upper()} ({len(group)}):")
        for r in sorted(group, key=lambda x: x.get("file", "")):
            file_loc = r.get("file", "")
            if r.get("line"):
                file_loc += f":{r['line']}"
            msg = r.get("msg") or r.get("message") or r.get("text") or str(r)
            parts.append(f"  [{r.get('key', '')}] {file_loc}: {msg}")
        parts.append("")

    return "\n".join(parts).strip()


if __name__ == "__main__":
    mcp.run(transport="http")
