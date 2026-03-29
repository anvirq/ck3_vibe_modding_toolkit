"""
Parser for Paradox Clausewitz/Jomini script files (.txt, .gui, .info).

Strategy:
- .info files  → entire file is one parent block (documentation)
- .txt / .gui  → each top-level named block ( identifier = { ... } ) is one parent
"""

import logging
from pathlib import Path
from typing import Generator

from src.config import BASE_DIR, GAME_DATA_DIR

log = logging.getLogger(__name__)


def _extract_blocks(text: str) -> Generator[tuple[str, str], None, None]:
    """
    Yield (block_name, block_text) for every top-level `name = { ... }` block.
    Handles nested braces, quoted strings, and `#` comments.
    """
    n = len(text)
    i = 0

    while i < n:
        # ── skip whitespace ──────────────────────────────────────────────────
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break

        # ── skip comment line ────────────────────────────────────────────────
        if text[i] == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue

        # ── record where the header starts ───────────────────────────────────
        header_start = i

        # Scan forward on this "statement" until we either:
        #   (a) find a `{` → it's a block we want
        #   (b) hit a newline at depth 0 → simple assignment, skip
        depth = 0
        block_open = -1

        j = i
        while j < n:
            c = text[j]

            if c == "#":  # comment to EOL
                while j < n and text[j] != "\n":
                    j += 1
                continue

            if c == '"':  # quoted string
                j += 1
                while j < n and text[j] != '"':
                    if text[j] == "\\":
                        j += 1
                    j += 1
                j += 1
                continue

            if c == "{":
                if depth == 0:
                    block_open = j
                depth += 1

            elif c == "}":
                depth -= 1
                if depth == 0 and block_open != -1:
                    block_text = text[header_start : j + 1]
                    # Extract name: text between header_start and block_open, before last '='
                    header = text[header_start:block_open].strip()
                    if "=" in header:
                        name = header[: header.rfind("=")].strip()
                    else:
                        name = header
                    yield name, block_text
                    i = j + 1
                    break  # back to outer while

            elif c == "\n" and depth == 0 and block_open == -1:
                # Simple `key = value` line with no block → skip
                i = j + 1
                break

            j += 1
        else:
            # reached end of text without completing a block
            break


def parse_game_file(path: Path) -> list[dict]:
    """
    Parse a single game file and return a list of parent-document dicts:
        {
            "parent_id": str,          # unique: relative path + block name
            "block_name": str,
            "content": str,
            "file_path": str,          # relative to BASE_DIR (e.g. data/game/events/foo.txt)
            "file_category": str,      # derived from directory structure
            "source_type": str,        # "game_docs" for .info, "game_file" otherwise
        }
    """
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return []

    rel_path = str(path.relative_to(BASE_DIR))
    suffix = path.suffix.lower()
    source_type = "game_docs" if suffix == ".info" else "game_file"
    file_category = _derive_category(path)

    # .info files: treat entire file as one parent
    if suffix == ".info":
        if not text.strip():
            return []
        return [
            {
                "parent_id": f"{rel_path}::__file__",
                "block_name": path.stem,
                "content": text.strip(),
                "file_path": rel_path,
                "file_category": file_category,
                "source_type": source_type,
            }
        ]

    # .txt / .gui files: extract top-level blocks
    results = []
    seen_names: dict[str, int] = {}

    for block_name, block_text in _extract_blocks(text):
        # Disambiguate duplicate names within the same file
        count = seen_names.get(block_name, 0)
        seen_names[block_name] = count + 1
        suffix_str = f"#{count}" if count > 0 else ""
        parent_id = f"{rel_path}::{block_name}{suffix_str}"

        results.append(
            {
                "parent_id": parent_id,
                "block_name": block_name,
                "content": block_text.strip(),
                "file_path": rel_path,
                "file_category": file_category,
                "source_type": source_type,
            }
        )

    return results


def _derive_category(path: Path) -> str:
    """
    Infer a human-readable category from the file's path relative to GAME_DATA_DIR.

    Examples:
      common/decisions/foo.txt  → "decisions"
      common/traits/bar.txt     → "traits"
      common/foo.info           → "common"
      events/birth_events.txt   → "events"
      gui/window/foo.gui        → "gui"   ← whole gui/ subtree maps to "gui"
    """
    try:
        rel = path.relative_to(GAME_DATA_DIR)
    except ValueError:
        return path.parts[-2] if len(path.parts) >= 2 else "unknown"

    top = rel.parts[0]  # "common" | "events" | "gui"
    if top == "common" and len(rel.parts) > 2:
        return rel.parts[1]  # subdirectory of common (decisions, traits, …)
    return top
