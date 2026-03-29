"""
Parser for CK3 modding wiki Markdown files.

Strategy: split by `##` headers → each section is one parent document.
The content before the first `##` (page intro) is kept as its own section.
"""

import logging
import re
from pathlib import Path

from src.config import BASE_DIR

log = logging.getLogger(__name__)


def parse_wiki_file(path: Path) -> list[dict]:
    """
    Parse a single wiki Markdown file into section-level parent documents.

        {
            "parent_id": str,       # relative path + section title
            "section_title": str,
            "content": str,         # full section text including its header
            "file_path": str,
            "source_type": "wiki",
        }
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return []

    rel_path = str(path.relative_to(BASE_DIR))
    page_title = path.stem.replace("_", " ")

    # Split on `##` or deeper headers (but not `#` which is the page title)
    # We keep the delimiter at the start of each section via a capturing group
    pattern = re.compile(r"(?=^#{2,}\s)", re.MULTILINE)
    raw_sections = pattern.split(text)

    results = []
    seen_titles: dict[str, int] = {}
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        # Derive section title
        first_line = section.splitlines()[0].strip()
        title_match = re.match(r"^(#{2,})\s+(.*)", first_line)
        if title_match:
            section_title = title_match.group(2).strip()
        else:
            # Intro text before any ## header
            section_title = f"{page_title} (intro)"

        count = seen_titles.get(section_title, 0)
        seen_titles[section_title] = count + 1
        suffix = f"_{count + 1}" if count > 0 else ""
        parent_id = f"{rel_path}::{section_title}{suffix}"

        results.append(
            {
                "parent_id": parent_id,
                "section_title": section_title,
                "content": section,
                "file_path": rel_path,
                "source_type": "wiki",
            }
        )

    return results
