"""
ExperimentConfig — loads a YAML experiment definition and resolves file paths.

YAML schema
-----------
name: str                     # used as collection suffix and results dir name
description: str              # human-readable

data:
  common_categories:          # subdirectories of data/game/common/ to include
    - decisions
    - on_action
    - scripted_triggers
  include_events: true        # include data/game/events/
  include_gui: false          # include data/game/gui/
  wiki_files:                 # filenames in data/wiki/  (or "all")
    - Event_modding.md
    - Effects.md

chunking:
  strategy: sentence_splitter   # "sentence_splitter" | "ast" (future)
  chunk_size: 256
  chunk_overlap: 50

retrieval:
  alpha: 0.5                  # vector weight; 1-alpha = BM25 weight
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml

from src.config import GAME_DATA_DIR, WIKI_DATA_DIR, MOD_DATA_DIR, GAME_FILE_EXTENSIONS

EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"


def _collect_tree_files(base: Path, roots: list[str]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        rel = root.replace("\\", "/").strip("/")
        p = (base / rel).resolve()
        if not p.exists():
            continue
        if p.is_file():
            out.append(p)
            continue
        for ext in GAME_FILE_EXTENSIONS:
            out.extend(p.rglob(f"*{ext}"))
    return out


def _load_file_list(base_dir: Path, relative_path: str) -> list[str]:
    """
    Load newline-delimited relative file paths from a text file.
    Empty lines and comments ('# ...') are ignored.
    """
    list_path = (base_dir / relative_path).resolve()
    with open(list_path, "r", encoding="utf-8") as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
    return [line for line in lines if line]


@dataclass
class ExperimentConfig:
    name: str
    description: str = ""

    # ── data ─────────────────────────────────────────────────────────────────
    common_categories: list[str] = field(default_factory=lambda: [
        "decisions", "on_action", "scripted_triggers", "scripted_effects", "traits",
    ])
    include_events: bool = True
    include_gui: bool = False
    wiki_files: Union[list[str], str] = "all"   # list of filenames or "all"
    # If set, overrides all category/event/gui discovery with an explicit file list.
    # Paths are relative to GAME_DATA_DIR (e.g. "common/decisions/00_artifact_decisions.txt").
    game_files: Optional[list[str]] = None
    # Optional path to a newline-delimited file list. Relative to config YAML file.
    game_files_txt: Optional[str] = None
    # Index only these subtrees (relative paths). Vanilla: under data/game/. Mod: under data/mod/.
    vanilla_roots: list[str] = field(default_factory=list)
    mod_roots: list[str] = field(default_factory=list)

    # ── chunking ──────────────────────────────────────────────────────────────
    chunking_strategy: str = "sentence_splitter"  # "sentence_splitter" | "ast"
    chunk_size: int = 256
    chunk_overlap: int = 50

    # ── retrieval ────────────────────────────────────────────────────────────
    alpha: float = 0.5

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def game_collection(self) -> str:
        return f"exp_{self.name}_game"

    @property
    def wiki_collection(self) -> str:
        return f"exp_{self.name}_wiki"

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.name

    @property
    def bm25_game_path(self) -> Path:
        return self.results_dir / "bm25_game.pkl"

    @property
    def bm25_wiki_path(self) -> Path:
        return self.results_dir / "bm25_wiki.pkl"

    @property
    def parents_game_path(self) -> Path:
        return self.results_dir / "parents_game.json"

    @property
    def parents_wiki_path(self) -> Path:
        return self.results_dir / "parents_wiki.json"

    # ── file collection ───────────────────────────────────────────────────────

    def game_file_paths(self) -> list[Path]:
        if self.game_files is not None:
            return [p for f in self.game_files if (p := GAME_DATA_DIR / f).exists()]
        if self.vanilla_roots or self.mod_roots:
            files = _collect_tree_files(GAME_DATA_DIR, self.vanilla_roots)
            files.extend(_collect_tree_files(MOD_DATA_DIR, self.mod_roots))
            return sorted(set(files), key=lambda x: str(x))
        files: list[Path] = []
        for cat in self.common_categories:
            base = GAME_DATA_DIR / "common" / cat
            if not base.exists():
                continue
            for ext in GAME_FILE_EXTENSIONS:
                files.extend(base.rglob(f"*{ext}"))
        if self.include_events:
            for ext in GAME_FILE_EXTENSIONS:
                files.extend((GAME_DATA_DIR / "events").rglob(f"*{ext}"))
        if self.include_gui:
            for ext in GAME_FILE_EXTENSIONS:
                files.extend((GAME_DATA_DIR / "gui").rglob(f"*{ext}"))
        return files

    def wiki_file_paths(self) -> list[Path]:
        if self.wiki_files == "all":
            return list(WIKI_DATA_DIR.glob("*.md"))
        return [WIKI_DATA_DIR / f for f in self.wiki_files
                if (WIKI_DATA_DIR / f).exists()]

    # ── loader ────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        data = raw.get("data", {})
        chunking = raw.get("chunking", {})
        retrieval = raw.get("retrieval", {})

        game_files = data.get("game_files", None)
        game_files_txt = data.get("game_files_txt", None)
        if game_files_txt:
            game_files = _load_file_list(path.parent, game_files_txt)

        roots = data.get("roots", {})
        vanilla_roots = roots.get("vanilla", data.get("vanilla_roots", []))
        mod_roots = roots.get("mod", data.get("mod_roots", []))

        return cls(
            name=raw["name"],
            description=raw.get("description", ""),
            common_categories=data.get("common_categories", cls.__dataclass_fields__["common_categories"].default_factory()),
            include_events=data.get("include_events", True),
            include_gui=data.get("include_gui", False),
            wiki_files=data.get("wiki_files", "all"),
            chunking_strategy=chunking.get("strategy", "sentence_splitter"),
            chunk_size=chunking.get("chunk_size", 256),
            chunk_overlap=chunking.get("chunk_overlap", 50),
            alpha=retrieval.get("alpha", 0.5),
            game_files=game_files,
            game_files_txt=game_files_txt,
            vanilla_roots=vanilla_roots,
            mod_roots=mod_roots,
        )
