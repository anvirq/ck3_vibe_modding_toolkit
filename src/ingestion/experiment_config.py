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
from typing import Union

import yaml

from src.config import GAME_DATA_DIR, WIKI_DATA_DIR, GAME_FILE_EXTENSIONS

EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"


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
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        data = raw.get("data", {})
        chunking = raw.get("chunking", {})
        retrieval = raw.get("retrieval", {})

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
        )
