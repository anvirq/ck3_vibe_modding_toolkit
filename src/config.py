import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
GAME_DATA_DIR = DATA_DIR / "game"
WIKI_DATA_DIR = DATA_DIR / "wiki"
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
BM25_DIR = STORAGE_DIR / "bm25"
PARENTS_DIR = STORAGE_DIR / "parents"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
EMBEDDING_DIMENSIONS = 4096

# Qwen3-Embedding: optional asymmetric "Instruct: … / Query: …" on search queries only (no reindex).
# Set EMBEDDING_QUERY_INSTRUCTION_ENABLED=true in .env to enable after baseline experiments.
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


EMBEDDING_QUERY_INSTRUCTION_ENABLED = _env_bool(
    "EMBEDDING_QUERY_INSTRUCTION_ENABLED", default=False
)
EMBEDDING_QUERY_INSTRUCTION_GAME = os.getenv(
    "EMBEDDING_QUERY_INSTRUCTION_GAME",
    "Given a user question about Crusader Kings III modding, retrieve relevant passages "
    "from Paradox game script files (events, decisions, traits, modifiers, GUI, Clausewitz/Jomini syntax, .info docs).",
).strip()
EMBEDDING_QUERY_INSTRUCTION_WIKI = os.getenv(
    "EMBEDDING_QUERY_INSTRUCTION_WIKI",
    "Given a user question about Crusader Kings III modding, retrieve relevant sections "
    "from modding wiki documentation (guides, concepts, file formats, effects, triggers, scripting reference).",
).strip()

CHUNK_SIZE = 256   # tokens
CHUNK_OVERLAP = 50  # tokens

GAME_COLLECTION = "ck3_game"
WIKI_COLLECTION = "ck3_wiki"

# Subdirs of data/game/ to index (skip localization, music, sound, etc.)
GAME_DIRS_TO_INDEX = ["common", "events", "gui"]
GAME_FILE_EXTENSIONS = {".txt", ".info", ".gui"}
