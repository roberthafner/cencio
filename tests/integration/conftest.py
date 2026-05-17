import shutil
import urllib.request
from pathlib import Path

import pytest

from src.embedding.ollama import OllamaEmbeddingFunction
from src.ingestion.indexer import Indexer
from src.ingestion.repository import GitRepository
from src.ingestion.store import ChunkStore

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_BUILD_DIR = _PROJECT_ROOT / "build"
_REPOS_DIR = _BUILD_DIR / "data" / "repos"
_DATABASE_DIR = _BUILD_DIR / "database"

_MUX_CLONE_PATH = _REPOS_DIR / "gorilla-mux"
_MUX_CHROMA_PATH = _DATABASE_DIR / "chroma"
_MUX_SQLITE_PATH = _DATABASE_DIR / "index.db"

_OLLAMA_URL = "http://localhost:11434"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--clean",
        action="store_true",
        default=False,
        help="Delete the build directory and force a full re-index.",
    )


@pytest.fixture(scope="session", autouse=True)
def require_ollama() -> None:
    try:
        urllib.request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=3)
    except Exception:
        pytest.skip(
            "Ollama not reachable at localhost:11434 — "
            "start Ollama and run: ollama pull nomic-embed-text:v1.5"
        )


@pytest.fixture(scope="session")
def handle_clean(request: pytest.FixtureRequest) -> None:
    if request.config.getoption("--clean") and _BUILD_DIR.exists():
        shutil.rmtree(_BUILD_DIR)


@pytest.fixture(scope="session")
def mux_repo(handle_clean) -> GitRepository:
    return GitRepository(
        name="gorilla-mux",
        url="git@github.com:gorilla/mux.git",
        branch="main",
        clone_path=_MUX_CLONE_PATH,
    )


@pytest.fixture(scope="session")
def mux_store(mux_repo: GitRepository):
    _MUX_CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    _MUX_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

    embedding_fn = OllamaEmbeddingFunction(base_url=_OLLAMA_URL)
    store = ChunkStore(
        chroma_path=_MUX_CHROMA_PATH,
        sqlite_path=_MUX_SQLITE_PATH,
        embedding_fn=embedding_fn,
    )
    indexer = Indexer(store)
    indexer.index_repository(mux_repo)
    yield store
    store.close()
