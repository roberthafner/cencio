import hashlib
import shutil
from pathlib import Path

import pytest

from src.ingestion.indexer import Indexer
from src.ingestion.repository import GitRepository
from src.ingestion.store import ChunkStore

# Reuse the fixture Go files that already exist in the test data directory
_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "golang"
_FIXTURE_A = _DATA_DIR / "function_example.go"
_FIXTURE_B = _DATA_DIR / "struct_example.go"


class StubEmbeddingFunction:
    def __call__(self, input: list[str]) -> list[list[float]]:
        result = []
        for text in input:
            h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 1e8
            result.append([h, h * 0.5, h * 0.25, h * 0.125])
        return result


@pytest.fixture()
def store(tmp_path):
    s = ChunkStore(
        chroma_path=tmp_path / "chroma",
        sqlite_path=tmp_path / "index.db",
        embedding_fn=StubEmbeddingFunction(),
    )
    yield s
    s.close()


def _make_repo(tmp_path: Path, files: dict[str, Path]) -> GitRepository:
    """Create a local directory with the given Go files (no real git needed)."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(exist_ok=True)
    for dest_name, src_path in files.items():
        shutil.copy(src_path, repo_dir / dest_name)
    repo = GitRepository("testrepo", "file://unused", "main", repo_dir)
    repo.clone_or_pull = lambda: None  # stub git ops
    return repo


# ------------------------------------------------------------------
# first run — all files are new
# ------------------------------------------------------------------

def test_first_run_indexes_all_files(tmp_path, store):
    repo = _make_repo(tmp_path, {
        "function_example.go": _FIXTURE_A,
        "struct_example.go": _FIXTURE_B,
    })
    indexer = Indexer(store)
    indexer.index_repository(repo)

    hashes = store.get_all_file_hashes("testrepo")
    assert "function_example.go" in hashes
    assert "struct_example.go" in hashes


def test_first_run_stores_chunks(tmp_path, store):
    repo = _make_repo(tmp_path, {"function_example.go": _FIXTURE_A})
    indexer = Indexer(store)
    indexer.index_repository(repo)

    assert store.chunk_count_for_file("testrepo", "function_example.go") > 0


# ------------------------------------------------------------------
# new file added
# ------------------------------------------------------------------

def test_new_file_gets_indexed(tmp_path, store):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    shutil.copy(_FIXTURE_A, repo_dir / "function_example.go")

    repo = GitRepository("testrepo", "file://unused", "main", repo_dir)
    repo.clone_or_pull = lambda: None

    indexer = Indexer(store)
    indexer.index_repository(repo)

    # Simulate a new file being added to the repo
    shutil.copy(_FIXTURE_B, repo_dir / "struct_example.go")
    indexer.index_repository(repo)

    hashes = store.get_all_file_hashes("testrepo")
    assert "struct_example.go" in hashes
    assert store.chunk_count_for_file("testrepo", "struct_example.go") > 0


# ------------------------------------------------------------------
# second run with no changes — nothing should be rewritten
# ------------------------------------------------------------------

def test_no_change_second_run_is_noop(tmp_path, store):
    repo = _make_repo(tmp_path, {"function_example.go": _FIXTURE_A})
    indexer = Indexer(store)
    indexer.index_repository(repo)

    count_after_first = store.chunk_count_for_file("testrepo", "function_example.go")
    hash_after_first = store.get_all_file_hashes("testrepo")["function_example.go"]

    indexer.index_repository(repo)

    count_after_second = store.chunk_count_for_file("testrepo", "function_example.go")
    hash_after_second = store.get_all_file_hashes("testrepo")["function_example.go"]

    assert count_after_first == count_after_second
    assert hash_after_first == hash_after_second


# ------------------------------------------------------------------
# modified file — old chunks deleted, new chunks inserted
# ------------------------------------------------------------------

def test_modified_file_replaces_chunks(tmp_path, store):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    target = repo_dir / "example.go"
    shutil.copy(_FIXTURE_A, target)

    repo = GitRepository("testrepo", "file://unused", "main", repo_dir)
    repo.clone_or_pull = lambda: None
    indexer = Indexer(store)
    indexer.index_repository(repo)

    old_hash = store.get_all_file_hashes("testrepo")["example.go"]

    # Replace with a different fixture (different content)
    shutil.copy(_FIXTURE_B, target)
    indexer.index_repository(repo)

    new_hash = store.get_all_file_hashes("testrepo")["example.go"]
    assert old_hash != new_hash
    assert store.chunk_count_for_file("testrepo", "example.go") > 0


# ------------------------------------------------------------------
# deleted file — chunks and tracking removed
# ------------------------------------------------------------------

def test_deleted_file_removes_chunks(tmp_path, store):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    shutil.copy(_FIXTURE_A, repo_dir / "function_example.go")
    shutil.copy(_FIXTURE_B, repo_dir / "struct_example.go")

    repo = GitRepository("testrepo", "file://unused", "main", repo_dir)
    repo.clone_or_pull = lambda: None
    indexer = Indexer(store)
    indexer.index_repository(repo)

    # Delete one file from the repo directory
    (repo_dir / "struct_example.go").unlink()
    indexer.index_repository(repo)

    hashes = store.get_all_file_hashes("testrepo")
    assert "struct_example.go" not in hashes
    assert store.chunk_count_for_file("testrepo", "struct_example.go") == 0


def test_deleted_file_does_not_affect_remaining_files(tmp_path, store):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    shutil.copy(_FIXTURE_A, repo_dir / "function_example.go")
    shutil.copy(_FIXTURE_B, repo_dir / "struct_example.go")

    repo = GitRepository("testrepo", "file://unused", "main", repo_dir)
    repo.clone_or_pull = lambda: None
    indexer = Indexer(store)
    indexer.index_repository(repo)

    (repo_dir / "struct_example.go").unlink()
    indexer.index_repository(repo)

    assert store.chunk_count_for_file("testrepo", "function_example.go") > 0
    assert "function_example.go" in store.get_all_file_hashes("testrepo")
