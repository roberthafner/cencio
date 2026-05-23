import hashlib
import shutil
from pathlib import Path

import pytest

from src.ingestion.indexer import Indexer, is_low_quality_chunk, LOW_QUALITY_NAMES
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


# ------------------------------------------------------------------
# is_low_quality_chunk tests
# ------------------------------------------------------------------

class TestIsLowQualityChunk:
    """Tests for the is_low_quality_chunk function."""

    def test_names_in_low_quality_set_are_low_quality(self):
        """Names in LOW_QUALITY_NAMES should be considered low quality."""
        for name in LOW_QUALITY_NAMES:
            assert is_low_quality_chunk(name, "function") is True, f"{name!r} should be low quality"

    def test_empty_name_is_low_quality(self):
        """Empty name should be low quality."""
        assert is_low_quality_chunk("", "function") is True

    def test_blank_identifier_is_low_quality(self):
        """Blank identifier '_' should be low quality."""
        assert is_low_quality_chunk("_", "var") is True

    def test_common_identifiers_are_low_quality(self):
        """Common identifiers like err, ctx, ok should be low quality."""
        assert is_low_quality_chunk("err", "var") is True
        assert is_low_quality_chunk("ctx", "var") is True
        assert is_low_quality_chunk("ok", "var") is True

    def test_loop_counters_are_low_quality(self):
        """Loop counters i, j, k, n, m should be low quality."""
        assert is_low_quality_chunk("i", "var") is True
        assert is_low_quality_chunk("j", "var") is True
        assert is_low_quality_chunk("k", "var") is True
        assert is_low_quality_chunk("n", "var") is True
        assert is_low_quality_chunk("m", "var") is True

    def test_single_letter_names_are_low_quality(self):
        """Any single-letter name should be low quality."""
        assert is_low_quality_chunk("x", "function") is True
        assert is_low_quality_chunk("y", "var") is True
        assert is_low_quality_chunk("z", "const") is True
        assert is_low_quality_chunk("a", "method") is True

    def test_short_var_const_names_are_low_quality(self):
        """Very short names (<=2 chars) for var/const are low quality."""
        assert is_low_quality_chunk("id", "var") is True
        assert is_low_quality_chunk("db", "const") is True
        assert is_low_quality_chunk("tx", "var") is True

    def test_short_names_for_other_types_not_low_quality(self):
        """Short names for non-var/const types are not automatically low quality."""
        assert is_low_quality_chunk("ID", "function") is False
        assert is_low_quality_chunk("Do", "method") is False
        assert is_low_quality_chunk("TX", "struct") is False

    def test_normal_names_not_low_quality(self):
        """Normal descriptive names should not be low quality."""
        assert is_low_quality_chunk("FindUser", "function") is False
        assert is_low_quality_chunk("UserService", "struct") is False
        assert is_low_quality_chunk("HandleRequest", "method") is False
        assert is_low_quality_chunk("MaxRetries", "const") is False
        assert is_low_quality_chunk("defaultTimeout", "var") is False

    def test_unnamed_block_with_good_doc_not_low_quality(self):
        """Unnamed blocks with sufficient documentation (>=50 chars) are not low quality."""
        good_doc = "This is a detailed documentation comment that explains what this block does in the codebase."
        assert len(good_doc.strip()) >= 50
        assert is_low_quality_chunk("", "block", doc=good_doc) is False

    def test_unnamed_block_with_short_doc_is_low_quality(self):
        """Unnamed blocks with insufficient documentation are low quality."""
        short_doc = "Short doc"
        assert len(short_doc.strip()) < 50
        assert is_low_quality_chunk("", "block", doc=short_doc) is True

    def test_unnamed_block_with_empty_doc_is_low_quality(self):
        """Unnamed blocks with no documentation are low quality."""
        assert is_low_quality_chunk("", "block", doc="") is True

    def test_unnamed_block_with_whitespace_doc_is_low_quality(self):
        """Unnamed blocks with only whitespace doc are low quality."""
        whitespace_doc = "   \n\t   "
        assert is_low_quality_chunk("", "block", doc=whitespace_doc) is True
