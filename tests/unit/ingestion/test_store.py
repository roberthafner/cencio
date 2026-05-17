import hashlib
from pathlib import Path

import pytest

from src.ingestion.store import ChunkStore
from src.models.chunk import Chunk, ChunkType


class StubEmbeddingFunction:
    """Deterministic stub — different texts get different embeddings."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        result = []
        for text in input:
            h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 1e8
            result.append([h, h * 0.5, h * 0.25, h * 0.125])
        return result


def _make_chunk(name: str, content: str, file_path: str = "main.go") -> Chunk:
    from src.models.chunk import generate_id
    return Chunk(
        id=generate_id(content),
        type=ChunkType.FUNCTION,
        content=content,
        name=name,
        package_name="main",
        file_path=file_path,
        start_line=1,
        end_line=5,
        doc=f"// {name} does something",
        signature=f"func {name}()",
    )


@pytest.fixture()
def store(tmp_path):
    s = ChunkStore(
        chroma_path=tmp_path / "chroma",
        sqlite_path=tmp_path / "index.db",
        embedding_fn=StubEmbeddingFunction(),
    )
    yield s
    s.close()


# ------------------------------------------------------------------
# upsert_chunks
# ------------------------------------------------------------------

def test_upsert_stores_chunks(store):
    chunks = [_make_chunk("Foo", "func Foo() { return }")]
    store.upsert_chunks(chunks, "myrepo", "main.go")

    assert store.chunk_count_for_file("myrepo", "main.go") == 1


def test_upsert_multiple_chunks(store):
    chunks = [
        _make_chunk("Foo", "func Foo() {}"),
        _make_chunk("Bar", "func Bar() {}"),
    ]
    store.upsert_chunks(chunks, "myrepo", "main.go")

    assert store.chunk_count_for_file("myrepo", "main.go") == 2


def test_upsert_empty_list_is_noop(store):
    store.upsert_chunks([], "myrepo", "main.go")
    assert store.chunk_count_for_file("myrepo", "main.go") == 0


def test_upsert_is_idempotent(store):
    chunks = [_make_chunk("Foo", "func Foo() {}")]
    store.upsert_chunks(chunks, "myrepo", "main.go")
    store.upsert_chunks(chunks, "myrepo", "main.go")

    assert store.chunk_count_for_file("myrepo", "main.go") == 1


# ------------------------------------------------------------------
# delete_chunks_for_file
# ------------------------------------------------------------------

def test_delete_removes_all_chunks_for_file(store):
    chunks = [
        _make_chunk("Foo", "func Foo() {}"),
        _make_chunk("Bar", "func Bar() {}"),
    ]
    store.upsert_chunks(chunks, "myrepo", "main.go")
    store.delete_chunks_for_file("myrepo", "main.go")

    assert store.chunk_count_for_file("myrepo", "main.go") == 0


def test_delete_does_not_affect_other_files(store):
    chunks_a = [_make_chunk("Foo", "func Foo() {}", "a.go")]
    chunks_b = [_make_chunk("Bar", "func Bar() {}", "b.go")]
    store.upsert_chunks(chunks_a, "myrepo", "a.go")
    store.upsert_chunks(chunks_b, "myrepo", "b.go")

    store.delete_chunks_for_file("myrepo", "a.go")

    assert store.chunk_count_for_file("myrepo", "a.go") == 0
    assert store.chunk_count_for_file("myrepo", "b.go") == 1


def test_delete_nonexistent_file_is_noop(store):
    store.delete_chunks_for_file("myrepo", "ghost.go")  # no error


# ------------------------------------------------------------------
# file hash tracking
# ------------------------------------------------------------------

def test_set_and_get_file_hash(store):
    store.set_file_hash("myrepo", "main.go", "abc123")
    hashes = store.get_all_file_hashes("myrepo")

    assert hashes == {"main.go": "abc123"}


def test_get_all_file_hashes_empty(store):
    assert store.get_all_file_hashes("myrepo") == {}


def test_set_file_hash_overwrite(store):
    store.set_file_hash("myrepo", "main.go", "old")
    store.set_file_hash("myrepo", "main.go", "new")
    hashes = store.get_all_file_hashes("myrepo")

    assert hashes["main.go"] == "new"


def test_delete_file_tracking(store):
    store.set_file_hash("myrepo", "main.go", "abc123")
    store.delete_file_tracking("myrepo", "main.go")
    hashes = store.get_all_file_hashes("myrepo")

    assert "main.go" not in hashes


def test_hashes_are_scoped_to_repo(store):
    store.set_file_hash("repo_a", "main.go", "hash_a")
    store.set_file_hash("repo_b", "main.go", "hash_b")

    assert store.get_all_file_hashes("repo_a") == {"main.go": "hash_a"}
    assert store.get_all_file_hashes("repo_b") == {"main.go": "hash_b"}


# ------------------------------------------------------------------
# keyword_search
# ------------------------------------------------------------------

def test_keyword_search_finds_matching_chunk(store):
    chunks = [_make_chunk("FindUser", "func FindUser(id string) *User { return nil }")]
    store.upsert_chunks(chunks, "myrepo", "main.go")

    results = store.keyword_search("FindUser")
    assert len(results) == 1
    assert results[0]["id"] == chunks[0].id


def test_keyword_search_no_match_returns_empty(store):
    chunks = [_make_chunk("Foo", "func Foo() {}")]
    store.upsert_chunks(chunks, "myrepo", "main.go")

    results = store.keyword_search("zzznomatch")
    assert results == []


def test_keyword_search_respects_top_k(store):
    chunks = [
        _make_chunk(f"Func{i}", f"func Func{i}() {{ return }}")
        for i in range(5)
    ]
    store.upsert_chunks(chunks, "myrepo", "main.go")

    results = store.keyword_search("func", top_k=2)
    assert len(results) <= 2


# ------------------------------------------------------------------
# semantic_search
# ------------------------------------------------------------------

def test_semantic_search_returns_results(store):
    chunks = [
        _make_chunk("Foo", "func Foo() {}"),
        _make_chunk("Bar", "func Bar() {}"),
    ]
    store.upsert_chunks(chunks, "myrepo", "main.go")

    results = store.semantic_search("function", top_k=2)
    assert len(results) == 2


def test_semantic_search_empty_store_returns_empty(store):
    results = store.semantic_search("anything")
    assert results == []


def test_semantic_search_respects_top_k(store):
    chunks = [_make_chunk(f"F{i}", f"func F{i}() {{}}") for i in range(5)]
    store.upsert_chunks(chunks, "myrepo", "main.go")

    results = store.semantic_search("function", top_k=3)
    assert len(results) <= 3


# ------------------------------------------------------------------
# hybrid_search
# ------------------------------------------------------------------

def test_hybrid_search_returns_results(store):
    chunks = [_make_chunk("FindUser", "func FindUser(id string) *User { return nil }")]
    store.upsert_chunks(chunks, "myrepo", "main.go")

    results = store.hybrid_search("FindUser")
    assert len(results) >= 1


def test_hybrid_search_empty_store_returns_empty(store):
    results = store.hybrid_search("anything")
    assert results == []


def test_hybrid_search_results_have_metadata(store):
    # Upsert a chunk that matches keyword search but not semantic search
    # by ensuring semantic returns nothing (empty store path won't work here,
    # so we verify all returned results carry metadata and content).
    chunks = [_make_chunk("FindUser", "func FindUser(id string) *User { return nil }")]
    store.upsert_chunks(chunks, "myrepo", "main.go")

    results = store.hybrid_search("FindUser")
    assert len(results) >= 1
    for r in results:
        assert "metadata" in r
        assert "content" in r
        assert r["metadata"]["name"] == "FindUser"


# ------------------------------------------------------------------
# repo_name filter
# ------------------------------------------------------------------

def test_semantic_search_filters_by_repo(store):
    store.upsert_chunks([_make_chunk("Foo", "func Foo() {}", "a.go")], "repo_a", "a.go")
    store.upsert_chunks([_make_chunk("Bar", "func Bar() {}", "b.go")], "repo_b", "b.go")

    results = store.semantic_search("function", top_k=10, repo_name="repo_a")
    assert all(r["metadata"]["repo_name"] == "repo_a" for r in results)


def test_keyword_search_filters_by_repo(store):
    store.upsert_chunks([_make_chunk("FooA", "func FooA() {}", "a.go")], "repo_a", "a.go")
    store.upsert_chunks([_make_chunk("FooB", "func FooB() {}", "b.go")], "repo_b", "b.go")

    results = store.keyword_search("FooA", repo_name="repo_a")
    assert len(results) == 1

    results = store.keyword_search("FooB", repo_name="repo_a")
    assert len(results) == 0


def test_hybrid_search_filters_by_repo(store):
    store.upsert_chunks([_make_chunk("Foo", "func Foo() {}", "a.go")], "repo_a", "a.go")
    store.upsert_chunks([_make_chunk("Bar", "func Bar() {}", "b.go")], "repo_b", "b.go")

    results = store.hybrid_search("function", top_k=10, repo_name="repo_a")
    assert all(r["metadata"]["repo_name"] == "repo_a" for r in results)
