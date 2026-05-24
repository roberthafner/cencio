import hashlib
from pathlib import Path

import pytest

from src.ingestion.store import ChunkStore, _to_fts_query
from src.models.chunk import Chunk, ChunkType


class StubEmbeddingFunction:
    """Deterministic stub — different texts get different embeddings."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        result = []
        for text in input:
            h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 1e8
            result.append([h, h * 0.5, h * 0.25, h * 0.125])
        return result


def _make_chunk(
    name: str,
    content: str,
    file_path: str = "main.go",
    is_test: bool = False,
    low_quality: bool = False,
) -> Chunk:
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
        is_test=is_test,
        low_quality=low_quality,
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
# _to_fts_query
# ------------------------------------------------------------------

def test_fts_query_strips_stop_words():
    terms = _to_fts_query("how to find a user").split(" OR ")
    assert "how" not in terms
    assert "to" not in terms
    assert "a" not in terms
    assert "find" in terms
    assert "user" in terms


def test_fts_query_joins_with_or():
    result = _to_fts_query("parse query parameters")
    assert result == "parse OR query OR parameters"


def test_fts_query_single_meaningful_term_has_no_or():
    assert _to_fts_query("FindUser") == "FindUser"


def test_fts_query_all_stop_words_falls_back_to_original():
    original = "how to be"
    assert _to_fts_query(original) == original


def test_fts_query_filters_single_char_tokens():
    terms = _to_fts_query("match a b c handler").split(" OR ")
    assert "a" not in terms
    assert "b" not in terms
    assert "c" not in terms
    assert "match" in terms
    assert "handler" in terms


# ------------------------------------------------------------------
# keyword_search
# ------------------------------------------------------------------

def test_keyword_search_natural_language_query(store):
    chunks = [_make_chunk(
        "HandleMiddleware",
        "// HandleMiddleware wraps an http.Handler with middleware logic.\n"
        "func HandleMiddleware(h http.Handler) http.Handler { return h }",
    )]
    store.upsert_chunks(chunks, "myrepo", "main.go")

    results = store.keyword_search("how to wrap an http handler")
    assert len(results) >= 1
    assert results[0]["id"] == chunks[0].id


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


# ------------------------------------------------------------------
# include_tests filter
# ------------------------------------------------------------------

def test_upsert_stores_is_test_metadata(store):
    """Verify is_test is stored in ChromaDB metadata."""
    test_chunk = _make_chunk("TestFoo", "func TestFoo() {}", "foo_test.go", is_test=True)
    store.upsert_chunks([test_chunk], "myrepo", "foo_test.go")

    results = store.semantic_search("TestFoo", top_k=10, include_tests=True)
    assert len(results) == 1
    assert results[0]["metadata"]["is_test"] is True


def test_semantic_search_excludes_test_chunks_by_default(store):
    """Test chunks should be excluded by default."""
    regular_chunk = _make_chunk("RegularFunc", "func RegularFunc() {}", "main.go", is_test=False)
    test_chunk = _make_chunk("TestFunc", "func TestFunc() {}", "main_test.go", is_test=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([test_chunk], "myrepo", "main_test.go")

    results = store.semantic_search("func", top_k=10)
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "RegularFunc"


def test_semantic_search_includes_test_chunks_when_flag_set(store):
    """Test chunks should be included when include_tests=True."""
    regular_chunk = _make_chunk("RegularFunc", "func RegularFunc() {}", "main.go", is_test=False)
    test_chunk = _make_chunk("TestFunc", "func TestFunc() {}", "main_test.go", is_test=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([test_chunk], "myrepo", "main_test.go")

    results = store.semantic_search("func", top_k=10, include_tests=True)
    assert len(results) == 2
    names = {r["metadata"]["name"] for r in results}
    assert names == {"RegularFunc", "TestFunc"}


def test_keyword_search_excludes_test_chunks_by_default(store):
    """Test chunks should be excluded from keyword search by default."""
    regular_chunk = _make_chunk("FindUser", "func FindUser() {}", "main.go", is_test=False)
    test_chunk = _make_chunk("TestFindUser", "func TestFindUser() {}", "main_test.go", is_test=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([test_chunk], "myrepo", "main_test.go")

    results = store.keyword_search("FindUser", top_k=10)
    assert len(results) == 1
    assert results[0]["id"] == regular_chunk.id


def test_keyword_search_includes_test_chunks_when_flag_set(store):
    """Test chunks should be included in keyword search when include_tests=True."""
    regular_chunk = _make_chunk("FindUser", "func FindUser() { /* find user */ }", "main.go", is_test=False)
    test_chunk = _make_chunk("TestFindUser", "func TestFindUser() { /* find user test */ }", "main_test.go", is_test=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([test_chunk], "myrepo", "main_test.go")

    # Both chunks contain "find" and "user" so both should match
    results = store.keyword_search("find user", top_k=10, include_tests=True)
    assert len(results) == 2
    ids = {r["id"] for r in results}
    assert ids == {regular_chunk.id, test_chunk.id}


def test_hybrid_search_excludes_test_chunks_by_default(store):
    """Test chunks should be excluded from hybrid search by default."""
    regular_chunk = _make_chunk("ProcessData", "func ProcessData() {}", "main.go", is_test=False)
    test_chunk = _make_chunk("TestProcessData", "func TestProcessData() {}", "main_test.go", is_test=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([test_chunk], "myrepo", "main_test.go")

    results = store.hybrid_search("ProcessData", top_k=10)
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "ProcessData"


def test_hybrid_search_includes_test_chunks_when_flag_set(store):
    """Test chunks should be included in hybrid search when include_tests=True."""
    regular_chunk = _make_chunk("ProcessData", "func ProcessData() {}", "main.go", is_test=False)
    test_chunk = _make_chunk("TestProcessData", "func TestProcessData() {}", "main_test.go", is_test=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([test_chunk], "myrepo", "main_test.go")

    results = store.hybrid_search("ProcessData", top_k=10, include_tests=True)
    assert len(results) == 2
    names = {r["metadata"]["name"] for r in results}
    assert names == {"ProcessData", "TestProcessData"}


def test_include_tests_works_with_repo_filter(store):
    """include_tests should work correctly when combined with repo_name filter."""
    regular_chunk = _make_chunk("Foo", "func Foo() {}", "main.go", is_test=False)
    test_chunk = _make_chunk("TestFoo", "func TestFoo() {}", "main_test.go", is_test=True)
    store.upsert_chunks([regular_chunk], "repo_a", "main.go")
    store.upsert_chunks([test_chunk], "repo_a", "main_test.go")

    # Exclude tests (default)
    results = store.semantic_search("Foo", top_k=10, repo_name="repo_a")
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "Foo"

    # Include tests
    results = store.semantic_search("Foo", top_k=10, repo_name="repo_a", include_tests=True)
    assert len(results) == 2


def test_only_test_chunks_returns_empty_when_excluded(store):
    """If only test chunks exist, excluding them should return empty results."""
    test_chunk = _make_chunk("TestOnly", "func TestOnly() {}", "only_test.go", is_test=True)
    store.upsert_chunks([test_chunk], "myrepo", "only_test.go")

    results = store.semantic_search("TestOnly", top_k=10)
    assert len(results) == 0

    results = store.keyword_search("TestOnly", top_k=10)
    assert len(results) == 0

    results = store.hybrid_search("TestOnly", top_k=10)
    assert len(results) == 0


# ------------------------------------------------------------------
# include_low_quality filter
# ------------------------------------------------------------------

def test_upsert_stores_low_quality_metadata(store):
    """Verify low_quality is stored in ChromaDB metadata."""
    low_quality_chunk = _make_chunk("err", "var err error", "main.go", low_quality=True)
    store.upsert_chunks([low_quality_chunk], "myrepo", "main.go")

    results = store.semantic_search("err", top_k=10, include_low_quality=True)
    assert len(results) == 1
    assert results[0]["metadata"]["low_quality"] is True


def test_upsert_stores_low_quality_in_sqlite(store):
    """Verify low_quality is stored in SQLite chunk_file_map."""
    low_quality_chunk = _make_chunk("err", "var err error", "main.go", low_quality=True)
    regular_chunk = _make_chunk("FindUser", "func FindUser() {}", "main.go", low_quality=False)
    store.upsert_chunks([low_quality_chunk, regular_chunk], "myrepo", "main.go")

    # Query SQLite directly to verify
    row = store._db.execute(
        "SELECT low_quality FROM chunk_file_map WHERE chunk_id = ?",
        (low_quality_chunk.id,)
    ).fetchone()
    assert row["low_quality"] == 1

    row = store._db.execute(
        "SELECT low_quality FROM chunk_file_map WHERE chunk_id = ?",
        (regular_chunk.id,)
    ).fetchone()
    assert row["low_quality"] == 0


def test_semantic_search_excludes_low_quality_by_default(store):
    """Low-quality chunks should be excluded by default."""
    regular_chunk = _make_chunk("FindUser", "func FindUser() {}", "main.go", low_quality=False)
    low_quality_chunk = _make_chunk("err", "var err error", "main.go", low_quality=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([low_quality_chunk], "myrepo", "errors.go")

    results = store.semantic_search("variable", top_k=10)
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "FindUser"


def test_semantic_search_includes_low_quality_when_flag_set(store):
    """Low-quality chunks should be included when include_low_quality=True."""
    regular_chunk = _make_chunk("FindUser", "func FindUser() {}", "main.go", low_quality=False)
    low_quality_chunk = _make_chunk("err", "var err error", "main.go", low_quality=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([low_quality_chunk], "myrepo", "errors.go")

    results = store.semantic_search("variable", top_k=10, include_low_quality=True)
    assert len(results) == 2
    names = {r["metadata"]["name"] for r in results}
    assert names == {"FindUser", "err"}


def test_keyword_search_excludes_low_quality_by_default(store):
    """Low-quality chunks should be excluded from keyword search by default."""
    regular_chunk = _make_chunk("FindUser", "func FindUser() { /* find user */ }", "main.go", low_quality=False)
    low_quality_chunk = _make_chunk("err", "func err() { /* find error */ }", "main.go", low_quality=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([low_quality_chunk], "myrepo", "errors.go")

    results = store.keyword_search("find", top_k=10)
    assert len(results) == 1
    assert results[0]["id"] == regular_chunk.id


def test_keyword_search_includes_low_quality_when_flag_set(store):
    """Low-quality chunks should be included in keyword search when include_low_quality=True."""
    regular_chunk = _make_chunk("FindUser", "func FindUser() { /* find user */ }", "main.go", low_quality=False)
    low_quality_chunk = _make_chunk("err", "func err() { /* find error */ }", "main.go", low_quality=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([low_quality_chunk], "myrepo", "errors.go")

    results = store.keyword_search("find", top_k=10, include_low_quality=True)
    assert len(results) == 2
    ids = {r["id"] for r in results}
    assert ids == {regular_chunk.id, low_quality_chunk.id}


def test_hybrid_search_excludes_low_quality_by_default(store):
    """Low-quality chunks should be excluded from hybrid search by default."""
    regular_chunk = _make_chunk("ProcessData", "func ProcessData() {}", "main.go", low_quality=False)
    low_quality_chunk = _make_chunk("ctx", "var ctx context.Context", "main.go", low_quality=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([low_quality_chunk], "myrepo", "context.go")

    results = store.hybrid_search("variable", top_k=10)
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "ProcessData"


def test_hybrid_search_includes_low_quality_when_flag_set(store):
    """Low-quality chunks should be included in hybrid search when include_low_quality=True."""
    regular_chunk = _make_chunk("ProcessData", "func ProcessData() {}", "main.go", low_quality=False)
    low_quality_chunk = _make_chunk("ctx", "var ctx context.Context", "main.go", low_quality=True)
    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([low_quality_chunk], "myrepo", "context.go")

    results = store.hybrid_search("variable", top_k=10, include_low_quality=True)
    assert len(results) == 2
    names = {r["metadata"]["name"] for r in results}
    assert names == {"ProcessData", "ctx"}


def test_include_low_quality_works_with_repo_filter(store):
    """include_low_quality should work correctly when combined with repo_name filter."""
    regular_chunk = _make_chunk("Foo", "func Foo() {}", "main.go", low_quality=False)
    low_quality_chunk = _make_chunk("err", "var err error", "errors.go", low_quality=True)
    store.upsert_chunks([regular_chunk], "repo_a", "main.go")
    store.upsert_chunks([low_quality_chunk], "repo_a", "errors.go")

    # Exclude low quality (default)
    results = store.semantic_search("variable", top_k=10, repo_name="repo_a")
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "Foo"

    # Include low quality
    results = store.semantic_search("variable", top_k=10, repo_name="repo_a", include_low_quality=True)
    assert len(results) == 2


def test_only_low_quality_chunks_returns_empty_when_excluded(store):
    """If only low-quality chunks exist, excluding them should return empty results."""
    low_quality_chunk = _make_chunk("err", "var err error", "main.go", low_quality=True)
    store.upsert_chunks([low_quality_chunk], "myrepo", "main.go")

    results = store.semantic_search("error", top_k=10)
    assert len(results) == 0

    results = store.keyword_search("err", top_k=10)
    assert len(results) == 0

    results = store.hybrid_search("error", top_k=10)
    assert len(results) == 0


def test_include_low_quality_and_include_tests_combined(store):
    """Both include_low_quality and include_tests should work together."""
    regular_chunk = _make_chunk("FindUser", "func FindUser() {}", "main.go", is_test=False, low_quality=False)
    test_chunk = _make_chunk("TestFindUser", "func TestFindUser() {}", "main_test.go", is_test=True, low_quality=False)
    low_quality_chunk = _make_chunk("err", "var err error", "main.go", is_test=False, low_quality=True)
    test_low_quality_chunk = _make_chunk("testErr", "var testErr error", "main_test.go", is_test=True, low_quality=True)

    store.upsert_chunks([regular_chunk], "myrepo", "main.go")
    store.upsert_chunks([test_chunk], "myrepo", "main_test.go")
    store.upsert_chunks([low_quality_chunk], "myrepo", "errors.go")
    store.upsert_chunks([test_low_quality_chunk], "myrepo", "errors_test.go")

    # Default: exclude both tests and low quality
    results = store.semantic_search("function", top_k=10)
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "FindUser"

    # Include tests only
    results = store.semantic_search("function", top_k=10, include_tests=True)
    assert len(results) == 2
    names = {r["metadata"]["name"] for r in results}
    assert names == {"FindUser", "TestFindUser"}

    # Include low quality only
    results = store.semantic_search("function", top_k=10, include_low_quality=True)
    assert len(results) == 2
    names = {r["metadata"]["name"] for r in results}
    assert names == {"FindUser", "err"}

    # Include both
    results = store.semantic_search("function", top_k=10, include_tests=True, include_low_quality=True)
    assert len(results) == 4
    names = {r["metadata"]["name"] for r in results}
    assert names == {"FindUser", "TestFindUser", "err", "testErr"}


# ------------------------------------------------------------------
# summary field tests
# ------------------------------------------------------------------

def test_upsert_stores_summary_in_sqlite(store):
    """Verify summary is stored in SQLite chunks_fts table."""
    chunk = _make_chunk("MyFunc", "func MyFunc() {}")
    chunk.summary = "This function does something useful for search."
    store.upsert_chunks([chunk], "myrepo", "main.go")

    # Query SQLite directly to verify
    row = store._db.execute(
        "SELECT summary FROM chunks_fts WHERE chunk_id = ?",
        (chunk.id,)
    ).fetchone()
    assert row["summary"] == "This function does something useful for search."


def test_upsert_stores_empty_summary_in_sqlite(store):
    """Verify empty summary is stored correctly."""
    chunk = _make_chunk("MyFunc", "func MyFunc() {}")
    # summary defaults to ""
    store.upsert_chunks([chunk], "myrepo", "main.go")

    row = store._db.execute(
        "SELECT summary FROM chunks_fts WHERE chunk_id = ?",
        (chunk.id,)
    ).fetchone()
    assert row["summary"] == ""


def test_keyword_search_matches_summary(store):
    """Keyword search should find matches in the summary field."""
    chunk = _make_chunk("MyFunc", "func MyFunc() {}")
    chunk.summary = "Handles HTTP requests and authentication tokens."
    store.upsert_chunks([chunk], "myrepo", "main.go")

    # Search for a term only in the summary
    results = store.keyword_search("authentication", top_k=10)
    assert len(results) == 1
    assert results[0]["id"] == chunk.id


def test_keyword_search_no_match_without_summary(store):
    """Keyword search should not find term if not in content, doc, or summary."""
    chunk = _make_chunk("MyFunc", "func MyFunc() {}")
    # No summary, term not in content or doc
    store.upsert_chunks([chunk], "myrepo", "main.go")

    results = store.keyword_search("authentication", top_k=10)
    assert len(results) == 0
