import pytest

from src.ingestion.indexer import Indexer
from src.ingestion.repository import GitRepository
from src.ingestion.store import ChunkStore


# ------------------------------------------------------------------
# Indexing
# ------------------------------------------------------------------

@pytest.mark.integration
def test_indexing_produces_chunks(mux_store: ChunkStore) -> None:
    hashes = mux_store.get_all_file_hashes("gorilla-mux")
    assert len(hashes) > 0


@pytest.mark.integration
def test_all_go_files_are_indexed(
    mux_store: ChunkStore, mux_repo: GitRepository
) -> None:
    hashes = mux_store.get_all_file_hashes("gorilla-mux")
    go_files = {str(f) for f in mux_repo.list_go_files()}
    assert go_files == set(hashes.keys())


@pytest.mark.integration
def test_indexed_files_have_chunks(mux_store: ChunkStore) -> None:
    hashes = mux_store.get_all_file_hashes("gorilla-mux")
    for file_path in hashes:
        count = mux_store.chunk_count_for_file("gorilla-mux", file_path)
        assert count > 0, f"No chunks indexed for {file_path}"


# ------------------------------------------------------------------
# Keyword search
# ------------------------------------------------------------------

@pytest.mark.integration
def test_keyword_search_finds_router(mux_store: ChunkStore) -> None:
    results = mux_store.keyword_search("Router", top_k=5)
    assert len(results) > 0


@pytest.mark.integration
def test_keyword_search_finds_serve_http(mux_store: ChunkStore) -> None:
    results = mux_store.keyword_search("ServeHTTP", top_k=5)
    assert len(results) > 0


@pytest.mark.integration
def test_keyword_search_no_match_returns_empty(mux_store: ChunkStore) -> None:
    results = mux_store.keyword_search("zzznomatchxyz")
    assert results == []


# ------------------------------------------------------------------
# Semantic search
# ------------------------------------------------------------------

@pytest.mark.integration
def test_semantic_search_returns_results(mux_store: ChunkStore) -> None:
    results = mux_store.semantic_search("HTTP request routing", top_k=5)
    assert len(results) > 0


@pytest.mark.integration
def test_semantic_search_result_structure(mux_store: ChunkStore) -> None:
    results = mux_store.semantic_search("router middleware", top_k=3)
    for r in results:
        assert "id" in r
        assert "metadata" in r
        assert "content" in r
        assert r["content"] != ""


@pytest.mark.integration
def test_semantic_search_repo_filter(mux_store: ChunkStore) -> None:
    results = mux_store.semantic_search(
        "HTTP routing", top_k=5, repo_name="gorilla-mux"
    )
    assert all(r["metadata"]["repo_name"] == "gorilla-mux" for r in results)


# ------------------------------------------------------------------
# Hybrid search
# ------------------------------------------------------------------

@pytest.mark.integration
def test_hybrid_search_returns_results(mux_store: ChunkStore) -> None:
    results = mux_store.hybrid_search("route matching", top_k=5)
    assert len(results) > 0


@pytest.mark.integration
def test_hybrid_search_result_structure(mux_store: ChunkStore) -> None:
    results = mux_store.hybrid_search("Router", top_k=5)
    for r in results:
        assert "id" in r
        assert "metadata" in r
        assert "content" in r
        meta = r["metadata"]
        assert "name" in meta
        assert "chunk_type" in meta
        assert "file_path" in meta
        assert "repo_name" in meta


@pytest.mark.integration
def test_hybrid_search_repo_filter(mux_store: ChunkStore) -> None:
    results = mux_store.hybrid_search(
        "Router", top_k=5, repo_name="gorilla-mux"
    )
    assert all(r["metadata"]["repo_name"] == "gorilla-mux" for r in results)


# ------------------------------------------------------------------
# Incremental indexing
# ------------------------------------------------------------------

@pytest.mark.integration
def test_second_run_with_no_changes_is_noop(
    mux_store: ChunkStore, mux_repo: GitRepository
) -> None:
    hashes_before = mux_store.get_all_file_hashes("gorilla-mux")

    indexer = Indexer(mux_store)
    indexer.index_repository(mux_repo)

    hashes_after = mux_store.get_all_file_hashes("gorilla-mux")
    assert hashes_before == hashes_after
