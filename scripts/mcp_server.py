#!/usr/bin/env python3
"""MCP server exposing Cencio's hybrid code search as a callable tool."""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from src.embedding.ollama import OllamaEmbeddingFunction
from src.ingestion.store import ChunkStore

_DEFAULT_CHROMA = _PROJECT_ROOT / "data" / "vector_store" / "chroma"
_DEFAULT_SQLITE = _PROJECT_ROOT / "data" / "vector_store" / "index.db"

mcp = FastMCP("cencio")

_store: ChunkStore | None = None


def _get_store() -> ChunkStore:
    global _store
    if _store is None:
        chroma_path = Path(os.environ.get("CENCIO_CHROMA_PATH", str(_DEFAULT_CHROMA)))
        sqlite_path = Path(os.environ.get("CENCIO_SQLITE_PATH", str(_DEFAULT_SQLITE)))
        ollama_url = os.environ.get("CENCIO_OLLAMA_URL", "http://localhost:11434")
        model = os.environ.get("CENCIO_EMBED_MODEL", "nomic-embed-text:v1.5")
        embedding_fn = OllamaEmbeddingFunction(model=model, base_url=ollama_url)
        _store = ChunkStore(
            chroma_path=chroma_path,
            sqlite_path=sqlite_path,
            embedding_fn=embedding_fn,
        )
    return _store


def _format_results(results: list[dict], query: str, mode: str) -> str:
    if not results:
        return f'No results found for "{query}" (mode: {mode}).'

    lines = [f'Search results for "{query}" — mode: {mode}, {len(results)} result(s)\n']
    sep = "─" * 72

    for i, result in enumerate(results, 1):
        meta = result.get("metadata") or {}
        chunk_type = meta.get("chunk_type", "unknown")
        name = meta.get("name") or "(unnamed)"
        package = meta.get("package_name", "")
        repo = meta.get("repo_name", "")
        file_path = meta.get("file_path", "")
        start = meta.get("start_line", "?")
        end = meta.get("end_line", "?")
        content = result.get("content", "")

        header = f"[{i}] {chunk_type}  {name}"
        if package:
            header += f"  ·  package {package}"

        location_parts = []
        if repo:
            location_parts.append(repo)
        if file_path:
            location_parts.append(f"{file_path}:{start}–{end}")
        location = " · ".join(location_parts)

        lines.append(header)
        if location:
            lines.append(f"    {location}")
        lines.append(f"    {sep}")
        for line in content.splitlines():
            lines.append(f"    {line}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def search_code(
    query: str,
    mode: str = "hybrid",
    repo: str | None = None,
    top_k: int = 5,
) -> str:
    """Search indexed Go source code.

    Args:
        query: Natural-language or keyword search query.
        mode: "hybrid" (default), "semantic", or "keyword".
        repo: Repository name to restrict results to (optional).
        top_k: Number of results to return (default 5, max 20).
    """
    top_k = min(top_k, 20)
    store = _get_store()
    if mode == "semantic":
        results = store.semantic_search(query, top_k, repo)
    elif mode == "keyword":
        results = store.keyword_search(query, top_k, repo)
    else:
        results = store.hybrid_search(query, top_k, repo)
    return _format_results(results, query, mode)


if __name__ == "__main__":
    mcp.run(transport="stdio")
