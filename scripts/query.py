#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.embedding.ollama import OllamaEmbeddingFunction
from src.ingestion.store import ChunkStore

_DEFAULT_CHROMA = _PROJECT_ROOT / "build" / "database" / "chroma"
_DEFAULT_SQLITE = _PROJECT_ROOT / "build" / "database" / "index.db"
_SEPARATOR = "─" * 72


def _print_results(results: list[dict], query: str, mode: str, top_k: int) -> None:
    count = len(results)
    repo_note = ""
    print(f'\nSearching "{query}" · {mode} · top {top_k}\n')

    if not results:
        print("No results found.")
        return

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

        header = f"[{i}] {chunk_type}"
        if name:
            header += f"  {name}"
        if package:
            header += f"  ·  package {package}"

        location = ""
        if repo:
            location += repo
        if file_path:
            location += f" · {file_path}:{start}–{end}"

        print(header)
        if location:
            print(f"    {location}")
        print(f"    {_SEPARATOR}")
        for line in content.splitlines():
            print(f"    {line}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the indexed Go codebase."
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "semantic", "keyword"],
        default="hybrid",
        help="Search mode (default: hybrid)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, metavar="N",
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--repo", default=None,
        help="Filter results to a specific repository name",
    )
    parser.add_argument("--chroma-path", type=Path, default=_DEFAULT_CHROMA)
    parser.add_argument("--sqlite-path", type=Path, default=_DEFAULT_SQLITE)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="nomic-embed-text:v1.5")
    args = parser.parse_args()

    embedding_fn = OllamaEmbeddingFunction(
        model=args.model, base_url=args.ollama_url
    )
    store = ChunkStore(
        chroma_path=args.chroma_path,
        sqlite_path=args.sqlite_path,
        embedding_fn=embedding_fn,
    )

    try:
        if args.mode == "hybrid":
            results = store.hybrid_search(args.query, args.top_k, args.repo)
        elif args.mode == "semantic":
            results = store.semantic_search(args.query, args.top_k, args.repo)
        else:
            results = store.keyword_search(args.query, args.top_k, args.repo)

        _print_results(results, args.query, args.mode, args.top_k)
    finally:
        store.close()


if __name__ == "__main__":
    main()
