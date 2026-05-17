#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from src.embedding.ollama import OllamaEmbeddingFunction
from src.ingestion.indexer import Indexer
from src.ingestion.repository import GitRepository
from src.ingestion.store import ChunkStore

_DEFAULT_CONFIG = Path("configs/repositories.json")
_DEFAULT_CHROMA = Path("data/vector_store/chroma")
_DEFAULT_SQLITE = Path("data/vector_store/index.db")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index Go repositories into the RAG store."
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--chroma-path", type=Path, default=_DEFAULT_CHROMA)
    parser.add_argument("--sqlite-path", type=Path, default=_DEFAULT_SQLITE)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="nomic-embed-text:v1.5")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())

    args.chroma_path.mkdir(parents=True, exist_ok=True)
    args.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_fn = OllamaEmbeddingFunction(
        model=args.model, base_url=args.ollama_url
    )
    store = ChunkStore(
        chroma_path=args.chroma_path,
        sqlite_path=args.sqlite_path,
        embedding_fn=embedding_fn,
    )
    indexer = Indexer(store)

    failed: list[str] = []
    try:
        for repo_config in config["repositories"]:
            repo = GitRepository(
                name=repo_config["name"],
                url=repo_config["url"],
                branch=repo_config["branch"],
                clone_path=Path(repo_config["clone_path"]),
            )
            print(f"[{repo.name}] Indexing...")
            try:
                indexer.index_repository(repo)
                print(f"[{repo.name}] Done.")
            except Exception as e:
                print(f"[{repo.name}] Error: {e}", file=sys.stderr)
                failed.append(repo.name)
    finally:
        store.close()

    if failed:
        print(
            f"\n{len(failed)} repo(s) failed: {', '.join(failed)}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
