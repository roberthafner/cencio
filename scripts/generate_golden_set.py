#!/usr/bin/env python3
"""Generate a golden evaluation set by sampling chunks from the index and
prompting a local chat model to write a search query for each one."""
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.embedding.ollama import OllamaEmbeddingFunction
from src.evaluation.golden_set import generate_golden_set, save_golden_set
from src.evaluation.ollama_chat import OllamaChatFunction
from src.ingestion.store import ChunkStore

_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "repositories.json"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "tests" / "evaluation" / "golden.json"
_DEFAULT_CHROMA = _PROJECT_ROOT / "data" / "vector_store" / "chroma"
_DEFAULT_SQLITE = _PROJECT_ROOT / "data" / "vector_store" / "index.db"


def _load_repo_names(config_path: Path) -> list[str]:
    """Load repository names from the config file."""
    config = json.loads(config_path.read_text())
    return [repo["name"] for repo in config.get("repositories", [])]


def main() -> None:
    repo_names = _load_repo_names(_DEFAULT_CONFIG)

    parser = argparse.ArgumentParser(
        description="Generate the golden evaluation set from indexed repositories."
    )
    parser.add_argument("--repo", action="append", dest="repos", metavar="NAME",
                        help=f"Repository name(s) to include. Can be specified multiple times. "
                             f"(default: all from config: {', '.join(repo_names)})")
    parser.add_argument("--chat-model", default="devstral-small-2:latest",
                        help="Ollama chat model used to generate queries (default: devstral-small-2:latest)")
    parser.add_argument("--embed-model", default="nomic-embed-text:v1.5",
                        help="Ollama embedding model (default: nomic-embed-text:v1.5)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--chroma-path", type=Path, default=_DEFAULT_CHROMA)
    parser.add_argument("--sqlite-path", type=Path, default=_DEFAULT_SQLITE)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                        help=f"Path to repositories config file (default: {_DEFAULT_CONFIG})")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                        help=f"Output path for the golden set JSON (default: {_DEFAULT_OUTPUT})")
    parser.add_argument("--samples-per-type", type=int, default=5,
                        help="Number of chunks to sample per chunk type (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--exclude-test-files", action="store_true",
                        help="Exclude chunks from *_test.go files")
    args = parser.parse_args()

    # Use all repos from config if none specified
    repos_to_process = args.repos if args.repos else repo_names
    if not repos_to_process:
        parser.error("No repositories found in config file. Please specify --repo.")

    embedding_fn = OllamaEmbeddingFunction(
        model=args.embed_model, base_url=args.ollama_url
    )
    store = ChunkStore(
        chroma_path=args.chroma_path,
        sqlite_path=args.sqlite_path,
        embedding_fn=embedding_fn,
    )
    chat_fn = OllamaChatFunction(model=args.chat_model, base_url=args.ollama_url)

    print(f"Generating golden set")
    print(f"  repos         : {', '.join(repos_to_process)}")
    print(f"  chat model    : {args.chat_model}")
    print(f"  samples/type  : {args.samples_per_type}")
    print(f"  exclude tests : {args.exclude_test_files}")
    print(f"  output        : {args.output}")
    print()

    try:
        all_entries = []
        for repo_name in repos_to_process:
            print(f"[{repo_name}]")
            entries = generate_golden_set(
                store=store,
                chat_fn=chat_fn,
                repo_name=repo_name,
                samples_per_type=args.samples_per_type,
                seed=args.seed,
                exclude_test_files=args.exclude_test_files,
            )
            all_entries.extend(entries)
            print()
        save_golden_set(all_entries, args.output)
    finally:
        store.close()

    print(f"Saved {len(all_entries)} entries → {args.output}")


if __name__ == "__main__":
    main()
