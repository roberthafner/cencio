#!/usr/bin/env python3
"""Generate a golden evaluation set by sampling chunks from the index and
prompting a local chat model to write a search query for each one."""
import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.embedding.ollama import OllamaEmbeddingFunction
from src.evaluation.golden_set import generate_golden_set, save_golden_set
from src.evaluation.ollama_chat import OllamaChatFunction
from src.ingestion.store import ChunkStore

_DEFAULT_OUTPUT = _PROJECT_ROOT / "tests" / "evaluation" / "golden_mux.json"
_DEFAULT_CHROMA = _PROJECT_ROOT / "build" / "database" / "chroma"
_DEFAULT_SQLITE = _PROJECT_ROOT / "build" / "database" / "index.db"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the golden evaluation set from the indexed gorilla/mux repo."
    )
    parser.add_argument("--repo", default="gorilla-mux",
                        help="Repository name as stored in the index (default: gorilla-mux)")
    parser.add_argument("--chat-model", default="devstral-small-2:latest",
                        help="Ollama chat model used to generate queries (default: devstral-small-2:latest)")
    parser.add_argument("--embed-model", default="nomic-embed-text:v1.5",
                        help="Ollama embedding model (default: nomic-embed-text:v1.5)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--chroma-path", type=Path, default=_DEFAULT_CHROMA)
    parser.add_argument("--sqlite-path", type=Path, default=_DEFAULT_SQLITE)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                        help=f"Output path for the golden set JSON (default: {_DEFAULT_OUTPUT})")
    parser.add_argument("--samples-per-type", type=int, default=5,
                        help="Number of chunks to sample per chunk type (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    args = parser.parse_args()

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
    print(f"  repo          : {args.repo}")
    print(f"  chat model    : {args.chat_model}")
    print(f"  samples/type  : {args.samples_per_type}")
    print(f"  output        : {args.output}")
    print()

    try:
        entries = generate_golden_set(
            store=store,
            chat_fn=chat_fn,
            repo_name=args.repo,
            samples_per_type=args.samples_per_type,
            seed=args.seed,
        )
        save_golden_set(entries, args.output)
    finally:
        store.close()

    print(f"\nSaved {len(entries)} entries → {args.output}")


if __name__ == "__main__":
    main()
