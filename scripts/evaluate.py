#!/usr/bin/env python3
"""Run the retrieval evaluation harness against a golden set."""
import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.embedding.ollama import OllamaEmbeddingFunction
from src.evaluation.golden_set import load_golden_set
from src.evaluation.harness import evaluate
from src.ingestion.store import ChunkStore

_DEFAULT_GOLDEN = _PROJECT_ROOT / "tests" / "evaluation" / "golden_mux.json"
_DEFAULT_CHROMA = _PROJECT_ROOT / "build" / "database" / "chroma"
_DEFAULT_SQLITE = _PROJECT_ROOT / "build" / "database" / "index.db"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against the golden set."
    )
    parser.add_argument("--golden-set", type=Path, default=_DEFAULT_GOLDEN,
                        help=f"Path to the golden set JSON (default: {_DEFAULT_GOLDEN})")
    parser.add_argument("--mode", default="hybrid",
                        choices=["hybrid", "semantic", "keyword"],
                        help="Search mode to evaluate (default: hybrid)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to retrieve per query (default: 5)")
    parser.add_argument("--embed-model", default="nomic-embed-text:v1.5")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--chroma-path", type=Path, default=_DEFAULT_CHROMA)
    parser.add_argument("--sqlite-path", type=Path, default=_DEFAULT_SQLITE)
    args = parser.parse_args()

    if not args.golden_set.exists():
        print(f"Golden set not found: {args.golden_set}", file=sys.stderr)
        print("Run scripts/generate_golden_set.py first.", file=sys.stderr)
        sys.exit(1)

    golden_set = load_golden_set(args.golden_set)
    embedding_fn = OllamaEmbeddingFunction(
        model=args.embed_model, base_url=args.ollama_url
    )
    store = ChunkStore(
        chroma_path=args.chroma_path,
        sqlite_path=args.sqlite_path,
        embedding_fn=embedding_fn,
    )

    try:
        report = evaluate(store, golden_set, mode=args.mode, top_k=args.top_k)
    finally:
        store.close()

    report.print()


if __name__ == "__main__":
    main()
