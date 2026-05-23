#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from pathlib import Path

import chromadb

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_CHROMA = _PROJECT_ROOT / "data" / "vector_store" / "chroma"
_DEFAULT_SQLITE = _PROJECT_ROOT / "data" / "vector_store" / "index.db"
_COLLECTION_NAME = "chunks"
_HEAVY = "═" * 72
_VALID_TYPES = [
    "package", "block", "function", "method", "struct",
    "interface", "const", "var", "type_alias",
]

# Names that are too generic to be meaningfully retrieved
_LOW_QUALITY_NAMES = frozenset({
    "",        # unnamed blocks
    "_",       # blank identifier
    "err",     # ubiquitous error variable
    "ctx",     # context variable
    "ok",      # boolean check variable
    "i", "j", "k", "n", "m",  # loop counters
    "H",       # single-letter test harness
    "T",       # generic type parameter
})


def _is_low_quality_chunk(name: str, chunk_type: str) -> bool:
    """Return True if this chunk is too generic to evaluate meaningfully."""
    if name in _LOW_QUALITY_NAMES:
        return True
    # Single-letter names are usually not retrievable
    if len(name) == 1:
        return True
    # Very short names for var/const are often generic
    if chunk_type in ("var", "const") and len(name) <= 2:
        return True
    return False


def _sqlite_extras(db: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = db.execute(
        f"SELECT chunk_id, doc, signature FROM chunks_fts"
        f" WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    return {
        row["chunk_id"]: {"doc": row["doc"] or "", "signature": row["signature"] or ""}
        for row in rows
    }


def _build_where(repo: str | None, chunk_type: str | None) -> dict | None:
    conditions = []
    if repo:
        conditions.append({"repo_name": {"$eq": repo}})
    if chunk_type:
        conditions.append({"chunk_type": {"$eq": chunk_type}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _print_chunk(
    index: int,
    total: int,
    chunk_id: str,
    meta: dict,
    content: str,
    extra: dict,
) -> None:
    chunk_type = meta.get("chunk_type", "unknown")
    name = meta.get("name") or "(unnamed)"
    package = meta.get("package_name", "")
    repo = meta.get("repo_name", "")
    file_path = meta.get("file_path", "")
    start = meta.get("start_line", "?")
    end = meta.get("end_line", "?")
    doc = extra.get("doc", "").strip()
    signature = extra.get("signature", "").strip()

    print(_HEAVY)
    header = f" {index}/{total}  {chunk_type}  {name}"
    if package:
        header += f"  ·  package {package}"
    print(header)
    print(_HEAVY)
    print()

    print("── ChromaDB " + "─" * 60)
    print(f"  id   : {chunk_id}")
    print(f"  repo : {repo}")
    print(f"  file : {file_path}:{start}–{end}")
    print()
    for line in content.splitlines():
        print(f"  {line}")
    print()

    print("── SQLite " + "─" * 62)
    if doc:
        print("  doc       :")
        for line in doc.splitlines():
            print(f"    {line}")
    else:
        print("  doc       : (empty)")
    print(f"  signature : {signature or '(empty)'}")
    print()


def _print_low_quality_summary(
    chunks: list[tuple[str, dict, str]],
) -> None:
    """Print a summary table of low-quality chunks."""
    print(f"Found {len(chunks)} low-quality chunks:\n")
    print(f"{'Type':<12} {'Name':<20} {'File':<50} {'Chunk ID'}")
    print("─" * 120)

    by_type: dict[str, int] = {}

    for chunk_id, meta, doc in chunks:
        chunk_type = meta.get("chunk_type", "unknown")
        name = meta.get("name") or "(unnamed)"
        file_path = meta.get("file_path", "")

        by_type[chunk_type] = by_type.get(chunk_type, 0) + 1
        print(f"{chunk_type:<12} {name:<20} {file_path:<50} {chunk_id}")

    print()
    print("Summary by type:")
    for chunk_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {chunk_type}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect chunks stored in ChromaDB and SQLite."
    )

    filters = parser.add_argument_group("filters (at least one required)")
    filters.add_argument("--id", dest="chunk_id", metavar="ID",
                         help="exact chunk ID")
    filters.add_argument("--name", metavar="STR",
                         help="substring match on symbol name")
    filters.add_argument("--repo", metavar="NAME",
                         help="exact repository name")
    filters.add_argument("--file", metavar="STR",
                         help="substring match on file path")
    filters.add_argument("--type", choices=_VALID_TYPES, dest="chunk_type",
                         metavar="TYPE",
                         help=f"chunk type: {', '.join(_VALID_TYPES)}")
    filters.add_argument("--package", metavar="STR",
                         help="substring match on package name")
    filters.add_argument("--low-quality", action="store_true",
                         help="show chunks with generic names that are hard to retrieve")

    parser.add_argument("--limit", type=int, default=20, metavar="N",
                        help="max chunks to display (default: 20)")
    parser.add_argument("--chroma-path", type=Path, default=_DEFAULT_CHROMA)
    parser.add_argument("--sqlite-path", type=Path, default=_DEFAULT_SQLITE)
    args = parser.parse_args()

    if not args.low_quality and not any([args.chunk_id, args.name, args.repo, args.file,
                args.chunk_type, args.package]):
        parser.error(
            "at least one filter is required: "
            "--id, --name, --repo, --file, --type, --package, or --low-quality"
        )

    try:
        chroma = chromadb.PersistentClient(path=str(args.chroma_path))
        collection = chroma.get_collection(name=_COLLECTION_NAME)
    except Exception as exc:
        print(f"error: could not open ChromaDB at {args.chroma_path}: {exc}",
              file=sys.stderr)
        sys.exit(1)

    try:
        db = sqlite3.connect(str(args.sqlite_path), check_same_thread=False)
        db.row_factory = sqlite3.Row
    except Exception as exc:
        print(f"error: could not open SQLite at {args.sqlite_path}: {exc}",
              file=sys.stderr)
        sys.exit(1)

    try:
        if args.low_quality:
            # Fetch all chunks and filter for low-quality ones
            where = _build_where(args.repo, args.chunk_type)
            kwargs: dict = dict(include=["metadatas", "documents"])
            if where:
                kwargs["where"] = where

            result = collection.get(**kwargs)

            # Filter for low-quality chunks
            low_quality = [
                (cid, meta, doc)
                for cid, meta, doc in zip(result["ids"], result["metadatas"], result["documents"])
                if _is_low_quality_chunk(
                    meta.get("name", ""),
                    meta.get("chunk_type", "")
                )
            ]

            if not low_quality:
                print("No low-quality chunks found.")
                return

            _print_low_quality_summary(low_quality[:args.limit])
            return

        if args.chunk_id:
            result = collection.get(
                ids=[args.chunk_id], include=["metadatas", "documents"]
            )
            ids: list[str] = result["ids"]
            metas: list[dict] = result["metadatas"]
            docs: list[str] = result["documents"]
        else:
            where = _build_where(args.repo, args.chunk_type)
            kwargs: dict = dict(include=["metadatas", "documents"])
            if where:
                kwargs["where"] = where

            result = collection.get(**kwargs)
            ids = result["ids"]
            metas = result["metadatas"]
            docs = result["documents"]

            # post-filter substring matches, then cap
            filtered = [
                (cid, meta, doc)
                for cid, meta, doc in zip(ids, metas, docs)
                if (not args.name or args.name.lower() in (meta.get("name") or "").lower())
                and (not args.file or args.file.lower() in (meta.get("file_path") or "").lower())
                and (not args.package or args.package.lower() in (meta.get("package_name") or "").lower())
            ]
            filtered = filtered[: args.limit]
            ids = [x[0] for x in filtered]
            metas = [x[1] for x in filtered]
            docs = [x[2] for x in filtered]

        if not ids:
            print("No chunks found matching the given filters.")
            return

        extras = _sqlite_extras(db, ids)
        total = len(ids)
        for i, (chunk_id, meta, content) in enumerate(zip(ids, metas, docs), 1):
            _print_chunk(i, total, chunk_id, meta, content, extras.get(chunk_id, {}))

    finally:
        db.close()


if __name__ == "__main__":
    main()
