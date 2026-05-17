import sqlite3
from pathlib import Path

import chromadb

from src.embedding.ollama import EmbeddingFunction
from src.models.chunk import Chunk

_COLLECTION_NAME = "chunks"
_RRF_K = 60


class _ChromaEmbeddingAdapter:
    """Wraps our EmbeddingFunction protocol to match ChromaDB's calling convention."""

    def __init__(self, fn: EmbeddingFunction) -> None:
        self._fn = fn

    @staticmethod
    def name() -> str:
        return "custom"

    def is_legacy(self) -> bool:
        return True

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._fn(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._fn(input)


class ChunkStore:
    def __init__(
        self,
        chroma_path: Path,
        sqlite_path: Path,
        embedding_fn: EmbeddingFunction,
    ) -> None:
        self._embedding_fn = embedding_fn

        self._chroma = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = self._chroma.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=_ChromaEmbeddingAdapter(embedding_fn),
            metadata={"hnsw:space": "cosine"},
        )

        self._db = sqlite3.connect(str(sqlite_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        self._db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                content,
                name,
                package_name,
                doc,
                signature
            );

            CREATE TABLE IF NOT EXISTS chunk_file_map (
                chunk_id  TEXT NOT NULL,
                repo_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                PRIMARY KEY (chunk_id)
            );

            CREATE TABLE IF NOT EXISTS file_index (
                repo_name    TEXT NOT NULL,
                file_path    TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                PRIMARY KEY (repo_name, file_path)
            );
        """)
        self._db.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_chunks(
        self, chunks: list[Chunk], repo_name: str, file_path: str
    ) -> None:
        if not chunks:
            return

        try:
            for chunk in chunks:
                self._db.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk.id,)
                )
                self._db.execute(
                    "INSERT INTO chunks_fts"
                    "(chunk_id, content, name, package_name, doc, signature)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk.id, chunk.content, chunk.name,
                     chunk.package_name, chunk.doc, chunk.signature),
                )
                self._db.execute(
                    "INSERT OR REPLACE INTO chunk_file_map"
                    "(chunk_id, repo_name, file_path) VALUES (?, ?, ?)",
                    (chunk.id, repo_name, file_path),
                )

            self._collection.upsert(
                ids=[c.id for c in chunks],
                documents=[c.content for c in chunks],
                metadatas=[{
                    "repo_name": repo_name,
                    "file_path": file_path,
                    "chunk_type": c.type.value,
                    "name": c.name,
                    "package_name": c.package_name,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                } for c in chunks],
            )

            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def delete_chunks_for_file(self, repo_name: str, file_path: str) -> None:
        rows = self._db.execute(
            "SELECT chunk_id FROM chunk_file_map"
            " WHERE repo_name = ? AND file_path = ?",
            (repo_name, file_path),
        ).fetchall()

        chunk_ids = [row["chunk_id"] for row in rows]
        if not chunk_ids:
            return

        try:
            for chunk_id in chunk_ids:
                self._db.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,)
                )
            placeholders = ",".join("?" * len(chunk_ids))
            self._db.execute(
                f"DELETE FROM chunk_file_map WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )

            self._collection.delete(ids=chunk_ids)

            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def set_file_hash(
        self, repo_name: str, file_path: str, content_hash: str
    ) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO file_index"
                "(repo_name, file_path, content_hash) VALUES (?, ?, ?)",
                (repo_name, file_path, content_hash),
            )

    def delete_file_tracking(self, repo_name: str, file_path: str) -> None:
        with self._db:
            self._db.execute(
                "DELETE FROM file_index WHERE repo_name = ? AND file_path = ?",
                (repo_name, file_path),
            )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_all_file_hashes(self, repo_name: str) -> dict[str, str]:
        rows = self._db.execute(
            "SELECT file_path, content_hash FROM file_index WHERE repo_name = ?",
            (repo_name,),
        ).fetchall()
        return {row["file_path"]: row["content_hash"] for row in rows}

    def chunk_count_for_file(self, repo_name: str, file_path: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM chunk_file_map"
            " WHERE repo_name = ? AND file_path = ?",
            (repo_name, file_path),
        ).fetchone()
        return row[0]

    def semantic_search(
        self, query: str, top_k: int = 10, repo_name: str | None = None
    ) -> list[dict]:
        count = self._collection.count()
        n = min(top_k, count)
        if n == 0:
            return []

        kwargs: dict = dict(
            query_texts=[query],
            n_results=n,
            include=["metadatas", "documents", "distances"],
        )
        if repo_name is not None:
            kwargs["where"] = {"repo_name": repo_name}

        results = self._collection.query(**kwargs)
        return [
            {
                "id": id_,
                "metadata": meta,
                "content": doc,
                "distance": dist,
            }
            for id_, meta, doc, dist in zip(
                results["ids"][0],
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0],
            )
        ]

    def keyword_search(
        self, query: str, top_k: int = 10, repo_name: str | None = None
    ) -> list[dict]:
        if repo_name is not None:
            rows = self._db.execute(
                "SELECT chunk_id, rank FROM chunks_fts"
                " WHERE chunks_fts MATCH ?"
                "   AND chunk_id IN"
                "       (SELECT chunk_id FROM chunk_file_map WHERE repo_name = ?)"
                " ORDER BY rank LIMIT ?",
                (query, repo_name, top_k),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT chunk_id, rank FROM chunks_fts"
                " WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, top_k),
            ).fetchall()
        return [{"id": row["chunk_id"], "rank": row["rank"]} for row in rows]

    def hybrid_search(
        self, query: str, top_k: int = 10, repo_name: str | None = None
    ) -> list[dict]:
        semantic = self.semantic_search(query, top_k, repo_name)
        keyword = self.keyword_search(query, top_k, repo_name)

        scores: dict[str, float] = {}
        for rank, result in enumerate(semantic):
            id_ = result["id"]
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (_RRF_K + rank + 1)
        for rank, result in enumerate(keyword):
            id_ = result["id"]
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (_RRF_K + rank + 1)

        ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]

        # Build result map; semantic results already carry full metadata
        result_by_id = {r["id"]: r for r in semantic}

        # Fetch metadata from ChromaDB for any keyword-only hits
        missing = [id_ for id_ in ranked_ids if id_ not in result_by_id]
        if missing:
            fetched = self._collection.get(
                ids=missing,
                include=["metadatas", "documents"],
            )
            for id_, meta, doc in zip(
                fetched["ids"], fetched["metadatas"], fetched["documents"]
            ):
                result_by_id[id_] = {"id": id_, "metadata": meta, "content": doc}

        return [result_by_id[id_] for id_ in ranked_ids if id_ in result_by_id]

    def close(self) -> None:
        self._db.close()
