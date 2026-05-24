from pathlib import Path
from typing import Optional

from src.evaluation.ollama_chat import ChatFunction
from src.ingestion.repository import GitRepository
from src.ingestion.store import ChunkStore
from src.ingestion.summarizer import enrich_chunks_with_summaries, needs_summary
from src.parsing.go_parser import parse_file

# Names that are too generic to be meaningfully retrieved
LOW_QUALITY_NAMES = frozenset({
    "",        # unnamed blocks
    "_",       # blank identifier
    "err",     # ubiquitous error variable
    "ctx",     # context variable
    "ok",      # boolean check variable
    "i", "j", "k", "n", "m",  # loop counters
    "H",       # single-letter test harness
    "T",       # generic type parameter
})


def is_low_quality_chunk(name: str, chunk_type: str, doc: str = "") -> bool:
    """Return True if this chunk is too generic to retrieve meaningfully.

    Args:
        name: The name/identifier of the chunk
        chunk_type: The type of chunk (e.g., "var", "const", "function")
        doc: The documentation string for the chunk

    Returns:
        True if the chunk is considered low quality for retrieval purposes
    """
    # Unnamed blocks with good documentation are retrievable
    if not name and doc and len(doc.strip()) >= 50:
        return False
    if name in LOW_QUALITY_NAMES:
        return True
    # Single-letter names are usually not retrievable
    if len(name) == 1:
        return True
    # Very short names for var/const are often generic
    if chunk_type in ("var", "const") and len(name) <= 2:
        return True
    return False


class Indexer:
    def __init__(
        self,
        store: ChunkStore,
        chat_fn: Optional[ChatFunction] = None,
        verbose: bool = False,
    ) -> None:
        self._store = store
        self._chat_fn = chat_fn
        self._verbose = verbose

    def _log(self, repo_name: str, message: str) -> None:
        """Print a log message if verbose mode is enabled."""
        if self._verbose:
            print(f"[{repo_name}] {message}")

    def index_repository(self, repo: GitRepository) -> None:
        self._log(repo.name, "Cloning/pulling repository...")
        repo.clone_or_pull()

        self._log(repo.name, "Scanning for changes...")
        current_files = repo.list_go_files()
        current_hashes = {
            str(f): repo.file_content_hash(f) for f in current_files
        }
        stored_hashes = self._store.get_all_file_hashes(repo.name)

        current_paths = set(current_hashes)
        stored_paths = set(stored_hashes)

        deleted = stored_paths - current_paths
        modified = {
            p for p in current_paths & stored_paths
            if stored_hashes[p] != current_hashes[p]
        }
        new = current_paths - stored_paths

        self._log(repo.name, f"  deleted: {len(deleted)} files")
        self._log(repo.name, f"  modified: {len(modified)} files")
        self._log(repo.name, f"  new: {len(new)} files")

        for rel_path in deleted | modified:
            self._store.delete_chunks_for_file(repo.name, rel_path)
            self._store.delete_file_tracking(repo.name, rel_path)

        files_to_process = sorted(modified | new)
        if files_to_process:
            self._log(repo.name, f"Processing {len(files_to_process)} files...")

        total_chunks = 0
        total_summarized = 0

        for rel_path in files_to_process:
            abs_path = repo.clone_path / rel_path
            chunks = parse_file(str(abs_path))

            # Mark low-quality chunks first (before summarization)
            for chunk in chunks:
                chunk.low_quality = is_low_quality_chunk(
                    chunk.name, chunk.type.value, chunk.doc
                )

            # Count how many chunks need summaries
            summaries_needed = sum(1 for c in chunks if needs_summary(c))

            if self._verbose:
                print(f"[{repo.name}]   {rel_path}: {len(chunks)} chunks, {summaries_needed} need summaries")

            # Enrich chunks with LLM summaries if chat function is available
            summaries_before = sum(1 for c in chunks if c.summary)
            if self._chat_fn is not None:
                chunks = enrich_chunks_with_summaries(
                    chunks, self._chat_fn, verbose=self._verbose
                )
            summaries_after = sum(1 for c in chunks if c.summary)

            total_chunks += len(chunks)
            total_summarized += (summaries_after - summaries_before)

            self._store.upsert_chunks(chunks, repo.name, rel_path)
            self._store.set_file_hash(repo.name, rel_path, current_hashes[rel_path])

        if files_to_process:
            self._log(repo.name, f"Complete: {len(files_to_process)} files, {total_chunks} chunks, {total_summarized} summarized")
