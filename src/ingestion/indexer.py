from pathlib import Path
from typing import Optional

from src.evaluation.ollama_chat import ChatFunction
from src.ingestion.repository import GitRepository
from src.ingestion.store import ChunkStore
from src.ingestion.summarizer import enrich_chunks_with_summaries
from src.parsing.go_parser import parse_file


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

    def index_repository(self, repo: GitRepository) -> None:
        repo.clone_or_pull()

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

        for rel_path in deleted | modified:
            self._store.delete_chunks_for_file(repo.name, rel_path)
            self._store.delete_file_tracking(repo.name, rel_path)

        for rel_path in modified | new:
            abs_path = repo.clone_path / rel_path
            chunks = parse_file(str(abs_path))

            # Enrich chunks with LLM summaries if chat function is available
            if self._chat_fn is not None:
                chunks = enrich_chunks_with_summaries(
                    chunks, self._chat_fn, verbose=self._verbose
                )

            self._store.upsert_chunks(chunks, repo.name, rel_path)
            self._store.set_file_hash(repo.name, rel_path, current_hashes[rel_path])
