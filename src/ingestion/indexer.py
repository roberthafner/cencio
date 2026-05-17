from pathlib import Path

from src.ingestion.repository import GitRepository
from src.ingestion.store import ChunkStore
from src.parsing.go_parser import parse_file


class Indexer:
    def __init__(self, store: ChunkStore) -> None:
        self._store = store

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
            self._store.upsert_chunks(chunks, repo.name, rel_path)
            self._store.set_file_hash(repo.name, rel_path, current_hashes[rel_path])
