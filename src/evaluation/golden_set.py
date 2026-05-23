import json
import random
from pathlib import Path

from src.evaluation.ollama_chat import ChatFunction
from src.ingestion.store import ChunkStore
from src.models.chunk import ChunkType

_CHUNK_TYPES = [t.value for t in ChunkType]

_PROMPT_TEMPLATE = """\
You are building a search evaluation dataset for a Go codebase RAG system.

Below is a Go code chunk from the {repo_name} repository:

Type: {chunk_type}
Name: {chunk_name}
File: {file_path}

```go
{content}
```

Write a single natural-language search query that a developer would type to find \
this specific piece of code. The query should:
- Include specific terms, identifiers, or concepts from the code itself
- Mention concrete functionality (e.g., "parse JSON response", "validate email format")
- Reference specific types, error names, or domain concepts when present
- Avoid generic phrases like "how to import" or "how to use" without specifics

Good example: "function that converts tenant state to orchestration status enum"
Bad example: "how to convert states in Go"

Respond with only the query. No explanation, no quotes, no trailing punctuation."""


def generate_golden_set(
    store: ChunkStore,
    chat_fn: ChatFunction,
    repo_name: str,
    samples_per_type: int = 5,
    seed: int = 42,
    exclude_test_files: bool = False,
) -> list[dict]:
    entries = []
    rng = random.Random(seed)

    for chunk_type in _CHUNK_TYPES:
        candidates = store.sample_chunks(
            repo_name, chunk_type=chunk_type, limit=samples_per_type * 4,
            exclude_test_files=exclude_test_files,
        )
        if not candidates:
            continue
        rng.shuffle(candidates)
        selected = candidates[:samples_per_type]

        for chunk in selected:
            meta = chunk["metadata"]
            content = chunk["content"][:2000]
            prompt = _PROMPT_TEMPLATE.format(
                repo_name=repo_name,
                chunk_type=meta.get("chunk_type", ""),
                chunk_name=meta.get("name", ""),
                file_path=meta.get("file_path", ""),
                content=content,
            )
            query = chat_fn(prompt)
            if not query:
                continue
            entries.append({
                "query": query,
                "chunk_id": chunk["id"],
                "chunk_name": meta.get("name", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "file_path": meta.get("file_path", ""),
                "repo_name": repo_name,
            })
        print(f"  {chunk_type}: {len(selected)} queries generated")

    return entries


def save_golden_set(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))


def load_golden_set(path: Path) -> list[dict]:
    return json.loads(path.read_text())
