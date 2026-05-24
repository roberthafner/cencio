"""LLM-based code summarization for improving chunk retrieval quality."""

from src.evaluation.ollama_chat import ChatFunction
from src.models.chunk import Chunk, ChunkType

# Chunk types that benefit from LLM-generated summaries
_SUMMARIZABLE_TYPES = frozenset({
    ChunkType.FUNCTION,
    ChunkType.METHOD,
    ChunkType.STRUCT,
    ChunkType.INTERFACE,
})

_SUMMARY_PROMPT = """\
Summarize this Go code for a semantic search index. Write 1-2 sentences that:
1. Describe what this code does and when you would use it
2. Include common terms a developer might search for

Type: {chunk_type}
Name: {chunk_name}
Package: {package_name}

```go
{content}
```

Respond with only the summary. No quotes, no explanation, no preamble."""


def needs_summary(chunk: Chunk) -> bool:
    """Return True if this chunk would benefit from an LLM-generated summary.

    Low-quality chunks are skipped since they are excluded from search results.
    Only certain chunk types (function, method, struct, interface) are summarized.
    """
    # Don't summarize low-quality chunks - they're excluded from search anyway
    if chunk.low_quality:
        return False
    # Only summarize certain chunk types
    if chunk.type not in _SUMMARIZABLE_TYPES:
        return False
    # Unnamed chunks (blocks)
    if not chunk.name:
        return True
    # No documentation at all
    if not chunk.doc:
        return True
    # Very short doc that's probably not helpful
    if len(chunk.doc.strip()) < 30:
        return True
    return False


def generate_summary(chunk: Chunk, chat_fn: ChatFunction) -> str:
    """Generate a brief summary of what this code does."""
    prompt = _SUMMARY_PROMPT.format(
        chunk_type=chunk.type.value,
        chunk_name=chunk.name or "(unnamed)",
        package_name=chunk.package_name,
        content=chunk.content[:1500],
    )
    try:
        return chat_fn(prompt).strip()
    except Exception:
        # If summarization fails, return empty string rather than crashing
        return ""


def enrich_chunks_with_summaries(
    chunks: list[Chunk],
    chat_fn: ChatFunction,
    verbose: bool = False,
) -> list[Chunk]:
    """Add LLM-generated summaries to chunks that need them.

    Summaries are stored in the chunk's summary field, separate from the
    original doc field.
    """
    enriched = []
    for chunk in chunks:
        if needs_summary(chunk):
            summary = generate_summary(chunk, chat_fn)
            if summary:
                chunk.summary = summary
                if verbose:
                    name = chunk.name or "(unnamed)"
                    print(f"    + summarized {chunk.type.value} '{name}'")
        enriched.append(chunk)
    return enriched
