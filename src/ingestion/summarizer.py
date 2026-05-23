"""LLM-based code summarization for improving chunk retrieval quality."""

from src.evaluation.ollama_chat import ChatFunction
from src.models.chunk import Chunk

_SUMMARY_PROMPT = """\
Summarize this Go code in 1-2 sentences. Focus on what it does and its purpose, not implementation details.

Type: {chunk_type}
Name: {chunk_name}
Package: {package_name}

```go
{content}
```

Respond with only the summary. No quotes, no explanation, no preamble."""


def needs_summary(chunk: Chunk) -> bool:
    """Return True if this chunk would benefit from an LLM-generated summary."""
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

    Summaries are appended to the chunk's doc field.
    """
    enriched = []
    for chunk in chunks:
        if needs_summary(chunk):
            summary = generate_summary(chunk, chat_fn)
            if summary:
                if chunk.doc:
                    chunk.doc = f"{chunk.doc.strip()}\n\n{summary}"
                else:
                    chunk.doc = summary
                if verbose:
                    name = chunk.name or "(unnamed)"
                    print(f"    + summarized {chunk.type.value} '{name}'")
        enriched.append(chunk)
    return enriched
