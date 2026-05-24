import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Any


class ChunkType(Enum):
    """Represents the semantic category of the code snippet."""
    PACKAGE = "package"
    BLOCK = "block"  # Used for const or var blocks
    FUNCTION = "function"
    METHOD = "method"
    STRUCT = "struct"
    INTERFACE = "interface"
    CONST = "const"
    VAR = "var"
    TYPE_ALIAS = "type_alias"


@dataclass
class Chunk:
    """
    Represents a semantically parsed unit of Go source code.

    This class is designed for use in a RAG (Retrieval-Augmented Generation)
    pipeline, providing both the raw code and the metadata necessary for
    high-precision retrieval and context injection.
    """
    # Unique identifier for the chunk. Should be generated via generate_id().
    id: str

    # The semantic type of the chunk (e.g., function, struct).
    type: ChunkType

    # The actual source code snippet.
    # IMPORTANT: This string should include the Signature and the Docstring
    # to provide maximum context to the LLM during retrieval.
    content: str

    # Name of the entity (e.g., the function name or struct name).
    name: str

    # The Go package name where this chunk resides.
    package_name: str

    # The relative path to the source file.
    file_path: str

    # The start line number in the source file.
    start_line: int

    # The end line number in the source and end.
    end_line: int

    # Indicates whether this chunk comes from test code (e.g., _test.go files
    # or files that import the "testing" package).
    is_test: bool

    # Indicates whether this chunk has a generic name that makes it hard to
    # retrieve accurately (e.g., single-letter names, common identifiers like
    # "err", "ctx", "_").
    low_quality: bool

    # The identifier of the parent scope (e.g., the Struct ID if this is a Method).
    # This allows for explicit hierarchical relationship traversal.
    parent_id: Optional[str] = None

    # List of IDs of the children within this scope (e.g., methods within a struct).
    children_ids: List[str] = field(default_factory=list)

    # The associated documentation/comments.
    doc: str = ""

    # The semantic signature (e.g., "func(ctx context.Context, id string) (*User, error)").
    signature: str = ""

    # A list of package names referenced within this chunk.
    # Used to build the dependency graph for the RAG system.
    imported_symbols: List[str] = field(default_factory=list)

    # Extensible metadata for additional information.
    metadata: Dict[str, str] = field(default_factory=dict)

    # LLM-generated summary for improving semantic search discoverability.
    # This is separate from `doc` to preserve original human-written documentation.
    summary: str = ""

    # Timestamp of when the chunk was created/parsed.
    created_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Chunk instance to a dictionary for JSON serialization."""
        # Convert enum to string for JSON compatibility
        data = asdict(self)
        data['type'] = self.type.value
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Creates a Chunk instance from a dictionary."""
        # Handle the enum conversion
        if 'type' in data:
            data['type'] = ChunkType(data['type'])
        if 'created_at' in data:
            data['created_at'] = datetime.datetime.fromisoformat(data['created_at'])
        return cls(**data)


def generate_id(content: str) -> str:
    """
    Computes a SHA-256 hash of the provided content to use as a Chunk ID.

    This implementation ensures ID idempotency: re-parsing the same source
    code content will always produce the same ID, preventing duplicate
    entries in the vector database during incremental indexing.

    Args:
        content: The raw string content of the chunk.

    Returns:
        A hex string representing the SHA-256 hash.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
