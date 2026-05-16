from pathlib import Path

import pytest

from src.models.chunk import ChunkType, generate_id
from src.parsing.go_parser import parse_file

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "golang" / "package_example.go"

EXPECTED_DOC = (
    "// Package users provides functionality for managing user accounts\n"
    "// within the Cencio RAG pipeline. It handles creation, retrieval,\n"
    "// and structured representation of user records."
)

EXPECTED_CONTENT = (
    "// Package users provides functionality for managing user accounts\n"
    "// within the Cencio RAG pipeline. It handles creation, retrieval,\n"
    "// and structured representation of user records.\n"
    "package users"
)


@pytest.fixture(scope="module")
def package_chunk():
    chunks = parse_file(str(DATA_FILE))
    matches = [c for c in chunks if c.type == ChunkType.PACKAGE]
    assert len(matches) == 1, f"Expected 1 PACKAGE chunk, got {len(matches)}"
    return matches[0]


def test_type(package_chunk):
    assert package_chunk.type == ChunkType.PACKAGE


def test_name(package_chunk):
    assert package_chunk.name == "users"


def test_package_name(package_chunk):
    assert package_chunk.package_name == "users"


def test_file_path(package_chunk):
    assert package_chunk.file_path == str(DATA_FILE)


def test_start_line(package_chunk):
    assert package_chunk.start_line == 1


def test_end_line(package_chunk):
    assert package_chunk.end_line == 4


def test_content(package_chunk):
    assert package_chunk.content == EXPECTED_CONTENT


def test_doc(package_chunk):
    assert package_chunk.doc == EXPECTED_DOC


def test_signature(package_chunk):
    assert package_chunk.signature == ""


def test_id(package_chunk):
    assert package_chunk.id == generate_id(EXPECTED_CONTENT)


def test_parent_id(package_chunk):
    assert package_chunk.parent_id is None


def test_children_ids(package_chunk):
    assert package_chunk.children_ids == []


def test_imported_symbols(package_chunk):
    assert package_chunk.imported_symbols == []


def test_metadata(package_chunk):
    assert package_chunk.metadata == {}
