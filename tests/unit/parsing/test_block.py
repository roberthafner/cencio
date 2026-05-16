from pathlib import Path

import pytest

from src.models.chunk import ChunkType, generate_id
from src.parsing.go_parser import parse_file

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "golang" / "block_example.go"

EXPECTED_CONST_DOC = "// StatusCodes contains HTTP status codes used by the API."
EXPECTED_CONST_CONTENT = (
    "// StatusCodes contains HTTP status codes used by the API.\n"
    "const (\n"
    "\tStatusOK       = 200\n"
    "\tStatusNotFound = 404\n"
    "\tStatusError    = 500\n"
    ")"
)

EXPECTED_VAR_DOC = "// DefaultLimits holds default resource limits for the pipeline."
EXPECTED_VAR_CONTENT = (
    "// DefaultLimits holds default resource limits for the pipeline.\n"
    "var (\n"
    "\tMaxConnections = 100\n"
    "\tMaxRetries     = 3\n"
    "\tTimeout        = 30\n"
    ")"
)


@pytest.fixture(scope="module")
def block_chunks():
    chunks = parse_file(str(DATA_FILE))
    matches = [c for c in chunks if c.type == ChunkType.BLOCK]
    assert len(matches) == 2, f"Expected 2 BLOCK chunks, got {len(matches)}"
    return matches


@pytest.fixture(scope="module")
def const_block(block_chunks):
    return block_chunks[0]


@pytest.fixture(scope="module")
def var_block(block_chunks):
    return block_chunks[1]


# --- const block ---

def test_const_type(const_block):
    assert const_block.type == ChunkType.BLOCK


def test_const_name(const_block):
    assert const_block.name == ""


def test_const_package_name(const_block):
    assert const_block.package_name == "example"


def test_const_file_path(const_block):
    assert const_block.file_path == str(DATA_FILE)


def test_const_start_line(const_block):
    assert const_block.start_line == 4


def test_const_end_line(const_block):
    assert const_block.end_line == 9


def test_const_content(const_block):
    assert const_block.content == EXPECTED_CONST_CONTENT


def test_const_doc(const_block):
    assert const_block.doc == EXPECTED_CONST_DOC


def test_const_signature(const_block):
    assert const_block.signature == ""


def test_const_id(const_block):
    assert const_block.id == generate_id(EXPECTED_CONST_CONTENT)


def test_const_parent_id(const_block):
    assert const_block.parent_id is None


def test_const_children_ids(const_block):
    assert const_block.children_ids == []


def test_const_imported_symbols(const_block):
    assert const_block.imported_symbols == []


def test_const_metadata(const_block):
    assert const_block.metadata == {}


# --- var block ---

def test_var_type(var_block):
    assert var_block.type == ChunkType.BLOCK


def test_var_name(var_block):
    assert var_block.name == ""


def test_var_package_name(var_block):
    assert var_block.package_name == "example"


def test_var_file_path(var_block):
    assert var_block.file_path == str(DATA_FILE)


def test_var_start_line(var_block):
    assert var_block.start_line == 11


def test_var_end_line(var_block):
    assert var_block.end_line == 16


def test_var_content(var_block):
    assert var_block.content == EXPECTED_VAR_CONTENT


def test_var_doc(var_block):
    assert var_block.doc == EXPECTED_VAR_DOC


def test_var_signature(var_block):
    assert var_block.signature == ""


def test_var_id(var_block):
    assert var_block.id == generate_id(EXPECTED_VAR_CONTENT)


def test_var_parent_id(var_block):
    assert var_block.parent_id is None


def test_var_children_ids(var_block):
    assert var_block.children_ids == []


def test_var_imported_symbols(var_block):
    assert var_block.imported_symbols == []


def test_var_metadata(var_block):
    assert var_block.metadata == {}
