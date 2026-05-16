from pathlib import Path

import pytest

from src.models.chunk import ChunkType, generate_id
from src.parsing.go_parser import parse_file

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "golang" / "var_example.go"

EXPECTED_ERRNOTFOUND_DOC = "// ErrNotFound is returned when a user cannot be located."
EXPECTED_ERRNOTFOUND_SIGNATURE = 'var ErrNotFound = errors.New("user not found")'
EXPECTED_ERRNOTFOUND_CONTENT = (
    "// ErrNotFound is returned when a user cannot be located.\n"
    'var ErrNotFound = errors.New("user not found")'
)

EXPECTED_PAGESIZE_DOC = "// defaultPageSize controls the number of results per page."
EXPECTED_PAGESIZE_SIGNATURE = "var defaultPageSize = 20"
EXPECTED_PAGESIZE_CONTENT = (
    "// defaultPageSize controls the number of results per page.\n"
    "var defaultPageSize = 20"
)

EXPECTED_DEBUG_DOC = ""
EXPECTED_DEBUG_SIGNATURE = "var debugMode = false"
EXPECTED_DEBUG_CONTENT = "var debugMode = false"


@pytest.fixture(scope="module")
def var_chunks():
    chunks = parse_file(str(DATA_FILE))
    matches = [c for c in chunks if c.type == ChunkType.VAR]
    assert len(matches) == 3, f"Expected 3 VAR chunks, got {len(matches)}"
    return matches


@pytest.fixture(scope="module")
def err_not_found_chunk(var_chunks):
    return var_chunks[0]


@pytest.fixture(scope="module")
def page_size_chunk(var_chunks):
    return var_chunks[1]


@pytest.fixture(scope="module")
def debug_mode_chunk(var_chunks):
    return var_chunks[2]


# --- ErrNotFound ---

def test_err_not_found_type(err_not_found_chunk):
    assert err_not_found_chunk.type == ChunkType.VAR


def test_err_not_found_name(err_not_found_chunk):
    assert err_not_found_chunk.name == "ErrNotFound"


def test_err_not_found_package_name(err_not_found_chunk):
    assert err_not_found_chunk.package_name == "users"


def test_err_not_found_file_path(err_not_found_chunk):
    assert err_not_found_chunk.file_path == str(DATA_FILE)


def test_err_not_found_start_line(err_not_found_chunk):
    assert err_not_found_chunk.start_line == 4


def test_err_not_found_end_line(err_not_found_chunk):
    assert err_not_found_chunk.end_line == 5


def test_err_not_found_content(err_not_found_chunk):
    assert err_not_found_chunk.content == EXPECTED_ERRNOTFOUND_CONTENT


def test_err_not_found_doc(err_not_found_chunk):
    assert err_not_found_chunk.doc == EXPECTED_ERRNOTFOUND_DOC


def test_err_not_found_signature(err_not_found_chunk):
    assert err_not_found_chunk.signature == EXPECTED_ERRNOTFOUND_SIGNATURE


def test_err_not_found_id(err_not_found_chunk):
    assert err_not_found_chunk.id == generate_id(EXPECTED_ERRNOTFOUND_CONTENT)


def test_err_not_found_parent_id(err_not_found_chunk):
    assert err_not_found_chunk.parent_id is None


def test_err_not_found_children_ids(err_not_found_chunk):
    assert err_not_found_chunk.children_ids == []


def test_err_not_found_imported_symbols(err_not_found_chunk):
    assert err_not_found_chunk.imported_symbols == []


def test_err_not_found_metadata(err_not_found_chunk):
    assert err_not_found_chunk.metadata == {}


# --- defaultPageSize ---

def test_page_size_type(page_size_chunk):
    assert page_size_chunk.type == ChunkType.VAR


def test_page_size_name(page_size_chunk):
    assert page_size_chunk.name == "defaultPageSize"


def test_page_size_package_name(page_size_chunk):
    assert page_size_chunk.package_name == "users"


def test_page_size_file_path(page_size_chunk):
    assert page_size_chunk.file_path == str(DATA_FILE)


def test_page_size_start_line(page_size_chunk):
    assert page_size_chunk.start_line == 7


def test_page_size_end_line(page_size_chunk):
    assert page_size_chunk.end_line == 8


def test_page_size_content(page_size_chunk):
    assert page_size_chunk.content == EXPECTED_PAGESIZE_CONTENT


def test_page_size_doc(page_size_chunk):
    assert page_size_chunk.doc == EXPECTED_PAGESIZE_DOC


def test_page_size_signature(page_size_chunk):
    assert page_size_chunk.signature == EXPECTED_PAGESIZE_SIGNATURE


def test_page_size_id(page_size_chunk):
    assert page_size_chunk.id == generate_id(EXPECTED_PAGESIZE_CONTENT)


def test_page_size_parent_id(page_size_chunk):
    assert page_size_chunk.parent_id is None


def test_page_size_children_ids(page_size_chunk):
    assert page_size_chunk.children_ids == []


def test_page_size_imported_symbols(page_size_chunk):
    assert page_size_chunk.imported_symbols == []


def test_page_size_metadata(page_size_chunk):
    assert page_size_chunk.metadata == {}


# --- debugMode (no doc comment) ---

def test_debug_mode_type(debug_mode_chunk):
    assert debug_mode_chunk.type == ChunkType.VAR


def test_debug_mode_name(debug_mode_chunk):
    assert debug_mode_chunk.name == "debugMode"


def test_debug_mode_package_name(debug_mode_chunk):
    assert debug_mode_chunk.package_name == "users"


def test_debug_mode_file_path(debug_mode_chunk):
    assert debug_mode_chunk.file_path == str(DATA_FILE)


def test_debug_mode_start_line(debug_mode_chunk):
    assert debug_mode_chunk.start_line == 10


def test_debug_mode_end_line(debug_mode_chunk):
    assert debug_mode_chunk.end_line == 10


def test_debug_mode_content(debug_mode_chunk):
    assert debug_mode_chunk.content == EXPECTED_DEBUG_CONTENT


def test_debug_mode_doc(debug_mode_chunk):
    assert debug_mode_chunk.doc == EXPECTED_DEBUG_DOC


def test_debug_mode_signature(debug_mode_chunk):
    assert debug_mode_chunk.signature == EXPECTED_DEBUG_SIGNATURE


def test_debug_mode_id(debug_mode_chunk):
    assert debug_mode_chunk.id == generate_id(EXPECTED_DEBUG_CONTENT)


def test_debug_mode_parent_id(debug_mode_chunk):
    assert debug_mode_chunk.parent_id is None


def test_debug_mode_children_ids(debug_mode_chunk):
    assert debug_mode_chunk.children_ids == []


def test_debug_mode_imported_symbols(debug_mode_chunk):
    assert debug_mode_chunk.imported_symbols == []


def test_debug_mode_metadata(debug_mode_chunk):
    assert debug_mode_chunk.metadata == {}
