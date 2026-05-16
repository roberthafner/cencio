from pathlib import Path

import pytest

from src.models.chunk import ChunkType, generate_id
from src.parsing.go_parser import parse_file

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "golang" / "const_example.go"

EXPECTED_MAXRETRIES_DOC = "// MaxRetries is the maximum number of retry attempts for a request."
EXPECTED_MAXRETRIES_SIGNATURE = "const MaxRetries = 3"
EXPECTED_MAXRETRIES_CONTENT = (
    "// MaxRetries is the maximum number of retry attempts for a request.\n"
    "const MaxRetries = 3"
)

EXPECTED_DEFAULTTIMEOUT_DOC = "// DefaultTimeout is the default request timeout in seconds."
EXPECTED_DEFAULTTIMEOUT_SIGNATURE = "const DefaultTimeout = 30"
EXPECTED_DEFAULTTIMEOUT_CONTENT = (
    "// DefaultTimeout is the default request timeout in seconds.\n"
    "const DefaultTimeout = 30"
)

EXPECTED_VERSION_DOC = ""
EXPECTED_VERSION_SIGNATURE = 'const version = "1.0.0"'
EXPECTED_VERSION_CONTENT = 'const version = "1.0.0"'


@pytest.fixture(scope="module")
def const_chunks():
    chunks = parse_file(str(DATA_FILE))
    matches = [c for c in chunks if c.type == ChunkType.CONST]
    assert len(matches) == 3, f"Expected 3 CONST chunks, got {len(matches)}"
    return matches


@pytest.fixture(scope="module")
def max_retries_chunk(const_chunks):
    return const_chunks[0]


@pytest.fixture(scope="module")
def default_timeout_chunk(const_chunks):
    return const_chunks[1]


@pytest.fixture(scope="module")
def version_chunk(const_chunks):
    return const_chunks[2]


# --- MaxRetries ---

def test_max_retries_type(max_retries_chunk):
    assert max_retries_chunk.type == ChunkType.CONST


def test_max_retries_name(max_retries_chunk):
    assert max_retries_chunk.name == "MaxRetries"


def test_max_retries_package_name(max_retries_chunk):
    assert max_retries_chunk.package_name == "users"


def test_max_retries_file_path(max_retries_chunk):
    assert max_retries_chunk.file_path == str(DATA_FILE)


def test_max_retries_start_line(max_retries_chunk):
    assert max_retries_chunk.start_line == 4


def test_max_retries_end_line(max_retries_chunk):
    assert max_retries_chunk.end_line == 5


def test_max_retries_content(max_retries_chunk):
    assert max_retries_chunk.content == EXPECTED_MAXRETRIES_CONTENT


def test_max_retries_doc(max_retries_chunk):
    assert max_retries_chunk.doc == EXPECTED_MAXRETRIES_DOC


def test_max_retries_signature(max_retries_chunk):
    assert max_retries_chunk.signature == EXPECTED_MAXRETRIES_SIGNATURE


def test_max_retries_id(max_retries_chunk):
    assert max_retries_chunk.id == generate_id(EXPECTED_MAXRETRIES_CONTENT)


def test_max_retries_parent_id(max_retries_chunk):
    assert max_retries_chunk.parent_id is None


def test_max_retries_children_ids(max_retries_chunk):
    assert max_retries_chunk.children_ids == []


def test_max_retries_imported_symbols(max_retries_chunk):
    assert max_retries_chunk.imported_symbols == []


def test_max_retries_metadata(max_retries_chunk):
    assert max_retries_chunk.metadata == {}


# --- DefaultTimeout ---

def test_default_timeout_type(default_timeout_chunk):
    assert default_timeout_chunk.type == ChunkType.CONST


def test_default_timeout_name(default_timeout_chunk):
    assert default_timeout_chunk.name == "DefaultTimeout"


def test_default_timeout_package_name(default_timeout_chunk):
    assert default_timeout_chunk.package_name == "users"


def test_default_timeout_file_path(default_timeout_chunk):
    assert default_timeout_chunk.file_path == str(DATA_FILE)


def test_default_timeout_start_line(default_timeout_chunk):
    assert default_timeout_chunk.start_line == 7


def test_default_timeout_end_line(default_timeout_chunk):
    assert default_timeout_chunk.end_line == 8


def test_default_timeout_content(default_timeout_chunk):
    assert default_timeout_chunk.content == EXPECTED_DEFAULTTIMEOUT_CONTENT


def test_default_timeout_doc(default_timeout_chunk):
    assert default_timeout_chunk.doc == EXPECTED_DEFAULTTIMEOUT_DOC


def test_default_timeout_signature(default_timeout_chunk):
    assert default_timeout_chunk.signature == EXPECTED_DEFAULTTIMEOUT_SIGNATURE


def test_default_timeout_id(default_timeout_chunk):
    assert default_timeout_chunk.id == generate_id(EXPECTED_DEFAULTTIMEOUT_CONTENT)


def test_default_timeout_parent_id(default_timeout_chunk):
    assert default_timeout_chunk.parent_id is None


def test_default_timeout_children_ids(default_timeout_chunk):
    assert default_timeout_chunk.children_ids == []


def test_default_timeout_imported_symbols(default_timeout_chunk):
    assert default_timeout_chunk.imported_symbols == []


def test_default_timeout_metadata(default_timeout_chunk):
    assert default_timeout_chunk.metadata == {}


# --- version (no doc comment) ---

def test_version_type(version_chunk):
    assert version_chunk.type == ChunkType.CONST


def test_version_name(version_chunk):
    assert version_chunk.name == "version"


def test_version_package_name(version_chunk):
    assert version_chunk.package_name == "users"


def test_version_file_path(version_chunk):
    assert version_chunk.file_path == str(DATA_FILE)


def test_version_start_line(version_chunk):
    assert version_chunk.start_line == 10


def test_version_end_line(version_chunk):
    assert version_chunk.end_line == 10


def test_version_content(version_chunk):
    assert version_chunk.content == EXPECTED_VERSION_CONTENT


def test_version_doc(version_chunk):
    assert version_chunk.doc == EXPECTED_VERSION_DOC


def test_version_signature(version_chunk):
    assert version_chunk.signature == EXPECTED_VERSION_SIGNATURE


def test_version_id(version_chunk):
    assert version_chunk.id == generate_id(EXPECTED_VERSION_CONTENT)


def test_version_parent_id(version_chunk):
    assert version_chunk.parent_id is None


def test_version_children_ids(version_chunk):
    assert version_chunk.children_ids == []


def test_version_imported_symbols(version_chunk):
    assert version_chunk.imported_symbols == []


def test_version_metadata(version_chunk):
    assert version_chunk.metadata == {}
