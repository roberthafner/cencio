from pathlib import Path

import pytest

from src.models.chunk import ChunkType, generate_id
from src.parsing.go_parser import parse_file

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "golang" / "method_example.go"

EXPECTED_VALIDATE_DOC = (
    "// Validate checks that the User fields are non-empty.\n"
    "// Returns an error if any required field is missing."
)
EXPECTED_VALIDATE_SIGNATURE = "func (u *User) Validate() error"
EXPECTED_VALIDATE_CONTENT = (
    "// Validate checks that the User fields are non-empty.\n"
    "// Returns an error if any required field is missing.\n"
    "func (u *User) Validate() error {\n"
    "\treturn nil\n"
    "}"
)

EXPECTED_DISPLAY_DOC = "// DisplayName returns a formatted display string for the user."
EXPECTED_DISPLAY_SIGNATURE = "func (u User) DisplayName() string"
EXPECTED_DISPLAY_CONTENT = (
    "// DisplayName returns a formatted display string for the user.\n"
    "func (u User) DisplayName() string {\n"
    "\treturn u.Email\n"
    "}"
)


@pytest.fixture(scope="module")
def all_chunks():
    return parse_file(str(DATA_FILE))


@pytest.fixture(scope="module")
def method_chunks(all_chunks):
    matches = [c for c in all_chunks if c.type == ChunkType.METHOD]
    assert len(matches) == 2, f"Expected 2 METHOD chunks, got {len(matches)}"
    return matches


@pytest.fixture(scope="module")
def user_struct(all_chunks):
    matches = [c for c in all_chunks if c.type == ChunkType.STRUCT]
    assert len(matches) == 1, f"Expected 1 STRUCT chunk, got {len(matches)}"
    return matches[0]


@pytest.fixture(scope="module")
def validate_chunk(method_chunks):
    return method_chunks[0]


@pytest.fixture(scope="module")
def display_name_chunk(method_chunks):
    return method_chunks[1]


# --- Validate ---

def test_validate_type(validate_chunk):
    assert validate_chunk.type == ChunkType.METHOD


def test_validate_name(validate_chunk):
    assert validate_chunk.name == "Validate"


def test_validate_package_name(validate_chunk):
    assert validate_chunk.package_name == "users"


def test_validate_file_path(validate_chunk):
    assert validate_chunk.file_path == str(DATA_FILE)


def test_validate_start_line(validate_chunk):
    assert validate_chunk.start_line == 10


def test_validate_end_line(validate_chunk):
    assert validate_chunk.end_line == 14


def test_validate_content(validate_chunk):
    assert validate_chunk.content == EXPECTED_VALIDATE_CONTENT


def test_validate_doc(validate_chunk):
    assert validate_chunk.doc == EXPECTED_VALIDATE_DOC


def test_validate_signature(validate_chunk):
    assert validate_chunk.signature == EXPECTED_VALIDATE_SIGNATURE


def test_validate_id(validate_chunk):
    assert validate_chunk.id == generate_id(EXPECTED_VALIDATE_CONTENT)


def test_validate_parent_id(validate_chunk, user_struct):
    assert validate_chunk.parent_id == user_struct.id


def test_validate_children_ids(validate_chunk):
    assert validate_chunk.children_ids == []


def test_validate_imported_symbols(validate_chunk):
    assert validate_chunk.imported_symbols == []


def test_validate_metadata(validate_chunk):
    assert validate_chunk.metadata == {}


# --- DisplayName ---

def test_display_name_type(display_name_chunk):
    assert display_name_chunk.type == ChunkType.METHOD


def test_display_name_name(display_name_chunk):
    assert display_name_chunk.name == "DisplayName"


def test_display_name_package_name(display_name_chunk):
    assert display_name_chunk.package_name == "users"


def test_display_name_file_path(display_name_chunk):
    assert display_name_chunk.file_path == str(DATA_FILE)


def test_display_name_start_line(display_name_chunk):
    assert display_name_chunk.start_line == 16


def test_display_name_end_line(display_name_chunk):
    assert display_name_chunk.end_line == 19


def test_display_name_content(display_name_chunk):
    assert display_name_chunk.content == EXPECTED_DISPLAY_CONTENT


def test_display_name_doc(display_name_chunk):
    assert display_name_chunk.doc == EXPECTED_DISPLAY_DOC


def test_display_name_signature(display_name_chunk):
    assert display_name_chunk.signature == EXPECTED_DISPLAY_SIGNATURE


def test_display_name_id(display_name_chunk):
    assert display_name_chunk.id == generate_id(EXPECTED_DISPLAY_CONTENT)


def test_display_name_parent_id(display_name_chunk, user_struct):
    assert display_name_chunk.parent_id == user_struct.id


def test_display_name_children_ids(display_name_chunk):
    assert display_name_chunk.children_ids == []


def test_display_name_imported_symbols(display_name_chunk):
    assert display_name_chunk.imported_symbols == []


def test_display_name_metadata(display_name_chunk):
    assert display_name_chunk.metadata == {}


# --- Struct wiring ---

def test_user_struct_children_ids(user_struct, validate_chunk, display_name_chunk):
    assert user_struct.children_ids == [validate_chunk.id, display_name_chunk.id]
