from pathlib import Path

import pytest

from src.models.chunk import ChunkType, generate_id
from src.parsing.go_parser import parse_file

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "golang" / "struct_example.go"

EXPECTED_USER_DOC = (
    "// User represents a user account in the system.\n"
    "// It contains core identity and profile information."
)
EXPECTED_USER_SIGNATURE = "type User struct"
EXPECTED_USER_CONTENT = (
    "// User represents a user account in the system.\n"
    "// It contains core identity and profile information.\n"
    "type User struct {\n"
    "\tID    string\n"
    "\tEmail string\n"
    "\tAge   int\n"
    "}"
)

EXPECTED_PROFILE_DOC = ""
EXPECTED_PROFILE_SIGNATURE = "type Profile struct"
EXPECTED_PROFILE_CONTENT = (
    "type Profile struct {\n"
    "\tBio     string\n"
    "\tWebsite string\n"
    "}"
)


@pytest.fixture(scope="module")
def struct_chunks():
    chunks = parse_file(str(DATA_FILE))
    matches = [c for c in chunks if c.type == ChunkType.STRUCT]
    assert len(matches) == 2, f"Expected 2 STRUCT chunks, got {len(matches)}"
    return matches


@pytest.fixture(scope="module")
def user_chunk(struct_chunks):
    return struct_chunks[0]


@pytest.fixture(scope="module")
def profile_chunk(struct_chunks):
    return struct_chunks[1]


# --- User ---

def test_user_type(user_chunk):
    assert user_chunk.type == ChunkType.STRUCT


def test_user_name(user_chunk):
    assert user_chunk.name == "User"


def test_user_package_name(user_chunk):
    assert user_chunk.package_name == "users"


def test_user_file_path(user_chunk):
    assert user_chunk.file_path == str(DATA_FILE)


def test_user_start_line(user_chunk):
    assert user_chunk.start_line == 4


def test_user_end_line(user_chunk):
    assert user_chunk.end_line == 10


def test_user_content(user_chunk):
    assert user_chunk.content == EXPECTED_USER_CONTENT


def test_user_doc(user_chunk):
    assert user_chunk.doc == EXPECTED_USER_DOC


def test_user_signature(user_chunk):
    assert user_chunk.signature == EXPECTED_USER_SIGNATURE


def test_user_id(user_chunk):
    assert user_chunk.id == generate_id(EXPECTED_USER_CONTENT)


def test_user_parent_id(user_chunk):
    assert user_chunk.parent_id is None


def test_user_children_ids(user_chunk):
    assert user_chunk.children_ids == []


def test_user_imported_symbols(user_chunk):
    assert user_chunk.imported_symbols == []


def test_user_metadata(user_chunk):
    assert user_chunk.metadata == {}


# --- Profile (no doc comment) ---

def test_profile_type(profile_chunk):
    assert profile_chunk.type == ChunkType.STRUCT


def test_profile_name(profile_chunk):
    assert profile_chunk.name == "Profile"


def test_profile_package_name(profile_chunk):
    assert profile_chunk.package_name == "users"


def test_profile_file_path(profile_chunk):
    assert profile_chunk.file_path == str(DATA_FILE)


def test_profile_start_line(profile_chunk):
    assert profile_chunk.start_line == 12


def test_profile_end_line(profile_chunk):
    assert profile_chunk.end_line == 15


def test_profile_content(profile_chunk):
    assert profile_chunk.content == EXPECTED_PROFILE_CONTENT


def test_profile_doc(profile_chunk):
    assert profile_chunk.doc == EXPECTED_PROFILE_DOC


def test_profile_signature(profile_chunk):
    assert profile_chunk.signature == EXPECTED_PROFILE_SIGNATURE


def test_profile_id(profile_chunk):
    assert profile_chunk.id == generate_id(EXPECTED_PROFILE_CONTENT)


def test_profile_parent_id(profile_chunk):
    assert profile_chunk.parent_id is None


def test_profile_children_ids(profile_chunk):
    assert profile_chunk.children_ids == []


def test_profile_imported_symbols(profile_chunk):
    assert profile_chunk.imported_symbols == []


def test_profile_metadata(profile_chunk):
    assert profile_chunk.metadata == {}
