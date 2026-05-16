from pathlib import Path

import pytest

from src.models.chunk import ChunkType, generate_id
from src.parsing.go_parser import parse_file

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "golang" / "type_alias_example.go"

EXPECTED_USERID_DOC = "// UserID is a named string type for user identifiers."
EXPECTED_USERID_SIGNATURE = "type UserID = string"
EXPECTED_USERID_CONTENT = (
    "// UserID is a named string type for user identifiers.\n"
    "type UserID = string"
)

EXPECTED_DURATION_DOC = "// Duration is a type definition based on int64."
EXPECTED_DURATION_SIGNATURE = "type Duration int64"
EXPECTED_DURATION_CONTENT = (
    "// Duration is a type definition based on int64.\n"
    "type Duration int64"
)

EXPECTED_RAWBYTES_DOC = ""
EXPECTED_RAWBYTES_SIGNATURE = "type RawBytes []byte"
EXPECTED_RAWBYTES_CONTENT = "type RawBytes []byte"


@pytest.fixture(scope="module")
def type_alias_chunks():
    chunks = parse_file(str(DATA_FILE))
    matches = [c for c in chunks if c.type == ChunkType.TYPE_ALIAS]
    assert len(matches) == 3, f"Expected 3 TYPE_ALIAS chunks, got {len(matches)}"
    return matches


@pytest.fixture(scope="module")
def userid_chunk(type_alias_chunks):
    return type_alias_chunks[0]


@pytest.fixture(scope="module")
def duration_chunk(type_alias_chunks):
    return type_alias_chunks[1]


@pytest.fixture(scope="module")
def rawbytes_chunk(type_alias_chunks):
    return type_alias_chunks[2]


# --- UserID (true alias with =) ---

def test_userid_type(userid_chunk):
    assert userid_chunk.type == ChunkType.TYPE_ALIAS


def test_userid_name(userid_chunk):
    assert userid_chunk.name == "UserID"


def test_userid_package_name(userid_chunk):
    assert userid_chunk.package_name == "users"


def test_userid_file_path(userid_chunk):
    assert userid_chunk.file_path == str(DATA_FILE)


def test_userid_start_line(userid_chunk):
    assert userid_chunk.start_line == 4


def test_userid_end_line(userid_chunk):
    assert userid_chunk.end_line == 5


def test_userid_content(userid_chunk):
    assert userid_chunk.content == EXPECTED_USERID_CONTENT


def test_userid_doc(userid_chunk):
    assert userid_chunk.doc == EXPECTED_USERID_DOC


def test_userid_signature(userid_chunk):
    assert userid_chunk.signature == EXPECTED_USERID_SIGNATURE


def test_userid_id(userid_chunk):
    assert userid_chunk.id == generate_id(EXPECTED_USERID_CONTENT)


def test_userid_parent_id(userid_chunk):
    assert userid_chunk.parent_id is None


def test_userid_children_ids(userid_chunk):
    assert userid_chunk.children_ids == []


def test_userid_imported_symbols(userid_chunk):
    assert userid_chunk.imported_symbols == []


def test_userid_metadata(userid_chunk):
    assert userid_chunk.metadata == {}


# --- Duration (type definition) ---

def test_duration_type(duration_chunk):
    assert duration_chunk.type == ChunkType.TYPE_ALIAS


def test_duration_name(duration_chunk):
    assert duration_chunk.name == "Duration"


def test_duration_package_name(duration_chunk):
    assert duration_chunk.package_name == "users"


def test_duration_file_path(duration_chunk):
    assert duration_chunk.file_path == str(DATA_FILE)


def test_duration_start_line(duration_chunk):
    assert duration_chunk.start_line == 7


def test_duration_end_line(duration_chunk):
    assert duration_chunk.end_line == 8


def test_duration_content(duration_chunk):
    assert duration_chunk.content == EXPECTED_DURATION_CONTENT


def test_duration_doc(duration_chunk):
    assert duration_chunk.doc == EXPECTED_DURATION_DOC


def test_duration_signature(duration_chunk):
    assert duration_chunk.signature == EXPECTED_DURATION_SIGNATURE


def test_duration_id(duration_chunk):
    assert duration_chunk.id == generate_id(EXPECTED_DURATION_CONTENT)


def test_duration_parent_id(duration_chunk):
    assert duration_chunk.parent_id is None


def test_duration_children_ids(duration_chunk):
    assert duration_chunk.children_ids == []


def test_duration_imported_symbols(duration_chunk):
    assert duration_chunk.imported_symbols == []


def test_duration_metadata(duration_chunk):
    assert duration_chunk.metadata == {}


# --- RawBytes (no doc comment) ---

def test_rawbytes_type(rawbytes_chunk):
    assert rawbytes_chunk.type == ChunkType.TYPE_ALIAS


def test_rawbytes_name(rawbytes_chunk):
    assert rawbytes_chunk.name == "RawBytes"


def test_rawbytes_package_name(rawbytes_chunk):
    assert rawbytes_chunk.package_name == "users"


def test_rawbytes_file_path(rawbytes_chunk):
    assert rawbytes_chunk.file_path == str(DATA_FILE)


def test_rawbytes_start_line(rawbytes_chunk):
    assert rawbytes_chunk.start_line == 10


def test_rawbytes_end_line(rawbytes_chunk):
    assert rawbytes_chunk.end_line == 10


def test_rawbytes_content(rawbytes_chunk):
    assert rawbytes_chunk.content == EXPECTED_RAWBYTES_CONTENT


def test_rawbytes_doc(rawbytes_chunk):
    assert rawbytes_chunk.doc == EXPECTED_RAWBYTES_DOC


def test_rawbytes_signature(rawbytes_chunk):
    assert rawbytes_chunk.signature == EXPECTED_RAWBYTES_SIGNATURE


def test_rawbytes_id(rawbytes_chunk):
    assert rawbytes_chunk.id == generate_id(EXPECTED_RAWBYTES_CONTENT)


def test_rawbytes_parent_id(rawbytes_chunk):
    assert rawbytes_chunk.parent_id is None


def test_rawbytes_children_ids(rawbytes_chunk):
    assert rawbytes_chunk.children_ids == []


def test_rawbytes_imported_symbols(rawbytes_chunk):
    assert rawbytes_chunk.imported_symbols == []


def test_rawbytes_metadata(rawbytes_chunk):
    assert rawbytes_chunk.metadata == {}
