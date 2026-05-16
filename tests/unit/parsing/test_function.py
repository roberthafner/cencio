from pathlib import Path

import pytest

from src.models.chunk import ChunkType, generate_id
from src.parsing.go_parser import parse_file

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "golang" / "function_example.go"

EXPECTED_FIND_DOC = (
    "// FindByID retrieves a user by their unique identifier.\n"
    "// Returns an error if the user is not found or the query fails."
)
EXPECTED_FIND_SIGNATURE = "func FindByID(ctx context.Context, id string) (*User, error)"
EXPECTED_FIND_CONTENT = (
    "// FindByID retrieves a user by their unique identifier.\n"
    "// Returns an error if the user is not found or the query fails.\n"
    "func FindByID(ctx context.Context, id string) (*User, error) {\n"
    "\treturn nil, nil\n"
    "}"
)

EXPECTED_SAVE_DOC = "// save persists a user record to the database."
EXPECTED_SAVE_SIGNATURE = "func save(u *User) error"
EXPECTED_SAVE_CONTENT = (
    "// save persists a user record to the database.\n"
    "func save(u *User) error {\n"
    "\treturn nil\n"
    "}"
)


@pytest.fixture(scope="module")
def function_chunks():
    chunks = parse_file(str(DATA_FILE))
    matches = [c for c in chunks if c.type == ChunkType.FUNCTION]
    assert len(matches) == 2, f"Expected 2 FUNCTION chunks, got {len(matches)}"
    return matches


@pytest.fixture(scope="module")
def find_by_id(function_chunks):
    return function_chunks[0]


@pytest.fixture(scope="module")
def save(function_chunks):
    return function_chunks[1]


# --- FindByID ---

def test_find_type(find_by_id):
    assert find_by_id.type == ChunkType.FUNCTION


def test_find_name(find_by_id):
    assert find_by_id.name == "FindByID"


def test_find_package_name(find_by_id):
    assert find_by_id.package_name == "users"


def test_find_file_path(find_by_id):
    assert find_by_id.file_path == str(DATA_FILE)


def test_find_start_line(find_by_id):
    assert find_by_id.start_line == 4


def test_find_end_line(find_by_id):
    assert find_by_id.end_line == 8


def test_find_content(find_by_id):
    assert find_by_id.content == EXPECTED_FIND_CONTENT


def test_find_doc(find_by_id):
    assert find_by_id.doc == EXPECTED_FIND_DOC


def test_find_signature(find_by_id):
    assert find_by_id.signature == EXPECTED_FIND_SIGNATURE


def test_find_id(find_by_id):
    assert find_by_id.id == generate_id(EXPECTED_FIND_CONTENT)


def test_find_parent_id(find_by_id):
    assert find_by_id.parent_id is None


def test_find_children_ids(find_by_id):
    assert find_by_id.children_ids == []


def test_find_imported_symbols(find_by_id):
    assert find_by_id.imported_symbols == []


def test_find_metadata(find_by_id):
    assert find_by_id.metadata == {}


# --- save ---

def test_save_type(save):
    assert save.type == ChunkType.FUNCTION


def test_save_name(save):
    assert save.name == "save"


def test_save_package_name(save):
    assert save.package_name == "users"


def test_save_file_path(save):
    assert save.file_path == str(DATA_FILE)


def test_save_start_line(save):
    assert save.start_line == 10


def test_save_end_line(save):
    assert save.end_line == 13


def test_save_content(save):
    assert save.content == EXPECTED_SAVE_CONTENT


def test_save_doc(save):
    assert save.doc == EXPECTED_SAVE_DOC


def test_save_signature(save):
    assert save.signature == EXPECTED_SAVE_SIGNATURE


def test_save_id(save):
    assert save.id == generate_id(EXPECTED_SAVE_CONTENT)


def test_save_parent_id(save):
    assert save.parent_id is None


def test_save_children_ids(save):
    assert save.children_ids == []


def test_save_imported_symbols(save):
    assert save.imported_symbols == []


def test_save_metadata(save):
    assert save.metadata == {}
