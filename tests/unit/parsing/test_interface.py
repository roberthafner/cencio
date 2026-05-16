from pathlib import Path

import pytest

from src.models.chunk import ChunkType, generate_id
from src.parsing.go_parser import parse_file

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "golang" / "interface_example.go"

EXPECTED_REPO_DOC = (
    "// Repository defines the persistence operations for user records.\n"
    "// All implementations must be safe for concurrent use."
)
EXPECTED_REPO_SIGNATURE = "type Repository interface"
EXPECTED_REPO_CONTENT = (
    "// Repository defines the persistence operations for user records.\n"
    "// All implementations must be safe for concurrent use.\n"
    "type Repository interface {\n"
    "\tFindByID(ctx context.Context, id string) (*User, error)\n"
    "\tSave(ctx context.Context, u *User) error\n"
    "\tDelete(ctx context.Context, id string) error\n"
    "}"
)

EXPECTED_VALIDATOR_DOC = ""
EXPECTED_VALIDATOR_SIGNATURE = "type Validator interface"
EXPECTED_VALIDATOR_CONTENT = (
    "type Validator interface {\n"
    "\tValidate() error\n"
    "}"
)


@pytest.fixture(scope="module")
def interface_chunks():
    chunks = parse_file(str(DATA_FILE))
    matches = [c for c in chunks if c.type == ChunkType.INTERFACE]
    assert len(matches) == 2, f"Expected 2 INTERFACE chunks, got {len(matches)}"
    return matches


@pytest.fixture(scope="module")
def repository_chunk(interface_chunks):
    return interface_chunks[0]


@pytest.fixture(scope="module")
def validator_chunk(interface_chunks):
    return interface_chunks[1]


# --- Repository ---

def test_repo_type(repository_chunk):
    assert repository_chunk.type == ChunkType.INTERFACE


def test_repo_name(repository_chunk):
    assert repository_chunk.name == "Repository"


def test_repo_package_name(repository_chunk):
    assert repository_chunk.package_name == "users"


def test_repo_file_path(repository_chunk):
    assert repository_chunk.file_path == str(DATA_FILE)


def test_repo_start_line(repository_chunk):
    assert repository_chunk.start_line == 4


def test_repo_end_line(repository_chunk):
    assert repository_chunk.end_line == 10


def test_repo_content(repository_chunk):
    assert repository_chunk.content == EXPECTED_REPO_CONTENT


def test_repo_doc(repository_chunk):
    assert repository_chunk.doc == EXPECTED_REPO_DOC


def test_repo_signature(repository_chunk):
    assert repository_chunk.signature == EXPECTED_REPO_SIGNATURE


def test_repo_id(repository_chunk):
    assert repository_chunk.id == generate_id(EXPECTED_REPO_CONTENT)


def test_repo_parent_id(repository_chunk):
    assert repository_chunk.parent_id is None


def test_repo_children_ids(repository_chunk):
    assert repository_chunk.children_ids == []


def test_repo_imported_symbols(repository_chunk):
    assert repository_chunk.imported_symbols == []


def test_repo_metadata(repository_chunk):
    assert repository_chunk.metadata == {}


# --- Validator (no doc comment) ---

def test_validator_type(validator_chunk):
    assert validator_chunk.type == ChunkType.INTERFACE


def test_validator_name(validator_chunk):
    assert validator_chunk.name == "Validator"


def test_validator_package_name(validator_chunk):
    assert validator_chunk.package_name == "users"


def test_validator_file_path(validator_chunk):
    assert validator_chunk.file_path == str(DATA_FILE)


def test_validator_start_line(validator_chunk):
    assert validator_chunk.start_line == 12


def test_validator_end_line(validator_chunk):
    assert validator_chunk.end_line == 14


def test_validator_content(validator_chunk):
    assert validator_chunk.content == EXPECTED_VALIDATOR_CONTENT


def test_validator_doc(validator_chunk):
    assert validator_chunk.doc == EXPECTED_VALIDATOR_DOC


def test_validator_signature(validator_chunk):
    assert validator_chunk.signature == EXPECTED_VALIDATOR_SIGNATURE


def test_validator_id(validator_chunk):
    assert validator_chunk.id == generate_id(EXPECTED_VALIDATOR_CONTENT)


def test_validator_parent_id(validator_chunk):
    assert validator_chunk.parent_id is None


def test_validator_children_ids(validator_chunk):
    assert validator_chunk.children_ids == []


def test_validator_imported_symbols(validator_chunk):
    assert validator_chunk.imported_symbols == []


def test_validator_metadata(validator_chunk):
    assert validator_chunk.metadata == {}
