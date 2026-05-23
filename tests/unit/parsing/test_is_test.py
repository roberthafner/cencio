"""Tests for is_test detection in the Go parser."""
from pathlib import Path

import pytest

from src.parsing.go_parser import parse_file, _is_test_file
from tree_sitter import Language, Parser
import tree_sitter_go as tsg

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "golang"
TEST_FILE = DATA_DIR / "test_file_example_test.go"
TESTING_IMPORT_FILE = DATA_DIR / "testing_import_example.go"
REGULAR_FILE = DATA_DIR / "regular_example.go"

GO_LANGUAGE = Language(tsg.language())


# ------------------------------------------------------------------
# _is_test_file helper function
# ------------------------------------------------------------------

class TestIsTestFileHelper:
    """Tests for the _is_test_file helper function."""

    def _parse(self, source: bytes):
        parser = Parser(GO_LANGUAGE)
        return parser.parse(source)

    def test_file_ending_with_test_go_is_test(self):
        source = b'package main'
        tree = self._parse(source)
        assert _is_test_file("/path/to/foo_test.go", tree, source) is True

    def test_file_not_ending_with_test_go_is_not_test(self):
        source = b'package main'
        tree = self._parse(source)
        assert _is_test_file("/path/to/foo.go", tree, source) is False

    def test_file_with_single_testing_import_is_test(self):
        source = b'package main\n\nimport "testing"'
        tree = self._parse(source)
        assert _is_test_file("/path/to/helper.go", tree, source) is True

    def test_file_with_grouped_testing_import_is_test(self):
        source = b'package main\n\nimport (\n\t"fmt"\n\t"testing"\n)'
        tree = self._parse(source)
        assert _is_test_file("/path/to/helper.go", tree, source) is True

    def test_file_with_other_imports_is_not_test(self):
        source = b'package main\n\nimport (\n\t"fmt"\n\t"strings"\n)'
        tree = self._parse(source)
        assert _is_test_file("/path/to/main.go", tree, source) is False

    def test_file_with_no_imports_is_not_test(self):
        source = b'package main\n\nfunc main() {}'
        tree = self._parse(source)
        assert _is_test_file("/path/to/main.go", tree, source) is False

    def test_test_go_suffix_takes_precedence(self):
        """File ending with _test.go is test even without testing import."""
        source = b'package main\n\nimport "fmt"'
        tree = self._parse(source)
        assert _is_test_file("/path/to/utils_test.go", tree, source) is True


# ------------------------------------------------------------------
# parse_file is_test field
# ------------------------------------------------------------------

class TestParseFileIsTest:
    """Tests for is_test field being set correctly by parse_file."""

    def test_test_file_chunks_have_is_test_true(self):
        """All chunks from a _test.go file should have is_test=True."""
        chunks = parse_file(str(TEST_FILE))
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.is_test is True, f"Chunk {chunk.name} should have is_test=True"

    def test_testing_import_file_chunks_have_is_test_true(self):
        """All chunks from a file importing 'testing' should have is_test=True."""
        chunks = parse_file(str(TESTING_IMPORT_FILE))
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.is_test is True, f"Chunk {chunk.name} should have is_test=True"

    def test_regular_file_chunks_have_is_test_false(self):
        """All chunks from a regular file should have is_test=False."""
        chunks = parse_file(str(REGULAR_FILE))
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.is_test is False, f"Chunk {chunk.name} should have is_test=False"

    def test_test_file_package_chunk_has_is_test_true(self):
        """Package chunk from test file should have is_test=True."""
        chunks = parse_file(str(TEST_FILE))
        package_chunks = [c for c in chunks if c.type.value == "package"]
        assert len(package_chunks) == 1
        assert package_chunks[0].is_test is True

    def test_test_file_function_chunk_has_is_test_true(self):
        """Function chunk from test file should have is_test=True."""
        chunks = parse_file(str(TEST_FILE))
        function_chunks = [c for c in chunks if c.type.value == "function"]
        assert len(function_chunks) == 1
        assert function_chunks[0].is_test is True
        assert function_chunks[0].name == "TestHelper"

    def test_regular_file_function_chunk_has_is_test_false(self):
        """Function chunk from regular file should have is_test=False."""
        chunks = parse_file(str(REGULAR_FILE))
        function_chunks = [c for c in chunks if c.type.value == "function"]
        assert len(function_chunks) == 1
        assert function_chunks[0].is_test is False
        assert function_chunks[0].name == "RegularFunc"
