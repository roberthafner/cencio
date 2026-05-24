import pytest

from src.ingestion.summarizer import (
    needs_summary,
    generate_summary,
    enrich_chunks_with_summaries,
)
from src.models.chunk import Chunk, ChunkType, generate_id


def _make_chunk(
    name: str = "TestFunc",
    content: str = "func TestFunc() {}",
    doc: str = "",
    low_quality: bool = False,
    chunk_type: ChunkType = ChunkType.FUNCTION,
) -> Chunk:
    return Chunk(
        id=generate_id(content),
        type=chunk_type,
        content=content,
        name=name,
        package_name="main",
        file_path="main.go",
        start_line=1,
        end_line=5,
        doc=doc,
        signature=f"func {name}()",
        is_test=False,
        low_quality=low_quality,
    )


# ------------------------------------------------------------------
# needs_summary tests
# ------------------------------------------------------------------

class TestNeedsSummary:
    """Tests for the needs_summary function."""

    def test_unnamed_chunk_needs_summary(self):
        """Chunks without a name should need a summary."""
        chunk = _make_chunk(name="", doc="// Some documentation here")
        assert needs_summary(chunk) is True

    def test_chunk_without_doc_needs_summary(self):
        """Chunks without documentation should need a summary."""
        chunk = _make_chunk(name="MyFunc", doc="")
        assert needs_summary(chunk) is True

    def test_chunk_with_short_doc_needs_summary(self):
        """Chunks with very short documentation (<30 chars) should need a summary."""
        chunk = _make_chunk(name="MyFunc", doc="// TODO")
        assert len(chunk.doc.strip()) < 30
        assert needs_summary(chunk) is True

    def test_chunk_with_whitespace_doc_needs_summary(self):
        """Chunks with only whitespace in doc should need a summary."""
        chunk = _make_chunk(name="MyFunc", doc="   \n\t   ")
        assert needs_summary(chunk) is True

    def test_chunk_with_good_doc_does_not_need_summary(self):
        """Chunks with meaningful documentation (>=30 chars) should not need a summary."""
        good_doc = "// MyFunc handles user authentication and session management"
        chunk = _make_chunk(name="MyFunc", doc=good_doc)
        assert len(good_doc.strip()) >= 30
        assert needs_summary(chunk) is False

    def test_chunk_with_exactly_30_char_doc_does_not_need_summary(self):
        """Chunks with exactly 30 character documentation should not need a summary."""
        doc = "// " + "x" * 27  # 30 chars total
        chunk = _make_chunk(name="MyFunc", doc=doc)
        assert len(doc.strip()) == 30
        assert needs_summary(chunk) is False

    def test_low_quality_chunk_does_not_need_summary(self):
        """Low-quality chunks should not be summarized."""
        chunk = _make_chunk(name="err", doc="", low_quality=True)
        assert needs_summary(chunk) is False

    def test_low_quality_chunk_with_no_doc_does_not_need_summary(self):
        """Low-quality chunks should be skipped even if they have no documentation."""
        chunk = _make_chunk(name="i", doc="", low_quality=True)
        assert needs_summary(chunk) is False

    def test_low_quality_unnamed_chunk_does_not_need_summary(self):
        """Low-quality unnamed chunks should be skipped."""
        chunk = _make_chunk(name="", doc="", low_quality=True)
        assert needs_summary(chunk) is False

    # Chunk type filtering tests

    def test_function_type_can_be_summarized(self):
        """Function chunks should be summarized."""
        chunk = _make_chunk(name="MyFunc", doc="", chunk_type=ChunkType.FUNCTION)
        assert needs_summary(chunk) is True

    def test_method_type_can_be_summarized(self):
        """Method chunks should be summarized."""
        chunk = _make_chunk(name="MyMethod", doc="", chunk_type=ChunkType.METHOD)
        assert needs_summary(chunk) is True

    def test_struct_type_can_be_summarized(self):
        """Struct chunks should be summarized."""
        chunk = _make_chunk(name="MyStruct", doc="", chunk_type=ChunkType.STRUCT)
        assert needs_summary(chunk) is True

    def test_interface_type_can_be_summarized(self):
        """Interface chunks should be summarized."""
        chunk = _make_chunk(name="MyInterface", doc="", chunk_type=ChunkType.INTERFACE)
        assert needs_summary(chunk) is True

    def test_const_type_not_summarized(self):
        """Const chunks should not be summarized."""
        chunk = _make_chunk(name="MaxRetries", doc="", chunk_type=ChunkType.CONST)
        assert needs_summary(chunk) is False

    def test_var_type_not_summarized(self):
        """Var chunks should not be summarized."""
        chunk = _make_chunk(name="defaultTimeout", doc="", chunk_type=ChunkType.VAR)
        assert needs_summary(chunk) is False

    def test_block_type_not_summarized(self):
        """Block chunks should not be summarized."""
        chunk = _make_chunk(name="", doc="", chunk_type=ChunkType.BLOCK)
        assert needs_summary(chunk) is False

    def test_package_type_not_summarized(self):
        """Package chunks should not be summarized."""
        chunk = _make_chunk(name="main", doc="", chunk_type=ChunkType.PACKAGE)
        assert needs_summary(chunk) is False

    def test_type_alias_not_summarized(self):
        """Type alias chunks should not be summarized."""
        chunk = _make_chunk(name="MyType", doc="", chunk_type=ChunkType.TYPE_ALIAS)
        assert needs_summary(chunk) is False


# ------------------------------------------------------------------
# generate_summary tests
# ------------------------------------------------------------------

class TestGenerateSummary:
    """Tests for the generate_summary function."""

    def test_generate_summary_calls_chat_function(self):
        """generate_summary should call the chat function with a formatted prompt."""
        chunk = _make_chunk(name="MyFunc", content="func MyFunc() { return 42 }")

        captured_prompt = None
        def mock_chat_fn(prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return "This function returns 42."

        result = generate_summary(chunk, mock_chat_fn)

        assert result == "This function returns 42."
        assert captured_prompt is not None
        assert "MyFunc" in captured_prompt
        assert "func MyFunc() { return 42 }" in captured_prompt
        assert "function" in captured_prompt  # chunk type
        assert "main" in captured_prompt  # package name

    def test_generate_summary_strips_whitespace(self):
        """generate_summary should strip whitespace from the result."""
        chunk = _make_chunk()

        def mock_chat_fn(prompt: str) -> str:
            return "  Summary with whitespace.  \n"

        result = generate_summary(chunk, mock_chat_fn)
        assert result == "Summary with whitespace."

    def test_generate_summary_handles_exception(self):
        """generate_summary should return empty string on exception."""
        chunk = _make_chunk()

        def failing_chat_fn(prompt: str) -> str:
            raise Exception("LLM error")

        result = generate_summary(chunk, failing_chat_fn)
        assert result == ""

    def test_generate_summary_truncates_long_content(self):
        """generate_summary should truncate content longer than 1500 chars."""
        long_content = "func LongFunc() {\n" + "x := 1\n" * 500 + "}"
        chunk = _make_chunk(content=long_content)

        captured_prompt = None
        def mock_chat_fn(prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Summary"

        generate_summary(chunk, mock_chat_fn)

        # The content in the prompt should be truncated
        assert len(long_content) > 1500
        assert long_content not in captured_prompt
        assert long_content[:1500] in captured_prompt


# ------------------------------------------------------------------
# enrich_chunks_with_summaries tests
# ------------------------------------------------------------------

class TestEnrichChunksWithSummaries:
    """Tests for the enrich_chunks_with_summaries function."""

    def test_enriches_chunk_that_needs_summary(self):
        """Chunks that need summaries should have summary field set."""
        chunk = _make_chunk(name="MyFunc", doc="")

        def mock_chat_fn(prompt: str) -> str:
            return "This is a generated summary."

        result = enrich_chunks_with_summaries([chunk], mock_chat_fn)

        assert len(result) == 1
        assert result[0].summary == "This is a generated summary."
        assert result[0].doc == ""  # Original doc unchanged

    def test_does_not_enrich_chunk_with_good_doc(self):
        """Chunks with good documentation should not be summarized."""
        good_doc = "// This function does something important and useful"
        chunk = _make_chunk(name="MyFunc", doc=good_doc)

        call_count = 0
        def mock_chat_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "Should not be called"

        result = enrich_chunks_with_summaries([chunk], mock_chat_fn)

        assert len(result) == 1
        assert result[0].summary == ""  # No summary added
        assert result[0].doc == good_doc  # Original doc preserved
        assert call_count == 0  # Chat function not called

    def test_preserves_original_doc(self):
        """Original doc field should be preserved, not modified."""
        original_doc = "// Short"
        chunk = _make_chunk(name="MyFunc", doc=original_doc)

        def mock_chat_fn(prompt: str) -> str:
            return "Generated summary."

        result = enrich_chunks_with_summaries([chunk], mock_chat_fn)

        assert result[0].doc == original_doc  # Original preserved
        assert result[0].summary == "Generated summary."

    def test_handles_empty_summary_from_llm(self):
        """If LLM returns empty string, summary should remain empty."""
        chunk = _make_chunk(name="MyFunc", doc="")

        def mock_chat_fn(prompt: str) -> str:
            return ""

        result = enrich_chunks_with_summaries([chunk], mock_chat_fn)

        assert result[0].summary == ""

    def test_enriches_multiple_chunks(self):
        """Should process multiple chunks correctly."""
        chunk1 = _make_chunk(name="Func1", doc="", content="func Func1() {}")
        chunk2 = _make_chunk(name="Func2", doc="// Good documentation for this function", content="func Func2() {}")
        chunk3 = _make_chunk(name="Func3", doc="", content="func Func3() {}")

        summaries = {
            "Func1": "Summary for Func1",
            "Func3": "Summary for Func3",
        }

        def mock_chat_fn(prompt: str) -> str:
            for name, summary in summaries.items():
                if name in prompt:
                    return summary
            return "Unknown"

        result = enrich_chunks_with_summaries([chunk1, chunk2, chunk3], mock_chat_fn)

        assert len(result) == 3
        assert result[0].summary == "Summary for Func1"
        assert result[1].summary == ""  # Good doc, not summarized
        assert result[2].summary == "Summary for Func3"

    def test_verbose_mode_prints_output(self, capsys):
        """Verbose mode should print summarization info."""
        chunk = _make_chunk(name="MyFunc", doc="")

        def mock_chat_fn(prompt: str) -> str:
            return "Summary"

        enrich_chunks_with_summaries([chunk], mock_chat_fn, verbose=True)

        captured = capsys.readouterr()
        assert "summarized" in captured.out
        assert "MyFunc" in captured.out
        assert "function" in captured.out

    def test_skips_low_quality_chunks(self):
        """Low-quality chunks should not be summarized."""
        regular_chunk = _make_chunk(name="MyFunc", doc="", low_quality=False)
        low_quality_chunk = _make_chunk(name="err", doc="", content="var err error", low_quality=True)

        call_count = 0
        def mock_chat_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "Summary"

        result = enrich_chunks_with_summaries([regular_chunk, low_quality_chunk], mock_chat_fn)

        assert len(result) == 2
        assert result[0].summary == "Summary"  # Regular chunk summarized
        assert result[1].summary == ""  # Low-quality chunk skipped
        assert call_count == 1  # Only called once for regular chunk
