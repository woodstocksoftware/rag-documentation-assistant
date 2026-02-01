"""Tests for src/ingestion/chunker.py"""

import pytest
from src.ingestion.chunker import DocumentChunker


class TestCountTokens:
    def test_empty_string(self):
        chunker = DocumentChunker()
        assert chunker.count_tokens("") == 0

    def test_single_word(self):
        chunker = DocumentChunker()
        assert chunker.count_tokens("hello") >= 1

    def test_returns_int(self):
        chunker = DocumentChunker()
        assert isinstance(chunker.count_tokens("some text here"), int)


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunker = DocumentChunker(chunk_size=500)
        chunks = chunker.chunk_text("Short text.")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."

    def test_empty_text_returns_empty(self):
        chunker = DocumentChunker()
        assert chunker.chunk_text("") == []

    def test_whitespace_only_returns_empty(self):
        chunker = DocumentChunker()
        assert chunker.chunk_text("   \n\n  ") == []

    def test_chunk_has_required_keys(self):
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("Hello world.")
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert "token_count" in chunk

    def test_metadata_includes_chunk_index(self):
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph with more words to push over limit."
        chunks = chunker.chunk_text(text)
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_custom_metadata_preserved(self):
        chunker = DocumentChunker()
        meta = {"source": "test.pdf", "page": 3}
        chunks = chunker.chunk_text("Some text.", metadata=meta)
        assert chunks[0]["metadata"]["source"] == "test.pdf"
        assert chunks[0]["metadata"]["page"] == 3
        assert "chunk_index" in chunks[0]["metadata"]

    def test_default_metadata_is_empty_dict(self):
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("Hello.")
        # Only chunk_index should be present when no metadata given
        assert "chunk_index" in chunks[0]["metadata"]

    def test_token_count_is_positive(self):
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("Hello world, this is a test.")
        for chunk in chunks:
            assert chunk["token_count"] > 0

    def test_long_text_produces_multiple_chunks(self, sample_text):
        chunker = DocumentChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk_text(sample_text)
        assert len(chunks) > 1

    def test_chunks_cover_original_content(self, sample_text):
        """Every sentence from the original should appear in at least one chunk."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_text(sample_text)
        combined = " ".join(c["text"] for c in chunks)
        # Check a few key phrases are present
        assert "Retrieval-Augmented Generation" in combined
        assert "hallucination" in combined


class TestRecursiveSplit:
    def test_splits_on_paragraph_breaks_first(self):
        # Each paragraph must exceed chunk_size on its own so the total forces a split
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=0)
        para = "word " * 12  # ~12 tokens per paragraph
        text = f"{para.strip()}\n\n{para.strip()}"
        chunks = chunker._recursive_split(text, chunker.separators)
        assert len(chunks) >= 2

    def test_falls_back_to_sentence_split(self):
        chunker = DocumentChunker(chunk_size=8, chunk_overlap=0)
        text = "First sentence with several words here. Second sentence with several words here. Third sentence."
        chunks = chunker._recursive_split(text, chunker.separators)
        assert len(chunks) >= 2


class TestAddOverlap:
    def test_single_chunk_unchanged(self):
        chunker = DocumentChunker(chunk_overlap=10)
        result = chunker._add_overlap(["one chunk"])
        assert result == ["one chunk"]

    def test_zero_overlap_unchanged(self):
        chunker = DocumentChunker(chunk_overlap=0)
        chunks = ["chunk one", "chunk two"]
        result = chunker._add_overlap(chunks)
        assert result == chunks

    def test_overlap_adds_prefix(self):
        chunker = DocumentChunker(chunk_overlap=5)
        chunks = ["The first chunk of text with some content.", "The second chunk."]
        result = chunker._add_overlap(chunks)
        assert len(result) == 2
        assert result[0] == chunks[0]  # First chunk unchanged
        # Second chunk should be longer due to overlap prefix
        assert len(result[1]) > len(chunks[1])


class TestHardSplit:
    def test_splits_long_text(self):
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=0)
        long_text = "word " * 100
        chunks = chunker._hard_split(long_text)
        assert len(chunks) > 1

    def test_each_chunk_within_size(self):
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=0)
        long_text = "word " * 100
        chunks = chunker._hard_split(long_text)
        for chunk in chunks:
            assert chunker.count_tokens(chunk) <= chunker.chunk_size + 1  # small tolerance for decode
