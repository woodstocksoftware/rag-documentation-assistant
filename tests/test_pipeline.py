"""Tests for src/ingestion/pipeline.py"""

import pytest
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def pipeline():
    return IngestionPipeline(chunk_size=200, chunk_overlap=20)


@pytest.fixture
def docs_dir(tmp_path):
    """Create a temp directory with two small docs."""
    (tmp_path / "alpha.txt").write_text(
        "Alpha document. " * 20, encoding="utf-8"
    )
    (tmp_path / "beta.md").write_text(
        "# Beta\n\n" + "Beta paragraph. " * 20, encoding="utf-8"
    )
    return tmp_path


class TestInit:
    def test_default_params(self):
        p = IngestionPipeline()
        assert p.chunker.chunk_size == 500
        assert p.chunker.chunk_overlap == 50

    def test_custom_params(self):
        p = IngestionPipeline(chunk_size=100, chunk_overlap=10)
        assert p.chunker.chunk_size == 100
        assert p.chunker.chunk_overlap == 10


class TestProcessFile:
    def test_returns_chunks(self, pipeline, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello world. " * 50, encoding="utf-8")
        chunks = pipeline.process_file(f)
        assert len(chunks) >= 1
        assert "text" in chunks[0]
        assert "metadata" in chunks[0]

    def test_metadata_contains_source(self, pipeline, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nBody text here.", encoding="utf-8")
        chunks = pipeline.process_file(f)
        assert chunks[0]["metadata"]["source"] == "readme.md"

    def test_accepts_string_path(self, pipeline, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Some content.", encoding="utf-8")
        chunks = pipeline.process_file(str(f))
        assert len(chunks) >= 1


class TestProcessDirectory:
    def test_processes_all_files(self, pipeline, docs_dir):
        chunks = pipeline.process_directory(docs_dir)
        sources = {c["metadata"]["source"] for c in chunks}
        assert "alpha.txt" in sources
        assert "beta.md" in sources

    def test_returns_combined_chunks(self, pipeline, docs_dir):
        chunks = pipeline.process_directory(docs_dir)
        assert len(chunks) >= 2

    def test_empty_directory(self, pipeline, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        chunks = pipeline.process_directory(empty)
        assert chunks == []

    def test_chunk_token_counts_present(self, pipeline, docs_dir):
        chunks = pipeline.process_directory(docs_dir)
        for chunk in chunks:
            assert "token_count" in chunk
            assert chunk["token_count"] > 0
