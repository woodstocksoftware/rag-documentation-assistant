"""Tests for src/query/rag.py"""

import pytest
from unittest.mock import patch, MagicMock
from src.query.rag import RAGPipeline


@pytest.fixture
def rag(tmp_path, mock_anthropic_response):
    """Create a RAGPipeline with mocked external dependencies."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("src.query.generator.Anthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create.return_value = mock_anthropic_response(
                text="The return policy allows refunds within 30 days.",
            )
            pipeline = RAGPipeline(
                collection_name="test_rag",
                persist_dir=str(tmp_path / "chroma"),
            )
            yield pipeline


@pytest.fixture
def rag_with_data(rag):
    """RAG pipeline with some documents already ingested."""
    from src.ingestion.chunker import DocumentChunker

    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_text(
        "Our return policy allows refunds within 30 days of purchase. "
        "Items must be unused and in original packaging.",
        metadata={"source": "policy.txt"},
    )
    rag.ingest(chunks)
    return rag


class TestQuery:
    def test_returns_required_keys(self, rag_with_data):
        result = rag_with_data.query("What is the return policy?")
        assert "answer" in result
        assert "sources" in result
        assert "usage" in result
        assert "retrieved_chunks" in result

    def test_no_chunks_returns_fallback(self, rag):
        """Query against an empty store should return a helpful message."""
        result = rag.query("anything")
        assert "couldn't find" in result["answer"].lower() or "no relevant" in result["answer"].lower() or result["retrieved_chunks"] == []

    def test_top_k_override(self, rag_with_data):
        result = rag_with_data.query("return policy", top_k=1)
        assert len(result["retrieved_chunks"]) <= 1

    def test_retrieved_chunks_included(self, rag_with_data):
        result = rag_with_data.query("return policy")
        assert len(result["retrieved_chunks"]) > 0
        assert "text" in result["retrieved_chunks"][0]


class TestIngest:
    def test_ingest_generates_embeddings(self, rag):
        chunks = [
            {"text": "Test chunk", "metadata": {"source": "t.txt", "chunk_index": 0}}
        ]
        rag.ingest(chunks)
        assert rag.vector_store.count() == 1

    def test_ingest_with_precomputed_embeddings(self, rag):
        dim = rag.embedding_model.dimension
        chunks = [
            {
                "text": "Test",
                "metadata": {"source": "t.txt", "chunk_index": 0},
                "embedding": [0.0] * dim,
            }
        ]
        rag.ingest(chunks)
        assert rag.vector_store.count() == 1

    def test_ingest_empty_list(self, rag):
        rag.ingest([])
        assert rag.vector_store.count() == 0
