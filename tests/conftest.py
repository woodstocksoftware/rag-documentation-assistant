"""
Shared fixtures for the RAG documentation assistant test suite.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Document fixtures ──────────────────────────────────────────────


@pytest.fixture
def sample_text():
    """A multi-paragraph sample text for chunking tests."""
    return (
        "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses "
        "by providing relevant context from a knowledge base.\n\n"
        "The process works in three steps. First, the user's question is converted into "
        "an embedding vector. Second, this vector is used to search a database of document "
        "embeddings to find the most similar chunks. Third, these chunks are added to the "
        "prompt as context for the LLM to reference when generating its answer.\n\n"
        "RAG solves the hallucination problem by grounding responses in actual documents. "
        "It also allows LLMs to access private or recent information not in their training data."
    )


@pytest.fixture
def sample_chunks():
    """Pre-built chunks with metadata, as produced by the chunker."""
    return [
        {
            "text": "Our return policy allows refunds within 30 days of purchase.",
            "metadata": {"source": "policy.txt", "chunk_index": 0},
            "token_count": 12,
        },
        {
            "text": "To contact customer support, email support@example.com or call 1-800-555-0123.",
            "metadata": {"source": "support.txt", "chunk_index": 0},
            "token_count": 16,
        },
        {
            "text": "Shipping takes 3-5 business days for standard delivery.",
            "metadata": {"source": "shipping.txt", "chunk_index": 0},
            "token_count": 11,
        },
    ]


@pytest.fixture
def sample_chunks_with_embeddings(sample_chunks):
    """Chunks that already have mock embedding vectors attached."""
    for i, chunk in enumerate(sample_chunks):
        chunk["embedding"] = [float(i)] * 384
    return sample_chunks


@pytest.fixture
def context_chunks_for_generation():
    """Chunks formatted for the response generator (with score)."""
    return [
        {
            "text": "Refunds are processed within 5-7 business days.",
            "metadata": {"source": "return_policy.pdf", "chunk_index": 0},
            "score": 0.89,
        },
        {
            "text": "To initiate a return, log into your account and select 'Order History'.",
            "metadata": {"source": "return_policy.pdf", "chunk_index": 1},
            "score": 0.82,
        },
    ]


# ── Temporary file fixtures ────────────────────────────────────────


@pytest.fixture
def tmp_text_file(tmp_path):
    """Create a temporary .txt file."""
    f = tmp_path / "test.txt"
    f.write_text("Hello world.\nThis is a test document.", encoding="utf-8")
    return f


@pytest.fixture
def tmp_markdown_file(tmp_path):
    """Create a temporary .md file."""
    f = tmp_path / "test.md"
    f.write_text("# Heading\n\nSome markdown content.", encoding="utf-8")
    return f


@pytest.fixture
def tmp_doc_directory(tmp_path):
    """Create a temp directory with several document files."""
    (tmp_path / "a.txt").write_text("Document A content.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Doc B\n\nMarkdown body.", encoding="utf-8")
    (tmp_path / "skip.jpg").write_bytes(b"\xff\xd8")  # unsupported
    return tmp_path


# ── Mock helpers for external services ─────────────────────────────


@pytest.fixture
def mock_anthropic_response():
    """Build a mock Anthropic Messages.create response."""
    def _build(text="Mock answer", input_tokens=100, output_tokens=50):
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = text
        response.content = [content_block]
        response.usage.input_tokens = input_tokens
        response.usage.output_tokens = output_tokens
        return response
    return _build


@pytest.fixture
def mock_bedrock_response():
    """Build a mock Bedrock invoke_model response."""
    import json
    import io

    def _build(embedding=None):
        if embedding is None:
            embedding = [0.1] * 1536
        body = io.BytesIO(json.dumps({"embedding": embedding}).encode())
        return {"body": body}
    return _build
