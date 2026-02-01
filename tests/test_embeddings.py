"""Tests for src/ingestion/embeddings.py"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestEmbeddingModel:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        """Load model once per class to avoid repeated downloads."""
        from src.ingestion.embeddings import EmbeddingModel
        self.model = EmbeddingModel()

    def test_dimension_attribute(self):
        assert isinstance(self.model.dimension, int)
        assert self.model.dimension > 0

    def test_embed_text_returns_list_of_floats(self):
        embedding = self.model.embed_text("hello world")
        assert isinstance(embedding, list)
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_text_consistent_dimension(self):
        e1 = self.model.embed_text("first")
        e2 = self.model.embed_text("second")
        assert len(e1) == len(e2) == self.model.dimension

    def test_embed_text_empty_string(self):
        embedding = self.model.embed_text("")
        assert len(embedding) == self.model.dimension

    def test_similar_texts_have_higher_similarity(self):
        e1 = np.array(self.model.embed_text("How do I return a product?"))
        e2 = np.array(self.model.embed_text("What is the return policy?"))
        e3 = np.array(self.model.embed_text("What is the weather today?"))

        sim_related = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        sim_unrelated = np.dot(e1, e3) / (np.linalg.norm(e1) * np.linalg.norm(e3))
        assert sim_related > sim_unrelated

    def test_embed_chunks_adds_embedding_key(self):
        chunks = [
            {"text": "chunk one", "metadata": {}},
            {"text": "chunk two", "metadata": {}},
        ]
        result = self.model.embed_chunks(chunks)
        for chunk in result:
            assert "embedding" in chunk
            assert len(chunk["embedding"]) == self.model.dimension

    def test_embed_chunks_preserves_existing_keys(self):
        chunks = [{"text": "hello", "metadata": {"source": "a.txt"}, "token_count": 1}]
        result = self.model.embed_chunks(chunks)
        assert result[0]["metadata"]["source"] == "a.txt"
        assert result[0]["token_count"] == 1

    def test_embed_chunks_empty_list(self):
        result = self.model.embed_chunks([])
        assert result == []
