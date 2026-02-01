"""Tests for src/shared/vector_store.py"""

import pytest
from src.shared.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore backed by a temp directory."""
    return VectorStore(collection_name="test", persist_dir=str(tmp_path / "chroma"))


@pytest.fixture
def populated_store(store):
    """A store with 3 chunks already added."""
    chunks = [
        {
            "text": "Our return policy allows refunds within 30 days.",
            "metadata": {"source": "policy.txt", "chunk_index": 0},
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "text": "Contact support at support@example.com.",
            "metadata": {"source": "support.txt", "chunk_index": 0},
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "text": "Shipping takes 3-5 business days.",
            "metadata": {"source": "shipping.txt", "chunk_index": 0},
            "embedding": [0.0, 0.0, 1.0],
        },
    ]
    store.add_chunks(chunks)
    return store


class TestAddChunks:
    def test_add_increases_count(self, store):
        assert store.count() == 0
        chunks = [
            {
                "text": "hello",
                "metadata": {"source": "a.txt", "chunk_index": 0},
                "embedding": [1.0, 0.0],
            }
        ]
        store.add_chunks(chunks)
        assert store.count() == 1

    def test_add_empty_list_is_noop(self, store):
        store.add_chunks([])
        assert store.count() == 0

    def test_upsert_does_not_duplicate(self, store):
        chunk = {
            "text": "hello",
            "metadata": {"source": "a.txt", "chunk_index": 0},
            "embedding": [1.0, 0.0],
        }
        store.add_chunks([chunk])
        store.add_chunks([chunk])  # same id → upsert
        assert store.count() == 1

    def test_upsert_updates_text(self, store):
        chunk_v1 = {
            "text": "version 1",
            "metadata": {"source": "a.txt", "chunk_index": 0},
            "embedding": [1.0, 0.0, 0.0],
        }
        chunk_v2 = {
            "text": "version 2",
            "metadata": {"source": "a.txt", "chunk_index": 0},
            "embedding": [1.0, 0.0, 0.0],
        }
        store.add_chunks([chunk_v1])
        store.add_chunks([chunk_v2])
        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert results[0]["text"] == "version 2"


class TestSearch:
    def test_returns_results(self, populated_store):
        results = populated_store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_result_has_required_keys(self, populated_store):
        results = populated_store.search([1.0, 0.0, 0.0], top_k=1)
        r = results[0]
        assert "text" in r
        assert "metadata" in r
        assert "score" in r

    def test_most_similar_ranked_first(self, populated_store):
        results = populated_store.search([1.0, 0.0, 0.0], top_k=3)
        # The chunk with embedding [1,0,0] should rank highest
        assert results[0]["metadata"]["source"] == "policy.txt"

    def test_score_is_similarity_not_distance(self, populated_store):
        results = populated_store.search([1.0, 0.0, 0.0], top_k=1)
        # Cosine similarity for identical direction should be close to 1
        assert results[0]["score"] > 0.5

    def test_top_k_limits_results(self, populated_store):
        results = populated_store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1

    def test_search_empty_store(self, store):
        # Searching an empty collection should return empty (or raise gracefully)
        # ChromaDB may raise; we just ensure no unhandled crash
        results = store.search([1.0, 0.0], top_k=3)
        assert isinstance(results, list)


class TestClear:
    def test_clear_removes_all(self, populated_store):
        assert populated_store.count() > 0
        populated_store.clear()
        assert populated_store.count() == 0

    def test_add_after_clear(self, populated_store):
        populated_store.clear()
        chunk = {
            "text": "new",
            "metadata": {"source": "new.txt", "chunk_index": 0},
            "embedding": [1.0, 0.0, 0.0],
        }
        populated_store.add_chunks([chunk])
        assert populated_store.count() == 1


class TestCount:
    def test_count_zero_initially(self, store):
        assert store.count() == 0

    def test_count_after_adds(self, populated_store):
        assert populated_store.count() == 3
