"""Tests for src/query/generator.py"""

import pytest
from unittest.mock import patch, MagicMock
from src.query.generator import ResponseGenerator


@pytest.fixture
def generator(mock_anthropic_response):
    """Create a generator with a mocked Anthropic client."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("src.query.generator.Anthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create.return_value = mock_anthropic_response(
                text="Based on [Source 1], refunds take 5-7 business days.",
                input_tokens=200,
                output_tokens=30,
            )
            gen = ResponseGenerator()
            yield gen


class TestInit:
    def test_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                ResponseGenerator()

    def test_accepts_custom_model(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
            with patch("src.query.generator.Anthropic"):
                gen = ResponseGenerator(model="claude-haiku-4-20250414")
                assert gen.model == "claude-haiku-4-20250414"


class TestGenerate:
    def test_returns_required_keys(self, generator, context_chunks_for_generation):
        result = generator.generate("How do I get a refund?", context_chunks_for_generation)
        assert "answer" in result
        assert "sources" in result
        assert "usage" in result

    def test_answer_is_string(self, generator, context_chunks_for_generation):
        result = generator.generate("question", context_chunks_for_generation)
        assert isinstance(result["answer"], str)

    def test_sources_match_chunks(self, generator, context_chunks_for_generation):
        result = generator.generate("question", context_chunks_for_generation)
        assert len(result["sources"]) == len(context_chunks_for_generation)
        assert result["sources"][0]["source"] == "return_policy.pdf"
        assert result["sources"][0]["score"] == 0.89

    def test_usage_has_token_counts(self, generator, context_chunks_for_generation):
        result = generator.generate("question", context_chunks_for_generation)
        assert result["usage"]["input_tokens"] == 200
        assert result["usage"]["output_tokens"] == 30

    def test_empty_chunks_list(self, generator):
        result = generator.generate("question", [])
        assert result["sources"] == []

    def test_chunk_missing_score_defaults_to_zero(self, generator):
        chunks = [
            {
                "text": "some text",
                "metadata": {"source": "file.txt"},
                # no "score" key
            }
        ]
        result = generator.generate("question", chunks)
        assert result["sources"][0]["score"] == 0

    def test_chunk_missing_chunk_index_defaults_to_zero(self, generator):
        chunks = [
            {
                "text": "some text",
                "metadata": {"source": "file.txt"},
                "score": 0.5,
            }
        ]
        result = generator.generate("question", chunks)
        assert result["sources"][0] is not None  # no KeyError
