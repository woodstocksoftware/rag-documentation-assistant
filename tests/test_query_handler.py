"""Tests for src/lambda/query/handler.py

These tests mock all AWS services (Bedrock, OpenSearch) and the Anthropic client
so they can run without credentials or network access.
"""

import json
import importlib
import pytest
from unittest.mock import patch, MagicMock


def _import_handler():
    """Import the query handler module (the 'lambda' dir name is a reserved word)."""
    return importlib.import_module("src.lambda.query.handler")


@pytest.fixture(autouse=True)
def patch_aws_and_anthropic(mock_anthropic_response, mock_bedrock_response):
    """Patch all external clients used at module level in the query handler."""
    mock_credentials = MagicMock()
    mock_credentials.access_key = "fake"
    mock_credentials.secret_key = "fake"
    mock_credentials.token = "fake"

    with patch.dict("os.environ", {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENSEARCH_ENDPOINT": "https://fake-endpoint.us-east-1.aoss.amazonaws.com",
    }):
        with patch("boto3.client") as mock_boto_client, \
             patch("boto3.Session") as mock_session:

            mock_session.return_value.get_credentials.return_value = mock_credentials

            # Bedrock mock
            bedrock_mock = MagicMock()
            bedrock_mock.invoke_model.return_value = mock_bedrock_response()
            mock_boto_client.return_value = bedrock_mock

            # We need to also patch Anthropic and AWS4Auth inside the handler module.
            # Since `lambda` is a keyword we can't do a normal dotted patch path,
            # so we import, then patch on the module object directly.
            handler = _import_handler()
            importlib.reload(handler)

            # Replace module-level globals with mocks
            orig_anthropic = handler.anthropic_client
            mock_anth = MagicMock()
            mock_anth.messages.create.return_value = mock_anthropic_response(text="Lambda answer")
            handler.anthropic_client = mock_anth

            orig_bedrock = handler.bedrock
            handler.bedrock = bedrock_mock

            yield handler

            # Restore
            handler.anthropic_client = orig_anthropic
            handler.bedrock = orig_bedrock


def _make_event(body=None):
    """Helper to build an API Gateway-style event."""
    return {"body": json.dumps(body) if body else "{}"}


class TestLambdaHandler:
    def test_valid_question(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        with patch.object(handler, "search_documents", return_value=[
            {"text": "chunk text", "metadata": {"source": "a.txt"}, "score": 0.9}
        ]):
            result = handler.lambda_handler(_make_event({"question": "What?"}), None)
            assert result["statusCode"] == 200
            body = json.loads(result["body"])
            assert "answer" in body

    def test_missing_question_returns_400(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        result = handler.lambda_handler(_make_event({"question": ""}), None)
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert "error" in body

    def test_no_body_returns_400(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        result = handler.lambda_handler(_make_event({}), None)
        assert result["statusCode"] == 400

    def test_no_chunks_found(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        with patch.object(handler, "search_documents", return_value=[]):
            result = handler.lambda_handler(_make_event({"question": "anything"}), None)
            assert result["statusCode"] == 200
            body = json.loads(result["body"])
            assert "no relevant" in body["answer"].lower()

    def test_cors_headers_on_success(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        with patch.object(handler, "search_documents", return_value=[
            {"text": "t", "metadata": {"source": "a.txt"}, "score": 0.9}
        ]):
            result = handler.lambda_handler(_make_event({"question": "Q?"}), None)
            assert result["headers"]["Access-Control-Allow-Origin"] == "*"

    def test_cors_headers_on_error(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        result = handler.lambda_handler(_make_event({"question": ""}), None)
        assert result["headers"]["Access-Control-Allow-Origin"] == "*"

    def test_exception_returns_500(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        with patch.object(handler, "generate_embedding", side_effect=RuntimeError("boom")):
            result = handler.lambda_handler(_make_event({"question": "Q?"}), None)
            assert result["statusCode"] == 500
            body = json.loads(result["body"])
            assert "boom" in body["error"]

    def test_whitespace_only_question_returns_400(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        result = handler.lambda_handler(_make_event({"question": "   "}), None)
        assert result["statusCode"] == 400


class TestGenerateEmbedding:
    def test_returns_list(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        result = handler.generate_embedding("hello")
        assert isinstance(result, list)


class TestSearchDocuments:
    def test_returns_list(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        with patch.object(handler, "get_opensearch_client") as mock_os:
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "hits": {"hits": [
                    {
                        "_source": {"text": "chunk", "metadata": {"source": "a.txt"}},
                        "_score": 0.9,
                    }
                ]}
            }
            mock_os.return_value = mock_client
            results = handler.search_documents([0.1] * 1536)
            assert len(results) == 1
            assert results[0]["text"] == "chunk"

    def test_search_error_returns_empty(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        with patch.object(handler, "get_opensearch_client") as mock_os:
            mock_os.return_value.search.side_effect = Exception("connection failed")
            results = handler.search_documents([0.1] * 1536)
            assert results == []


class TestGetOpensearchClient:
    def test_returns_client(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        with patch("src.lambda.query.handler.OpenSearch") as MockOS:
            client = handler.get_opensearch_client()
            MockOS.assert_called_once()
            call_kwargs = MockOS.call_args[1]
            assert call_kwargs["use_ssl"] is True


class TestGenerateResponse:
    def test_returns_answer_and_sources(self, patch_aws_and_anthropic):
        handler = patch_aws_and_anthropic
        chunks = [
            {"text": "chunk text", "metadata": {"source": "a.txt"}, "score": 0.9}
        ]
        result = handler.generate_response("question?", chunks)
        assert "answer" in result
        assert "sources" in result
        assert "usage" in result
