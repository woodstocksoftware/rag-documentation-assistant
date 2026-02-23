"""Tests for src/lambda/ingest/handler.py

Mocks all AWS services (S3, Bedrock, OpenSearch) so tests run offline.
"""

import json
import importlib
import pytest
from unittest.mock import patch, MagicMock


def _import_handler():
    """Import the ingest handler module (the 'lambda' dir name is a reserved word)."""
    return importlib.import_module("src.lambda.ingest.handler")


@pytest.fixture(autouse=True)
def patch_aws(mock_bedrock_response):
    """Patch AWS clients at module level before importing the handler."""
    mock_credentials = MagicMock()
    mock_credentials.access_key = "fake"
    mock_credentials.secret_key = "fake"
    mock_credentials.token = "fake"

    with patch.dict("os.environ", {
        "OPENSEARCH_ENDPOINT": "https://fake-endpoint.us-east-1.aoss.amazonaws.com",
    }):
        with patch("boto3.client") as mock_boto_client, \
             patch("boto3.Session") as mock_session:

            mock_session.return_value.get_credentials.return_value = mock_credentials

            # S3 mock
            s3_mock = MagicMock()
            s3_mock.get_object.return_value = {
                "Body": MagicMock(read=lambda: b"Sample document content for testing.")
            }

            # Bedrock mock
            bedrock_mock = MagicMock()
            bedrock_mock.invoke_model.return_value = mock_bedrock_response()

            def client_factory(service, **kwargs):
                if service == "s3":
                    return s3_mock
                return bedrock_mock

            mock_boto_client.side_effect = client_factory

            handler = _import_handler()
            importlib.reload(handler)

            # Replace module-level globals
            orig_s3 = handler.s3
            orig_bedrock = handler.bedrock
            handler.s3 = s3_mock
            handler.bedrock = bedrock_mock

            yield handler

            handler.s3 = orig_s3
            handler.bedrock = orig_bedrock


def _make_sqs_event(bucket="my-bucket", key="docs/test.txt"):
    """Build an SQS event wrapping an S3 notification."""
    s3_event = {
        "Records": [{
            "s3": {
                "bucket": {"name": bucket},
                "object": {"key": key},
            }
        }]
    }
    return {
        "Records": [{
            "body": json.dumps(s3_event)
        }]
    }


class TestExtractText:
    def test_txt(self, patch_aws):
        handler = patch_aws
        text = handler.extract_text(b"hello world", ".txt")
        assert text == "hello world"

    def test_md(self, patch_aws):
        handler = patch_aws
        text = handler.extract_text(b"# heading\ncontent", ".md")
        assert "heading" in text

    def test_unsupported_raises(self, patch_aws):
        handler = patch_aws
        with pytest.raises(ValueError, match="Unsupported"):
            handler.extract_text(b"data", ".exe")


class TestChunkText:
    def test_short_text_single_chunk(self, patch_aws):
        handler = patch_aws
        chunks = handler.chunk_text("short text", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_long_text_multiple_chunks(self, patch_aws):
        handler = patch_aws
        text = "word " * 200
        chunks = handler.chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_empty_text(self, patch_aws):
        handler = patch_aws
        chunks = handler.chunk_text("")
        assert chunks == []

    def test_respects_sentence_boundaries(self, patch_aws):
        handler = patch_aws
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = handler.chunk_text(text, chunk_size=40, overlap=5)
        for chunk in chunks:
            if chunk != chunks[-1]:
                assert chunk.endswith(".") or chunk.endswith(". ") or len(chunk) <= 40


class TestLambdaHandler:
    def test_processes_txt_file(self, patch_aws):
        handler = patch_aws
        with patch.object(handler, "process_document", return_value=1):
            result = handler.lambda_handler(_make_sqs_event(), None)
            assert result["processed"] >= 1
            assert result["errors"] == []

    def test_skips_unsupported_files(self, patch_aws):
        handler = patch_aws
        event = _make_sqs_event(key="images/photo.jpg")
        result = handler.lambda_handler(event, None)
        assert result["processed"] == 0
        assert result["errors"] == []

    def test_handles_processing_error(self, patch_aws):
        handler = patch_aws
        with patch.object(handler, "process_document", side_effect=RuntimeError("s3 down")):
            result = handler.lambda_handler(_make_sqs_event(), None)
            assert len(result["errors"]) > 0

    def test_empty_event(self, patch_aws):
        handler = patch_aws
        result = handler.lambda_handler({"Records": []}, None)
        assert result["processed"] == 0
        assert result["errors"] == []

    def test_multiple_records(self, patch_aws):
        handler = patch_aws
        event = {
            "Records": [
                {"body": json.dumps({"Records": [
                    {"s3": {"bucket": {"name": "b"}, "object": {"key": "a.txt"}}}
                ]})},
                {"body": json.dumps({"Records": [
                    {"s3": {"bucket": {"name": "b"}, "object": {"key": "b.md"}}}
                ]})},
            ]
        }
        with patch.object(handler, "process_document", return_value=1):
            result = handler.lambda_handler(event, None)
            assert result["processed"] == 2


class TestExtractTextPdfDocx:
    def test_pdf_extraction_blank(self, patch_aws):
        handler = patch_aws
        from pypdf import PdfWriter
        from io import BytesIO

        buf = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        text = handler.extract_text(pdf_bytes, ".pdf")
        assert isinstance(text, str)

    def test_pdf_extraction_with_text(self, patch_aws):
        handler = patch_aws
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello from PDF"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            text = handler.extract_text(b"fake", ".pdf")
            assert "Hello from PDF" in text

    def test_docx_extraction(self, patch_aws):
        handler = patch_aws
        from docx import Document
        from io import BytesIO

        doc = Document()
        doc.add_paragraph("Hello from docx")
        buf = BytesIO()
        doc.save(buf)
        docx_bytes = buf.getvalue()

        text = handler.extract_text(docx_bytes, ".docx")
        assert "Hello from docx" in text


class TestGenerateEmbedding:
    def test_calls_bedrock(self, patch_aws):
        handler = patch_aws
        embedding = handler.generate_embedding("test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 1536

    def test_truncates_long_text(self, patch_aws):
        handler = patch_aws
        long_text = "x" * 30000
        handler.generate_embedding(long_text)
        call_args = handler.bedrock.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert len(body["inputText"]) == 20000


class TestProcessDocument:
    def test_processes_txt(self, patch_aws):
        handler = patch_aws
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        from opensearchpy.helpers import bulk as real_bulk

        with patch.object(handler, "get_opensearch_client", return_value=mock_client), \
             patch("src.lambda.ingest.handler.bulk", return_value=(1, [])):
            result = handler.process_document("bucket", "docs/readme.txt")
            assert result == 1

    def test_processes_md(self, patch_aws):
        handler = patch_aws
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True

        with patch.object(handler, "get_opensearch_client", return_value=mock_client), \
             patch("src.lambda.ingest.handler.bulk", return_value=(2, [])):
            result = handler.process_document("bucket", "docs/guide.md")
            assert result == 2

    def test_reports_bulk_errors(self, patch_aws):
        handler = patch_aws
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True

        with patch.object(handler, "get_opensearch_client", return_value=mock_client), \
             patch("src.lambda.ingest.handler.bulk", return_value=(0, ["error1"])):
            result = handler.process_document("bucket", "docs/file.txt")
            assert result == 0


class TestGetOpensearchClient:
    def test_returns_client(self, patch_aws):
        handler = patch_aws
        with patch("src.lambda.ingest.handler.OpenSearch") as MockOS:
            client = handler.get_opensearch_client()
            MockOS.assert_called_once()


class TestEnsureIndexExists:
    def test_creates_index_if_missing(self, patch_aws):
        handler = patch_aws
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False
        handler.ensure_index_exists(mock_client)
        mock_client.indices.create.assert_called_once()

    def test_skips_if_exists(self, patch_aws):
        handler = patch_aws
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        handler.ensure_index_exists(mock_client)
        mock_client.indices.create.assert_not_called()
