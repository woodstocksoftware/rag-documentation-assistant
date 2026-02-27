"""
Ingest Lambda: Processes documents uploaded to S3.

Flow:
1. Triggered by SQS (which receives S3 events)
2. Download document from S3
3. Extract text and chunk it
4. Generate embeddings using Bedrock Titan
5. Store in OpenSearch
"""

import json
import os
from io import BytesIO

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from requests_aws4auth import AWS4Auth

# Initialize clients
s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# OpenSearch connection
region = 'us-east-1'
service = 'aoss'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    service,
    session_token=credentials.token
)

# Index configuration
INDEX_NAME = 'documents'
EMBEDDING_DIMENSION = 1536  # Titan embedding dimension


def get_opensearch_client():
    """Create OpenSearch client."""
    endpoint = os.environ['OPENSEARCH_ENDPOINT']
    host = endpoint.replace('https://', '')

    return OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30
    )


def ensure_index_exists(client):
    """Create the vector index if it doesn't exist."""
    if not client.indices.exists(index=INDEX_NAME):
        index_body = {
            'settings': {
                'index': {
                    'knn': True
                }
            },
            'mappings': {
                'properties': {
                    'text': {'type': 'text'},
                    'embedding': {
                        'type': 'knn_vector',
                        'dimension': EMBEDDING_DIMENSION,
                        'method': {
                            'name': 'hnsw',
                            'space_type': 'cosinesimil',
                            'engine': 'faiss'
                        }
                    },
                    'metadata': {
                        'type': 'object',
                        'properties': {
                            'source': {'type': 'keyword'},
                            'chunk_index': {'type': 'integer'},
                            'file_type': {'type': 'keyword'}
                        }
                    }
                }
            }
        }

        client.indices.create(index=INDEX_NAME, body=index_body)
        print(f"Created index: {INDEX_NAME}")


def extract_text(file_bytes, file_type):
    """Extract text from document bytes."""

    if file_type in ['.txt', '.md']:
        return file_bytes.decode('utf-8')

    elif file_type == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return '\n\n'.join(pages)

    elif file_type == '.docx':
        from docx import Document
        doc = Document(BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return '\n\n'.join(paragraphs)

    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Simple chunking by character count with overlap.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        if end < len(text):
            for sep in ['. ', '.\n', '\n\n', '\n']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def generate_embedding(text):
    """Generate embedding using Amazon Bedrock Titan."""
    # Truncate text if too long (Titan has 8k token limit)
    max_chars = 20000
    if len(text) > max_chars:
        text = text[:max_chars]

    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({'inputText': text})
    )

    result = json.loads(response['body'].read())
    return result['embedding']


def process_document(bucket, key):
    """Process a single document from S3."""
    print(f"Processing: s3://{bucket}/{key}")

    # Get file extension
    file_type = '.' + key.rsplit('.', 1)[-1].lower()
    file_name = key.rsplit('/', 1)[-1]

    # Download from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = response['Body'].read()
    print(f"Downloaded {len(file_bytes)} bytes")

    # Extract text
    text = extract_text(file_bytes, file_type)
    print(f"Extracted {len(text)} characters")

    # Chunk text
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    # Generate embeddings and prepare documents
    documents = []
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)

        doc = {
            '_index': INDEX_NAME,
            '_source': {
                'text': chunk,
                'embedding': embedding,
                'metadata': {
                    'source': file_name,
                    'chunk_index': i,
                    'file_type': file_type,
                    's3_key': key
                }
            }
        }
        documents.append(doc)
        print(f"  Embedded chunk {i + 1}/{len(chunks)}")

    # Bulk index to OpenSearch
    client = get_opensearch_client()
    ensure_index_exists(client)

    success, errors = bulk(client, documents)
    print(f"Indexed {success} documents, {len(errors)} errors")

    if errors:
        print(f"Errors: {errors}")

    return success


def lambda_handler(event, context):
    """Main Lambda handler - processes SQS messages from S3 events."""
    print(f"Received event: {json.dumps(event)}")

    processed = 0
    errors = []

    for record in event.get('Records', []):
        try:
            # Parse SQS message body (contains S3 event)
            body = json.loads(record['body'])

            # Handle S3 event notification
            for s3_record in body.get('Records', []):
                bucket = s3_record['s3']['bucket']['name']
                key = s3_record['s3']['object']['key']

                # Skip if not a supported file type
                if not key.lower().endswith(('.pdf', '.txt', '.md', '.docx')):
                    print(f"Skipping unsupported file: {key}")
                    continue

                success = process_document(bucket, key)
                processed += success

        except Exception as e:
            error_msg = f"Error processing record: {str(e)}"
            print(error_msg)
            errors.append(error_msg)

    result = {
        'processed': processed,
        'errors': errors
    }

    print(f"Result: {json.dumps(result)}")
    return result
