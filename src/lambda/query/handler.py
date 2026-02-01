"""
Query Lambda: Handles RAG queries via API Gateway.

Flow:
1. Receive question from API Gateway
2. Generate embedding using Bedrock Titan
3. Search OpenSearch for relevant chunks
4. Send context + question to Claude
5. Return cited answer
"""

import json
import os
import boto3
from anthropic import Anthropic
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth


# Initialize clients
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
anthropic_client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

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


def get_opensearch_client():
    """Create OpenSearch client."""
    endpoint = os.environ['OPENSEARCH_ENDPOINT']
    # Remove https:// if present
    host = endpoint.replace('https://', '')
    
    return OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30
    )


def generate_embedding(text: str) -> list[float]:
    """Generate embedding using Amazon Bedrock Titan."""
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({'inputText': text})
    )
    
    result = json.loads(response['body'].read())
    return result['embedding']


def search_documents(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search OpenSearch for similar documents."""
    client = get_opensearch_client()
    index_name = 'documents'
    
    query = {
        'size': top_k,
        'query': {
            'knn': {
                'embedding': {
                    'vector': query_embedding,
                    'k': top_k
                }
            }
        },
        '_source': ['text', 'metadata']
    }
    
    try:
        response = client.search(index=index_name, body=query)
        
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'text': hit['_source']['text'],
                'metadata': hit['_source']['metadata'],
                'score': hit['_score']
            })
        
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_response(question: str, context_chunks: list[dict]) -> dict:
    """Generate response using Claude."""
    
    # Build context string
    context_parts = []
    sources = []
    
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk['metadata'].get('source', 'unknown')
        context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")
        sources.append({
            'source': source,
            'score': chunk.get('score', 0)
        })
    
    context_string = "\n\n---\n\n".join(context_parts)
    
    system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only answer based on the provided context. If the context doesn't contain enough information, say so.
2. Cite your sources by referencing [Source N] when you use information from that source.
3. Be concise but complete.
4. If the question cannot be answered from the context, explain what information is missing."""

    user_prompt = f"""Context:
{context_string}

---

Question: {question}

Please answer the question based on the context provided above."""

    response = anthropic_client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=1024,
        system=system_prompt,
        messages=[{'role': 'user', 'content': user_prompt}]
    )
    
    return {
        'answer': response.content[0].text,
        'sources': sources,
        'usage': {
            'input_tokens': response.usage.input_tokens,
            'output_tokens': response.usage.output_tokens
        }
    }


def lambda_handler(event, context):
    """Main Lambda handler."""
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))
        question = body.get('question', '').strip()
        
        if not question:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'Question is required'})
            }
        
        print(f"Question: {question}")
        
        # Step 1: Generate embedding for the question
        query_embedding = generate_embedding(question)
        print(f"Generated embedding: {len(query_embedding)} dimensions")
        
        # Step 2: Search for relevant chunks
        chunks = search_documents(query_embedding, top_k=5)
        print(f"Found {len(chunks)} relevant chunks")
        
        if not chunks:
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({
                    'answer': 'No relevant documents found in the knowledge base.',
                    'sources': [],
                    'usage': {'input_tokens': 0, 'output_tokens': 0}
                })
            }
        
        # Step 3: Generate response with Claude
        result = generate_response(question, chunks)
        print(f"Generated response: {result['usage']}")
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps(result)
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': str(e)})
        }
```

Create `src/lambda/query/requirements.txt`:
```
anthropic>=0.40.0
opensearch-py>=2.4.0
requests-aws4auth>=1.2.0
boto3>=1.34.0