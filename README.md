# RAG Documentation Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions based on your document knowledge base. Deployed on AWS with serverless architecture.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![AWS](https://img.shields.io/badge/AWS-Serverless-orange)
![Claude](https://img.shields.io/badge/LLM-Claude-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

## What It Does

Upload documents → Ask questions → Get accurate, cited answers.
```
curl -X POST https://your-api.execute-api.us-east-1.amazonaws.com/dev/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'

{
  "answer": "Based on the provided context, items may be returned within 30 days of purchase...",
  "sources": [{"source": "return_policy.md", "score": 0.84}]
}
```

## Architecture

### AWS Production Deployment
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER REQUEST                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    API Gateway      │
                         │    (REST API)       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Lambda (Query)    │
                         │                     │
                         │ 1. Embed question   │
                         │ 2. Search vectors   │
                         │ 3. Call Claude      │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
   │ Bedrock Titan   │   │   OpenSearch    │   │   Claude API    │
   │ (Embeddings)    │   │   Serverless    │   │   (Answers)     │
   └─────────────────┘   └─────────────────┘   └─────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOCUMENT INGESTION                                │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐       ┌──────────┐       ┌─────────────────┐
  │    S3    │──────▶│   SQS    │──────▶│ Lambda (Ingest) │
  │  Bucket  │trigger│  Queue   │       │                 │
  └──────────┘       └──────────┘       │ 1. Download doc │
                                        │ 2. Chunk text   │
       Drop files here                  │ 3. Embed chunks │
       Auto-indexed!                    │ 4. Store vectors│
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────┐
                                      │   OpenSearch    │
                                      │   Serverless    │
                                      └─────────────────┘
```

### Components

| Component | Service | Purpose |
|-----------|---------|---------|
| **API** | API Gateway | REST endpoint for queries |
| **Query Processing** | Lambda | Orchestrates RAG pipeline |
| **Document Ingestion** | Lambda + SQS | Async document processing |
| **Vector Store** | OpenSearch Serverless | k-NN similarity search |
| **Embeddings** | Bedrock Titan | 1536-dimension vectors |
| **Generation** | Claude API | Answer synthesis with citations |
| **Storage** | S3 | Document uploads |
| **IaC** | SAM/CloudFormation | Infrastructure as Code |

## Features

- **Semantic Search**: Finds relevant content even without keyword matches
- **Source Citations**: Every answer references its source documents
- **Auto-Ingestion**: Drop files in S3 → automatically indexed
- **Serverless**: Pay only for what you use, scales automatically
- **Production-Ready**: Error handling, logging, observability

## Quick Start

### Prerequisites

- Python 3.12+
- AWS CLI configured
- SAM CLI installed
- Anthropic API key

### Local Development
```bash
# Clone and setup
git clone https://github.com/woodstocksoftware/rag-documentation-assistant.git
cd rag-documentation-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# Run locally with Gradio UI
python app.py
# Open http://localhost:7860
```

### AWS Deployment
```bash
# Build
sam build --template infrastructure/template.yaml

# Deploy
sam deploy \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
  --parameter-overrides AnthropicApiKey=$ANTHROPIC_API_KEY

# Configure S3 notifications (replace with your bucket name)
aws s3api put-bucket-notification-configuration \
  --bucket rag-documents-YOUR_ACCOUNT_ID-dev \
  --notification-configuration '{
    "QueueConfigurations": [{
      "QueueArn": "arn:aws:sqs:us-east-1:YOUR_ACCOUNT_ID:rag-document-processing-dev",
      "Events": ["s3:ObjectCreated:*"]
    }]
  }'
```

### Usage

**Upload documents:**
```bash
aws s3 cp your-document.pdf s3://rag-documents-YOUR_ACCOUNT_ID-dev/
```

**Query the API:**
```bash
curl -X POST https://YOUR_API.execute-api.us-east-1.amazonaws.com/dev/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

## Project Structure
```
rag-documentation-assistant/
├── app.py                      # Gradio UI for local development
├── infrastructure/
│   └── template.yaml           # SAM/CloudFormation template
├── src/
│   ├── ingestion/
│   │   ├── chunker.py          # Document chunking logic
│   │   ├── embeddings.py       # Local embedding model
│   │   ├── loader.py           # Document text extraction
│   │   └── pipeline.py         # Ingestion orchestration
│   ├── query/
│   │   ├── generator.py        # Claude response generation
│   │   └── rag.py              # RAG pipeline
│   ├── shared/
│   │   └── vector_store.py     # ChromaDB wrapper
│   └── lambda/
│       ├── query/
│       │   └── handler.py      # Query Lambda function
│       └── ingest/
│           └── handler.py      # Ingest Lambda function
├── sample_docs/                # Test documents
└── requirements.txt
```

## How RAG Works

1. **Chunking**: Documents are split into overlapping chunks (~500 tokens)
2. **Embedding**: Each chunk is converted to a 1536-dimension vector
3. **Indexing**: Vectors are stored in OpenSearch with k-NN indexing
4. **Query**: User question is embedded and used for similarity search
5. **Retrieval**: Top-k most similar chunks are retrieved
6. **Generation**: Claude generates an answer using retrieved context
7. **Citation**: Sources are tracked and included in the response

## Cost Estimate

| Service | Monthly Cost (Dev) |
|---------|-------------------|
| OpenSearch Serverless | ~$25-30 |
| Lambda | < $1 |
| Bedrock Titan | < $1 |
| Claude API | ~$5-20 (usage dependent) |
| S3, SQS, API Gateway | < $1 |
| **Total** | **~$35-50/month** |

*Delete the stack when not in use to minimize costs.*

## Cleanup
```bash
# Delete all AWS resources
sam delete --stack-name rag-documentation-assistant
```

## Tech Stack

- **Runtime**: Python 3.12
- **LLM**: Claude Sonnet (Anthropic)
- **Embeddings**: Amazon Bedrock Titan (AWS) / sentence-transformers (local)
- **Vector DB**: OpenSearch Serverless (AWS) / ChromaDB (local)
- **Infrastructure**: AWS SAM, CloudFormation
- **UI**: Gradio

## License

MIT

---

Built by [Jim Williams](https://www.linkedin.com/in/woodstocksoftware/) | [GitHub](https://github.com/woodstocksoftware)

## Authentication

The API requires an API key for all requests. Include it in the `x-api-key` header:
```bash
curl -X POST https://YOUR_API.execute-api.us-east-1.amazonaws.com/dev/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{"question": "Your question here"}'
```

### Rate Limits

| Limit | Value |
|-------|-------|
| Daily quota | 1,000 requests |
| Rate limit | 5 requests/second |
| Burst limit | 10 requests |

To get an API key, deploy your own instance or contact the maintainer.

## Monitoring & Alerts

### CloudWatch Dashboard

View real-time metrics at:
```
https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=RAG-Documentation-Assistant
```

### Alarms

| Alarm | Condition | Action |
|-------|-----------|--------|
| RAG-Query-Errors | Any Lambda errors | Email alert |
| RAG-Ingest-Errors | Any ingest errors | Email alert |
| RAG-Query-HighLatency | Avg response > 10s | Email alert |
| RAG-API-5xxErrors | Any server errors | Email alert |
| RAG-API-4xxErrors | > 50 client errors/5min | Email alert |
| RAG-SQS-Backlog | > 10 messages stuck | Email alert |
| RAG-Lambda-Throttled | Any throttling | Email alert |

### Setting Up Alerts
```bash
# Create SNS topic
aws sns create-topic --name rag-alerts-dev

# Subscribe your email
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:YOUR_ACCOUNT_ID:rag-alerts-dev \
  --protocol email \
  --notification-endpoint your@email.com

# Create alarms
./scripts/create-alarms.sh
```
