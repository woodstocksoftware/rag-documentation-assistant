# CLAUDE.md — RAG Documentation Assistant

> **Purpose:** Production-ready RAG system with document ingestion, semantic search, and cited answers
> **Owner:** Jim Williams - Woodstock Software LLC
> **Repo:** woodstocksoftware/rag-documentation-assistant (public)

---

## Tech Stack

- Python 3.12
- Anthropic Claude Sonnet (response generation)
- ChromaDB (local vector store) / AWS OpenSearch Serverless (production)
- sentence-transformers all-MiniLM-L6-v2 (local embeddings, 384-dim) / AWS Bedrock Titan (production, 1536-dim)
- pypdf, python-docx (document parsing)
- tiktoken (token counting)
- Gradio (local UI)
- AWS Lambda + SAM (production deployment)
- pytest (96 tests)

## Project Structure

```
rag-documentation-assistant/
├── app.py                         # Gradio UI (3 tabs: chat, upload, manage)
├── requirements.txt
├── pytest.ini
├── src/
│   ├── ingestion/
│   │   ├── chunker.py             # Recursive text splitting with overlap
│   │   ├── embeddings.py          # Embedding model wrapper
│   │   ├── loader.py              # PDF/DOCX/TXT/MD document loader
│   │   └── pipeline.py            # Ingestion orchestrator
│   ├── query/
│   │   ├── rag.py                 # End-to-end RAG pipeline
│   │   └── generator.py           # Claude response generation
│   ├── shared/
│   │   └── vector_store.py        # ChromaDB wrapper
│   └── lambda/
│       ├── query/handler.py       # Lambda: API Gateway queries
│       └── ingest/handler.py      # Lambda: S3→SQS ingestion
├── tests/                         # 96 tests across 8 modules
├── infrastructure/
│   └── template.yaml              # AWS SAM/CloudFormation
├── scripts/
│   └── create-alarms.sh           # CloudWatch alarm setup
├── sample_docs/                   # 3 sample documents
├── data/                          # Local ChromaDB stores
├── LICENSE                        # MIT
└── README.md
```

## How to Run

```bash
cd /Users/james/projects/rag-documentation-assistant
source venv/bin/activate
export ANTHROPIC_API_KEY="sk-ant-..."
python app.py
# Opens http://localhost:7860
```

## How to Test

```bash
cd /Users/james/projects/rag-documentation-assistant
source venv/bin/activate
pytest tests/ -v
# 96 tests, all passing (~27s)
```

### Test Breakdown
| Module | Tests | Covers |
|--------|-------|--------|
| test_chunker.py | 20 | Token counting, recursive splitting, overlap |
| test_vector_store.py | 14 | Add/search/clear, similarity ranking |
| test_loader.py | 10 | PDF, DOCX, TXT, MD parsing |
| test_embeddings.py | 8 | Model loading, batch processing, similarity |
| test_generator.py | 8 | API key validation, response structure |
| test_rag_pipeline.py | 7 | Full query flow, ingestion, edge cases |
| test_ingest_handler.py | 8 | Lambda SQS→S3 flow |
| test_query_handler.py | 8 | Query Lambda, error responses |

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes | Claude API for generation |
| `OPENSEARCH_ENDPOINT` | Production only | AWS OpenSearch Serverless |

## Architecture

### Local (Gradio)
```
Upload doc → Loader (PDF/DOCX/TXT/MD) → Chunker (~500 tokens, 50 overlap)
  → Embeddings (all-MiniLM-L6-v2) → ChromaDB

Question → Embed → ChromaDB search (top-k) → Claude generation → cited answer
```

### Production (AWS)
```
S3 upload → SQS → Ingest Lambda → Bedrock Titan embeddings → OpenSearch

API Gateway → Query Lambda → Bedrock Titan embed → OpenSearch search → Claude → response
```

## Key Patterns

- **Recursive chunking**: Tries paragraph → line → sentence → word → char splits. Token-aware via tiktoken. ~500 tokens per chunk, 50 overlap.
- **Dual-mode architecture**: ChromaDB + MiniLM locally, OpenSearch + Bedrock Titan in production
- **Source citation**: System prompt enforces `[Title](url)` inline citations
- **Token tracking**: Reports input + output token usage on every query
- **Cosine similarity**: Scores normalized to [0,1]
- **Lambda architecture**: Query handler is synchronous (API Gateway), ingest handler is async (SQS)

## AWS Deployment

```bash
sam build --template infrastructure/template.yaml
sam deploy --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
  --parameter-overrides AnthropicApiKey=$ANTHROPIC_API_KEY
```

API auth: `x-api-key` header, rate limited (5 req/s, 100/day).

## What's Missing

- [ ] CI workflow (.github/workflows/)
- [ ] pyproject.toml
- [ ] Dirty git: minor `flush=True` removal in `src/query/rag.py` — commit or discard
