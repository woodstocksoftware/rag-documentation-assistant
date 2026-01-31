"""
Complete RAG pipeline.

Combines retrieval and generation into a single query interface.
This is the main entry point for asking questions.
"""

from src.ingestion.embeddings import EmbeddingModel
from src.shared.vector_store import VectorStore
from src.query.generator import ResponseGenerator


class RAGPipeline:
    """
    End-to-end RAG pipeline.
    
    Usage:
        rag = RAGPipeline()
        result = rag.query("What is the return policy?")
        print(result["answer"])
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_dir: str = "./data/chroma",
        top_k: int = 5,
        model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            collection_name: Vector store collection name
            persist_dir: Where to persist the vector database
            top_k: Number of chunks to retrieve per query
            model: Claude model for generation
        """
        print("Initializing RAG pipeline...")
        
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(collection_name, persist_dir)
        self.generator = ResponseGenerator(model)
        self.top_k = top_k
        
        print("✓ RAG pipeline ready\n")
    
    def query(self, question: str, top_k: int = None) -> dict:
        """
        Answer a question using RAG.
        
        Args:
            question: The user's question
            top_k: Override default number of chunks to retrieve
            
        Returns:
            Dict with 'answer', 'sources', 'usage', and 'retrieved_chunks'
        """
        k = top_k or self.top_k
        
        # Step 1: Embed the question
        print(f"📝 Question: {question}")
        query_embedding = self.embedding_model.embed_text(question)
        
        # Step 2: Retrieve relevant chunks
        print(f"🔍 Retrieving top {k} relevant chunks...")
        chunks = self.vector_store.search(query_embedding, top_k=k)
        
        if not chunks:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "sources": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "retrieved_chunks": []
            }
        
        # Show what was retrieved
        print(f"   Found {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"   {i}. [{chunk['score']:.3f}] {chunk['metadata']['source']}")
        
        # Step 3: Generate response
        print("🤖 Generating response...")
        result = self.generator.generate(question, chunks)
        
        # Add retrieved chunks to result for transparency
        result["retrieved_chunks"] = chunks
        
        return result
    
    def ingest(self, chunks: list[dict]) -> None:
        """
        Add pre-processed chunks to the knowledge base.
        
        Args:
            chunks: Chunks with 'text' and 'metadata' keys (embeddings will be generated)
        """
        # Generate embeddings if not present
        if chunks and "embedding" not in chunks[0]:
            chunks = self.embedding_model.embed_chunks(chunks)
        
        self.vector_store.add_chunks(chunks)


if __name__ == "__main__":
    from src.ingestion.pipeline import IngestionPipeline
    
    # Initialize pipelines
    ingestion = IngestionPipeline(chunk_size=300, chunk_overlap=50)
    rag = RAGPipeline(collection_name="demo", persist_dir="./data/chroma_demo")
    
    # Clear previous data for clean demo
    rag.vector_store.clear()
    
    # Create sample documents
    print("=" * 60)
    print("STEP 1: Creating sample knowledge base")
    print("=" * 60)
    
    from pathlib import Path
    
    sample_docs = Path("sample_docs")
    sample_docs.mkdir(exist_ok=True)
    
    # Document 1: Return Policy
    (sample_docs / "return_policy.md").write_text("""# Return Policy

## Eligibility
Items may be returned within 30 days of purchase. Products must be unused, 
in original packaging, and accompanied by a receipt or proof of purchase.

## Non-Returnable Items
The following items cannot be returned:
- Gift cards
- Downloadable software
- Personalized items
- Items marked "Final Sale"

## How to Return
1. Log into your account at our website
2. Go to Order History and select the order
3. Click "Return Item" and select a reason
4. Print the prepaid shipping label
5. Pack the item securely and drop off at any UPS location

## Refund Timeline
- Refunds are processed within 5-7 business days of receiving the return
- Original payment method will be credited
- Shipping costs are non-refundable unless the return is due to our error

## Exchanges
We do not offer direct exchanges. Please return the original item and place 
a new order for the desired item.
""")
    
    # Document 2: Shipping Information
    (sample_docs / "shipping_info.md").write_text("""# Shipping Information

## Delivery Options

### Standard Shipping
- 5-7 business days
- Free on orders over $50
- $5.99 for orders under $50

### Express Shipping
- 2-3 business days
- $12.99 flat rate

### Next Day Delivery
- Order by 2pm for next business day delivery
- $24.99 flat rate
- Not available for PO boxes

## Order Tracking
All orders include tracking. Once shipped, you'll receive an email with 
your tracking number. Track your package at our website or directly on 
the carrier's site.

## International Shipping
We currently ship to Canada and Mexico only. International orders may be 
subject to customs fees and import duties, which are the customer's responsibility.

## Premium Membership
Premium members receive:
- Free express shipping on all orders
- Early access to sales
- Exclusive member discounts
""")
    
    # Document 3: Account & Support
    (sample_docs / "support.md").write_text("""# Customer Support

## Contact Us

### Email Support
- General inquiries: help@example.com
- Order issues: orders@example.com
- Response time: 24-48 hours

### Phone Support
- 1-800-555-0123
- Monday-Friday: 8am-8pm EST
- Saturday: 9am-5pm EST
- Closed Sunday

### Live Chat
Available on our website during business hours. Average wait time: 2 minutes.

## Account Management

### Password Reset
1. Click "Forgot Password" on the login page
2. Enter your email address
3. Check your email for reset link
4. Link expires in 24 hours

### Update Payment Method
1. Log into your account
2. Go to Settings > Payment Methods
3. Click "Add New" or edit existing
4. Save changes

### Delete Account
To delete your account, please contact support. Note that this action is 
irreversible and you will lose access to order history and saved preferences.
""")
    
    print(f"Created 3 sample documents in {sample_docs}/\n")
    
    # Ingest documents
    print("=" * 60)
    print("STEP 2: Ingesting documents into knowledge base")
    print("=" * 60)
    
    chunks = ingestion.process_directory(sample_docs)
    rag.ingest(chunks)
    
    # Test queries
    print("\n" + "=" * 60)
    print("STEP 3: Testing RAG queries")
    print("=" * 60 + "\n", flush=True)
    
    test_questions = [
        "How do I get a refund?",
        "What shipping options do you have?",
        "How can I contact customer support?",
    ]
    
    for question in test_questions:
        print("-" * 60, flush=True)
        result = rag.query(question, top_k=3)
        print(f"\n💬 Answer:\n{result['answer']}", flush=True)
        print(f"\n📊 Tokens used: {result['usage']['input_tokens']} in / {result['usage']['output_tokens']} out", flush=True)
        print("\n", flush=True)