"""
Vector store for RAG pipeline.

Stores document embeddings and enables semantic search.
Uses ChromaDB locally, designed for easy swap to OpenSearch in production.
"""

import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    Stores and searches document embeddings.
    
    Wraps ChromaDB for local development.
    Interface designed to match what we'll use with OpenSearch.
    """
    
    def __init__(self, collection_name: str = "documents", persist_dir: str = "./data/chroma"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name for this collection of documents
            persist_dir: Directory to persist the database
        """
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        # We manage embeddings ourselves, so we disable ChromaDB's auto-embedding
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"✓ Vector store initialized: {collection_name}")
        print(f"  Existing documents: {self.collection.count()}")
    
    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Add embedded chunks to the vector store.
        
        Args:
            chunks: List of chunks with 'text', 'embedding', and 'metadata' keys
        """
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        ids = [f"{chunk['metadata']['source']}_{chunk['metadata']['chunk_index']}" for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Upsert (add or update)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(chunks)} chunks to vector store")
    
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: The embedding vector of the search query
            top_k: Number of results to return
            
        Returns:
            List of results with 'text', 'metadata', and 'score' keys
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
        
        return formatted
    
    def clear(self) -> None:
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("✓ Vector store cleared")
    
    def count(self) -> int:
        """Return the number of documents in the store."""
        return self.collection.count()


if __name__ == "__main__":
    # Test the vector store
    from src.ingestion.embeddings import EmbeddingModel
    
    # Initialize
    store = VectorStore(collection_name="test_collection", persist_dir="./data/chroma_test")
    store.clear()  # Start fresh for testing
    
    model = EmbeddingModel()
    
    # Create some test chunks with embeddings
    test_chunks = [
        {"text": "Our return policy allows refunds within 30 days of purchase.", 
         "metadata": {"source": "policy.txt", "chunk_index": 0}},
        {"text": "To contact customer support, email support@example.com or call 1-800-555-0123.", 
         "metadata": {"source": "support.txt", "chunk_index": 0}},
        {"text": "Shipping takes 3-5 business days for standard delivery.", 
         "metadata": {"source": "shipping.txt", "chunk_index": 0}},
        {"text": "Premium members get free expedited shipping on all orders.", 
         "metadata": {"source": "membership.txt", "chunk_index": 0}},
    ]
    
    # Generate embeddings
    chunks_with_embeddings = model.embed_chunks(test_chunks)
    
    # Add to store
    store.add_chunks(chunks_with_embeddings)
    
    # Test search
    print("\n--- Search Test ---")
    query = "How can I get my money back?"
    print(f"Query: '{query}'\n")
    
    query_embedding = model.embed_text(query)
    results = store.search(query_embedding, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [Score: {result['score']:.3f}] {result['metadata']['source']}")
        print(f"   {result['text'][:80]}...")
        print()