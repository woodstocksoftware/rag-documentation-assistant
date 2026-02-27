"""
Embedding generation for RAG pipeline.

Converts text chunks into vector embeddings for semantic search.
Uses a local model for development, easily swappable to Bedrock for production.
"""

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Generates embeddings for text chunks.

    Uses all-MiniLM-L6-v2 locally—fast, good quality, 384 dimensions.
    Production would use AWS Bedrock Titan embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model name. Default is a good balance
                       of speed and quality for development.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded. Embedding dimension: {self.dimension}")

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of chunk dicts with 'text' key

        Returns:
            Same chunks with 'embedding' key added
        """
        # Extract texts for batch processing (much faster)
        texts = [chunk["text"] for chunk in chunks]

        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()

        print(f"✓ Generated {len(embeddings)} embeddings")
        return chunks


if __name__ == "__main__":
    # Test the embedding model
    model = EmbeddingModel()

    # Test single embedding
    text = "How do I return a product for a refund?"
    embedding = model.embed_text(text)
    print("\nSingle embedding test:")
    print(f"  Text: '{text}'")
    print(f"  Embedding dimensions: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")

    # Test semantic similarity
    print("\n--- Semantic Similarity Test ---")
    sentences = [
        "How do I return a product for a refund?",
        "What is the return policy?",
        "Can I get my money back?",
        "What's the weather forecast?",
        "How do I contact support?"
    ]

    embeddings = model.model.encode(sentences)

    # Calculate similarity to first sentence
    from numpy import dot
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    print(f"\nSimilarity to: '{sentences[0]}'\n")
    for i, (sentence, emb) in enumerate(zip(sentences[1:], embeddings[1:]), 1):
        similarity = cosine_similarity(embeddings[0], emb)
        bar = "█" * int(similarity * 20)
        print(f"  {similarity:.3f} {bar} '{sentence}'")
