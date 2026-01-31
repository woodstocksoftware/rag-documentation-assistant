"""
Response generation for RAG pipeline.

Takes retrieved context and user question, generates answer using Claude.
"""

import os
from anthropic import Anthropic


class ResponseGenerator:
    """
    Generates responses using Claude with retrieved context.
    
    This is where RAG comes together:
    1. Receive user question + relevant chunks
    2. Construct a prompt with context
    3. Claude generates a grounded answer
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the generator.
        
        Args:
            model: Claude model to use. Sonnet is a good balance of
                   quality and cost for RAG applications.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = Anthropic(api_key=api_key)
        self.model = model
        
        print(f"✓ Generator initialized with {model}")
    
    def generate(
        self,
        question: str,
        context_chunks: list[dict],
        max_tokens: int = 1024
    ) -> dict:
        """
        Generate a response to the question using retrieved context.
        
        Args:
            question: The user's question
            context_chunks: Retrieved chunks with 'text', 'metadata', 'score'
            max_tokens: Maximum response length
            
        Returns:
            Dict with 'answer', 'sources', and 'usage' keys
        """
        # Build context string with source attribution
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk["metadata"].get("source", "unknown")
            context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")
            sources.append({
                "source": source,
                "score": chunk.get("score", 0),
                "chunk_index": chunk["metadata"].get("chunk_index", 0)
            })
        
        context_string = "\n\n---\n\n".join(context_parts)
        
        # Construct the RAG prompt
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

        # Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return {
            "answer": response.content[0].text,
            "sources": sources,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }


if __name__ == "__main__":
    # Test the generator with mock context
    generator = ResponseGenerator()
    
    # Simulate retrieved chunks
    mock_chunks = [
        {
            "text": "Our return policy allows refunds within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days.",
            "metadata": {"source": "return_policy.pdf", "chunk_index": 0},
            "score": 0.89
        },
        {
            "text": "To initiate a return, log into your account and select 'Order History'. Click 'Return Item' next to the product. Print the prepaid shipping label.",
            "metadata": {"source": "return_policy.pdf", "chunk_index": 1},
            "score": 0.82
        }
    ]
    
    question = "How do I return a product and get a refund?"
    
    print(f"Question: {question}\n")
    print("=" * 50)
    
    result = generator.generate(question, mock_chunks)
    
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\n--- Sources Used ---")
    for src in result["sources"]:
        print(f"  • {src['source']} (relevance: {src['score']:.2f})")
    print(f"\n--- Token Usage ---")
    print(f"  Input: {result['usage']['input_tokens']}")
    print(f"  Output: {result['usage']['output_tokens']}")