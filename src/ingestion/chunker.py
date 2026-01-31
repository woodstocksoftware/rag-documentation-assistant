"""
Document chunking for RAG pipeline.

Chunks documents into smaller pieces suitable for embedding and retrieval.
Uses recursive character splitting with overlap to maintain context.
"""

import tiktoken


class DocumentChunker:
    """
    Splits documents into chunks optimized for RAG retrieval.
    
    Why these defaults?
    - chunk_size=500 tokens: Small enough for precise retrieval, large enough for context
    - chunk_overlap=50 tokens: Prevents cutting sentences mid-thought
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model: str = "cl100k_base"  # Tokenizer used by Claude and GPT-4
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(model)
        
        # Split on these characters, in order of preference
        # Tries to keep paragraphs together, then sentences, then words
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a string."""
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, text: str, metadata: dict = None) -> list[dict]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: The document text to chunk
            metadata: Optional dict of metadata to attach to each chunk
                     (e.g., {"source": "policy.pdf", "page": 1})
        
        Returns:
            List of chunk dicts: {"text": str, "metadata": dict, "token_count": int}
        """
        if metadata is None:
            metadata = {}
        
        chunks = self._recursive_split(text, self.separators)
        
        return [
            {
                "text": chunk,
                "metadata": {**metadata, "chunk_index": i},
                "token_count": self.count_tokens(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text, trying each separator in order.
        
        This is the core algorithm:
        1. Try to split on paragraph breaks (\n\n)
        2. If chunks still too big, split on line breaks (\n)
        3. Then sentences (. )
        4. Then words ( )
        5. Finally, hard character split as last resort
        """
        if not text:
            return []
        
        # Base case: text fits in one chunk
        if self.count_tokens(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Try each separator
        for i, separator in enumerate(separators):
            if separator and separator in text:
                splits = text.split(separator)
                chunks = []
                current_chunk = ""
                
                for split in splits:
                    # Add separator back (except for empty string separator)
                    piece = split + separator if separator else split
                    
                    # Would adding this piece exceed chunk size?
                    combined = current_chunk + piece
                    if self.count_tokens(combined) <= self.chunk_size:
                        current_chunk = combined
                    else:
                        # Save current chunk if it has content
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        
                        # Is this single piece too big? Recurse with finer separator
                        if self.count_tokens(piece) > self.chunk_size:
                            sub_chunks = self._recursive_split(piece, separators[i+1:])
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = piece
                
                # Don't forget the last chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Add overlap between chunks
                return self._add_overlap(chunks)
        
        # Last resort: hard split by characters
        return self._hard_split(text)
    
    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """
        Add overlap between chunks to maintain context continuity.
        
        Takes the last N tokens from chunk[i] and prepends to chunk[i+1].
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Get overlap from end of previous chunk
            prev_tokens = self.tokenizer.encode(prev_chunk)
            overlap_tokens = prev_tokens[-self.chunk_overlap:]
            overlap_text = self.tokenizer.decode(overlap_tokens)
            
            # Prepend overlap to current chunk
            overlapped.append(overlap_text + " " + curr_chunk)
        
        return overlapped
    
    def _hard_split(self, text: str) -> list[str]:
        """Last resort: split by token count directly."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        
        return chunks


# Quick test when run directly
if __name__ == "__main__":
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    sample_text = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses 
    by providing relevant context from a knowledge base.
    
    The process works in three steps. First, the user's question is converted into 
    an embedding vector. Second, this vector is used to search a database of document 
    embeddings to find the most similar chunks. Third, these chunks are added to the 
    prompt as context for the LLM to reference when generating its answer.
    
    RAG solves the hallucination problem by grounding responses in actual documents.
    It also allows LLMs to access private or recent information not in their training data.
    """
    
    chunks = chunker.chunk_text(sample_text, metadata={"source": "rag_intro.txt"})
    
    print(f"Split into {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"--- Chunk {chunk['metadata']['chunk_index']} ({chunk['token_count']} tokens) ---")
        print(chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'])
        print()