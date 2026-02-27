"""
Ingestion pipeline: Load documents, chunk them, prepare for embedding.
"""

from pathlib import Path

from .chunker import DocumentChunker
from .loader import DocumentLoader


class IngestionPipeline:
    """
    Orchestrates document loading and chunking.

    Usage:
        pipeline = IngestionPipeline()
        chunks = pipeline.process_directory("./docs")
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_file(self, file_path: str | Path) -> list[dict]:
        """
        Load and chunk a single document.

        Returns:
            List of chunks with text and metadata
        """
        doc = self.loader.load(file_path)
        chunks = self.chunker.chunk_text(
            text=doc["text"],
            metadata=doc["metadata"]
        )
        return chunks

    def process_directory(self, dir_path: str | Path) -> list[dict]:
        """
        Load and chunk all documents in a directory.

        Returns:
            List of all chunks from all documents
        """
        documents = self.loader.load_directory(dir_path)

        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_text(
                text=doc["text"],
                metadata=doc["metadata"]
            )
            all_chunks.extend(chunks)

        print("\n📊 Ingestion complete:")
        print(f"   Documents: {len(documents)}")
        print(f"   Chunks: {len(all_chunks)}")
        if all_chunks:
            print(f"   Avg tokens/chunk: {sum(c['token_count'] for c in all_chunks) // len(all_chunks)}")

        return all_chunks


if __name__ == "__main__":
    pipeline = IngestionPipeline(chunk_size=200, chunk_overlap=30)
    chunks = pipeline.process_directory("sample_docs")

    print("\n--- Sample Chunks ---")
    for chunk in chunks[:3]:
        print(f"\nSource: {chunk['metadata']['source']}, Chunk: {chunk['metadata']['chunk_index']}")
        print(f"Tokens: {chunk['token_count']}")
        print(chunk['text'][:150] + "...")
