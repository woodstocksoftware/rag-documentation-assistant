"""
Document loading for RAG pipeline.

Extracts text from various file formats (PDF, TXT, Markdown, DOCX).
"""

from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument


class DocumentLoader:
    """
    Loads documents from various file formats and extracts text.
    
    Supported formats: .txt, .md, .pdf, .docx
    """
    
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
    
    def load(self, file_path: str | Path) -> dict:
        """
        Load a document and extract its text.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dict with keys: text, metadata (source, pages, file_type)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        
        # Route to appropriate loader
        if extension == ".pdf":
            text, page_count = self._load_pdf(path)
        elif extension == ".docx":
            text, page_count = self._load_docx(path)
        else:  # .txt, .md
            text, page_count = self._load_text(path)
        
        return {
            "text": text,
            "metadata": {
                "source": path.name,
                "file_path": str(path.absolute()),
                "file_type": extension,
                "page_count": page_count
            }
        }
    
    def load_directory(self, dir_path: str | Path) -> list[dict]:
        """
        Load all supported documents from a directory.
        
        Args:
            dir_path: Path to directory containing documents
            
        Returns:
            List of document dicts
        """
        path = Path(dir_path)
        
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        documents = []
        
        for file_path in path.iterdir():
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load(file_path)
                    documents.append(doc)
                    print(f"✓ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"✗ Failed to load {file_path.name}: {e}")
        
        return documents
    
    def _load_pdf(self, path: Path) -> tuple[str, int]:
        """Extract text from PDF."""
        reader = PdfReader(path)
        
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        
        return "\n\n".join(pages), len(reader.pages)
    
    def _load_docx(self, path: Path) -> tuple[str, int]:
        """Extract text from Word document."""
        doc = DocxDocument(path)
        
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        
        # DOCX doesn't have explicit pages, estimate based on content
        return "\n\n".join(paragraphs), 1
    
    def _load_text(self, path: Path) -> tuple[str, int]:
        """Load plain text or markdown file."""
        text = path.read_text(encoding="utf-8")
        return text, 1


# Quick test
if __name__ == "__main__":
    loader = DocumentLoader()
    
    # Create a sample document for testing
    sample_path = Path("sample_docs/test.txt")
    sample_path.parent.mkdir(exist_ok=True)
    sample_path.write_text("""
# RAG Architecture Overview

Retrieval-Augmented Generation (RAG) combines the power of large language models 
with external knowledge retrieval. This approach addresses key limitations of 
standalone LLMs.

## Key Components

1. Document Store: Holds your knowledge base
2. Embedding Model: Converts text to vectors
3. Vector Database: Enables similarity search
4. LLM: Generates final response

## Benefits

- Reduces hallucinations by grounding responses in real documents
- Enables access to private or up-to-date information
- Provides citations and source attribution
- More cost-effective than fine-tuning
""")
    
    # Test loading
    doc = loader.load(sample_path)
    print(f"Loaded: {doc['metadata']['source']}")
    print(f"Type: {doc['metadata']['file_type']}")
    print(f"Text length: {len(doc['text'])} characters")
    print(f"\nPreview:\n{doc['text'][:300]}...")