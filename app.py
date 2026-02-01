"""
Gradio UI for RAG Documentation Assistant.

A clean, demo-ready interface for asking questions about your documents.
"""

import gradio as gr
from pathlib import Path
from src.query.rag import RAGPipeline
from src.ingestion.pipeline import IngestionPipeline


# Initialize pipelines (global so we don't reload on every query)
print("Initializing RAG system...")
rag = RAGPipeline(
    collection_name="documents",
    persist_dir="./data/chroma_ui",
    top_k=5
)
ingestion = IngestionPipeline(chunk_size=500, chunk_overlap=50)
print("Ready!\n")


def ask_question(question: str, history: list) -> tuple:
    """
    Process a question through the RAG pipeline.
    """
    if not question.strip():
        return history, ""
    
    # Check if we have documents
    if rag.vector_store.count() == 0:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "⚠️ No documents loaded yet. Please upload some documents first using the 'Upload Documents' tab."})
        return history, ""
    
    # Query the RAG system
    result = rag.query(question)
    
    # Format the response with sources
    answer = result["answer"]
    
    # Add source information
    if result["sources"]:
        sources = list(set(src["source"] for src in result["sources"]))
        answer += f"\n\n---\n📚 **Sources:** {', '.join(sources)}"
    
    # Add token usage
    usage = result["usage"]
    answer += f"\n\n🔢 *Tokens: {usage['input_tokens']} in / {usage['output_tokens']} out*"
    
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, ""


def upload_files(files) -> str:
    """
    Process uploaded files and add them to the knowledge base.
    """
    if not files:
        return "No files selected."
    
    results = []
    all_chunks = []
    
    for file in files:
        try:
            chunks = ingestion.process_file(file.name)
            all_chunks.extend(chunks)
            results.append(f"✅ {Path(file.name).name}: {len(chunks)} chunks")
        except Exception as e:
            results.append(f"❌ {Path(file.name).name}: {str(e)}")
    
    if all_chunks:
        rag.ingest(all_chunks)
        results.append(f"\n📊 **Total:** {len(all_chunks)} chunks added to knowledge base")
        results.append(f"📁 **Documents in store:** {rag.vector_store.count()}")
    
    return "\n".join(results)


def clear_knowledge_base() -> str:
    """Clear all documents from the knowledge base."""
    rag.vector_store.clear()
    return "🗑️ Knowledge base cleared. Upload new documents to get started."


def get_status() -> str:
    """Get current status of the knowledge base."""
    count = rag.vector_store.count()
    if count == 0:
        return "📭 Knowledge base is empty. Upload documents to get started."
    return f"📚 Knowledge base contains **{count}** document chunks."


# Build the Gradio interface
with gr.Blocks(title="RAG Documentation Assistant") as app:
    
    gr.Markdown(
        """
        # 📚 RAG Documentation Assistant
        
        Ask questions about your documents and get accurate, cited answers powered by Claude.
        
        **How it works:** Upload documents → Ask questions → Get answers with source citations
        """
    )
    
    with gr.Tabs():
        # Tab 1: Chat Interface
        with gr.TabItem("💬 Ask Questions"):
            chatbot = gr.Chatbot(height=450)
            
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    show_label=False,
                    scale=9
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)
            
            status_display = gr.Markdown(value=get_status)
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
            
            submit_btn.click(
                fn=ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            question_input.submit(
                fn=ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            refresh_btn.click(fn=get_status, outputs=status_display)
        
        # Tab 2: Document Upload
        with gr.TabItem("📁 Upload Documents"):
            gr.Markdown(
                """
                ### Upload your documents
                
                Supported formats: **PDF**, **DOCX**, **TXT**, **Markdown**
                """
            )
            
            file_upload = gr.File(
                label="Select files",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt", ".md"]
            )
            
            upload_btn = gr.Button("📤 Upload & Process", variant="primary")
            upload_output = gr.Markdown(label="Upload Results")
            
            upload_btn.click(
                fn=upload_files,
                inputs=file_upload,
                outputs=upload_output
            )
            
            gr.Markdown("---")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Knowledge Base", variant="stop")
                clear_output = gr.Markdown()
            
            clear_btn.click(fn=clear_knowledge_base, outputs=clear_output)
        
        # Tab 3: About
        with gr.TabItem("ℹ️ About"):
            gr.Markdown(
                """
                ### How This Works
                
                This is a **Retrieval-Augmented Generation (RAG)** system that combines:
                
                1. **Document Processing** — Files are split into chunks with overlapping context
                2. **Semantic Embeddings** — Each chunk becomes a vector capturing its meaning
                3. **Vector Search** — Your question finds the most relevant chunks
                4. **LLM Generation** — Claude generates a cited answer
                
                ### Tech Stack
                
                - **LLM:** Claude Sonnet (Anthropic)
                - **Embeddings:** all-MiniLM-L6-v2
                - **Vector Store:** ChromaDB
                - **UI:** Gradio
                
                ---
                
                [GitHub](https://github.com/woodstocksoftware/rag-documentation-assistant) | 
                Built by [Jim Williams](https://www.linkedin.com/in/woodstocksoftware/)
                """
            )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)