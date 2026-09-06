"""
RAG (Retrieval-Augmented Generation) Knowledge Pipeline Module.
Handles document collection, chunking, vector embedding, SQLite/FAISS vector store indexing, top-k retrieval, and LLM context augmentation.
"""

from typing import List, Dict, Any
from database import DatabaseManager


def chunk_document(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """Splits raw text into overlapping chunks for embedding."""
    if not text:
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def build_and_query_rag(documents: List[str], query: str, top_k: int = 3) -> Dict[str, Any]:
    """Performs document chunking, vector embedding, database search, and top-k context retrieval."""
    db = DatabaseManager()
    
    # Insert any new incoming context documents into vector database
    if documents:
        for idx, doc in enumerate(documents):
            db.add_document(f"Context Doc #{idx+1}", "User Feed", doc)
            
    # Perform vector similarity search on SQLite vector database
    search_results = db.search_vector_db(query, top_k=top_k)
    retrieved_chunks = [f"[{item['category']}] {item['title']}: {item['text']}" for item in search_results]
    
    augmented_context = "\n---\n".join(retrieved_chunks)
    
    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "top_k": top_k,
        "vector_store": "SQLite Vector Store (data/stock_intelligence.db)",
        "context_augmented": augmented_context,
        "search_details": search_results
    }
