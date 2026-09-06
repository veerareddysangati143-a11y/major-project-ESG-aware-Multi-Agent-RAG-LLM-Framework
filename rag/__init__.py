"""
RAG (Retrieval-Augmented Generation) Knowledge Pipeline Module.
Handles document collection, chunking, vector embedding, SQLite/FAISS vector store indexing, top-k retrieval, and LLM context augmentation.
"""

from typing import List, Dict, Any

import numpy as np
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
            chunks = chunk_document(doc)
            for chunk_index, chunk in enumerate(chunks, 1):
                db.add_document(
                    f"Context Doc #{idx + 1} Chunk {chunk_index}",
                    "User Feed",
                    chunk,
                    source="User-provided context",
                    document_type="Context document",
                )
            
    # Perform vector similarity search on SQLite vector database
    search_results = db.search_vector_db(query, top_k=top_k)
    top_similarity = max((item["similarity_score"] for item in search_results), default=-1.0)
    relevance = float(np.clip((top_similarity + 1.0) / 2.0, 0.0, 1.0))
    coverage = float(np.clip(len(search_results) / max(top_k, 1), 0.0, 1.0))
    evidence_quality = float(np.clip(0.6 * relevance + 0.4 * coverage, 0.0, 1.0))
    evidence_status = "Sufficient evidence" if evidence_quality >= 0.35 else "Insufficient evidence"
    retrieved_chunks = [
        f"[{item['document_type']}] {item['title']} | Source: {item['source']} | "
        f"Date: {item['publication_date'] or 'Not provided'} | Relevance: {item['similarity_score']:.3f}\n"
        f"Evidence: {item['text']}"
        for item in search_results
    ]
    
    augmented_context = "\n---\n".join(retrieved_chunks)
    
    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "top_k": top_k,
        "vector_store": "SQLite Vector Store (data/stock_intelligence.db)",
        "context_augmented": augmented_context,
        "search_details": search_results,
        "evidence_quality": evidence_quality,
        "evidence_status": evidence_status,
    }
