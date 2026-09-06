"""
RAG Knowledge Agent (Document collection, Chunking, Embedding, FAISS Indexing & Retrieval).
Retrieves top-k relevant context chunks for LLM augmentation.
"""

from typing import Dict, Any
from agents import BaseAgent, AgentResult
from utils.constants import AgentName, SignalType
from rag import build_and_query_rag


class RAGKnowledgeAgent(BaseAgent):
    """Autonomous agent for RAG Knowledge retrieval using FAISS vector store."""
    
    def __init__(self):
        super().__init__(AgentName.RAG)

    def analyze(self, context: Dict[str, Any]) -> AgentResult:
        documents = context.get("documents", [])
        ticker = context.get("ticker", "Stock")
        query = f"ESG sustainability, regulatory compliance, and risk for {ticker}"
        
        result = build_and_query_rag(documents, query, top_k=3)
        retrieved = result["retrieved_chunks"]
        
        evidence = retrieved if retrieved else ["No local ESG or filing documents were indexed."]
        
        return AgentResult(
            agent_name=self.name,
            signal=SignalType.NEUTRAL,
            score=0.0,
            confidence=0.50 if documents else 0.25,
            evidence=evidence,
            metadata={
                "vector_store": result["vector_store"],
                "retrieved_chunks_count": len(retrieved),
                "top_k": result["top_k"],
                "query": query,
                "analysis": "Retrieved evidence is supplied to the decision agent for grounded reasoning."
            }
        )
