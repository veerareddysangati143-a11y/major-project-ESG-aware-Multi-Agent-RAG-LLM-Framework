"""
ESG Analysis Agent (Carbon, Renewables, Governance, ESG Score).
Evaluates sustainability & ESG risk factors.
"""

from typing import Dict, Any
from agents import BaseAgent, AgentResult
from utils.constants import AgentName, SignalType
from esg import evaluate_esg_factors


class ESGAnalysisAgent(BaseAgent):
    """Autonomous agent for ESG factors analysis and scoring."""
    
    def __init__(self):
        super().__init__(AgentName.ESG)

    def analyze(self, context: Dict[str, Any]) -> AgentResult:
        documents = context.get("documents", [])
        result = evaluate_esg_factors(documents, context.get("ticker", "RELIANCE.NS"))
        
        signal_map = {"BUY": SignalType.BUY, "HOLD": SignalType.HOLD, "SELL": SignalType.SELL}
        signal = signal_map.get(result["signal"], SignalType.HOLD)
        
        return AgentResult(
            agent_name=self.name,
            signal=signal,
            score=result["normalized_score"],
            confidence=0.65 if documents else 0.40,
            evidence=result["evidence"],
            metadata={
                "overall_esg_score": result["overall_score"],
                "category_scores": result["category_scores"],
                "document_count": result["document_count"],
                "esg_record_count": result["esg_record_count"],
                "as_of_date": result["as_of_date"],
                "source": result["source"],
                "source_evidence": result["source_evidence"],
                "analysis": "Weighted ESG table scores are combined with document evidence; ESG is an active decision input."
            }
        )
