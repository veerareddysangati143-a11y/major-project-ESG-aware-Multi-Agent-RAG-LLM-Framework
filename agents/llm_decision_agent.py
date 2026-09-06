"""
LLM Decision Agent (Llama 3 / FinGPT).
Generates final investment decision with rationale.
"""

from typing import Dict, Any
from agents import BaseAgent, AgentResult
from utils.constants import AgentName, SignalType
from llm import generate_llm_decision


class LLMDecisionAgent(BaseAgent):
    """Autonomous LLM Decision Agent generating evidence-grounded final recommendations."""
    
    def __init__(self):
        super().__init__(AgentName.LLM_DECISION)

    def analyze(self, context: Dict[str, Any]) -> AgentResult:
        results = context.get("results", [])
        consensus_data = context.get("consensus_data", {"recommendation": "HOLD", "consensus_score": 0.0, "confidence": 0.7})
        rag_context = context.get("rag_context", "")
        
        output = generate_llm_decision(consensus_data, results, rag_context)
        rec_str = output["recommendation"]
        signal = SignalType.BUY if rec_str == "BUY" else SignalType.SELL if rec_str == "SELL" else SignalType.HOLD
        
        return AgentResult(
            agent_name=self.name,
            signal=signal,
            score=consensus_data.get("consensus_score", 0.0),
            confidence=consensus_data.get("confidence", 0.75),
            evidence=[output["rationale"]],
            metadata={
                "provider": output["provider"],
                "model_version": output["model_version"],
                "evidence_grounded": output["evidence_grounded"],
                "esg_aligned": output["esg_aligned"]
            }
        )
