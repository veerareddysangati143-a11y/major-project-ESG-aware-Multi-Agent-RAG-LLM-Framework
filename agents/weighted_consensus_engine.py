"""
Weighted Consensus Engine Agent.
Combines agent outputs with dynamic weights (Forecast 80%, Sentiment 70%, Risk 60%, ESG 90%) into Final Confidence Score.
"""

from typing import Dict, Any, List
from agents import BaseAgent, AgentResult
from utils.constants import AgentName, SignalType
from consensus import compute_weighted_consensus


class WeightedConsensusEngine(BaseAgent):
    """Engine agent for combining specialist outputs with dynamic dynamic weighting."""
    
    def __init__(self):
        super().__init__(AgentName.CONSENSUS)

    def analyze(self, context: Dict[str, Any]) -> AgentResult:
        results = context.get("results", [])
        custom_weights = context.get("weights", None)
        
        consensus_output = compute_weighted_consensus(
            results,
            custom_weights,
            context.get("regime", {}),
        )
        rec_str = consensus_output["recommendation"]
        signal = SignalType.BUY if rec_str == "BUY" else SignalType.SELL if rec_str == "SELL" else SignalType.HOLD
        
        evidence = [
            f"Consensus Recommendation: {rec_str}",
            f"Final Confidence Score: {consensus_output['confidence']:.0%}",
            f"Weighted Composite Score: {consensus_output['consensus_score']:+.3f}",
            f"Agents Evaluated: {consensus_output['total_agents']}"
        ]
        
        return AgentResult(
            agent_name=self.name,
            signal=signal,
            score=consensus_output["consensus_score"],
            confidence=consensus_output["confidence"],
            evidence=evidence,
            metadata={
                "recommendation": rec_str,
                "breakdown": consensus_output["breakdown"],
                "custom_weights": custom_weights,
                "normalized_weights": consensus_output["normalized_weights"],
                "confidence_details": consensus_output["confidence_details"],
                "regime": consensus_output["regime"],
            }
        )
