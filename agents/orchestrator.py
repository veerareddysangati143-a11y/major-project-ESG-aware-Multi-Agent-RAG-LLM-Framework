"""
Multi-Agent Orchestrator.
Orchestrates all autonomous specialist agents, the Weighted Consensus Engine, and the LLM Decision Agent.
"""

from typing import Any, Dict, List, Optional
from agents import AgentResult
from agents.market_forecasting_agent import MarketForecastingAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.esg_analysis_agent import ESGAnalysisAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.rag_knowledge_agent import RAGKnowledgeAgent
from agents.llm_decision_agent import LLMDecisionAgent
from agents.weighted_consensus_engine import WeightedConsensusEngine
from agents.specialists import RegimeAgent
from utils.constants import AgentName


class MultiAgentOrchestrator:
    """Orchestrates all autonomous specialist agents end-to-end according to the workflow diagram."""

    def __init__(self):
        self.market_forecasting_agent = MarketForecastingAgent()
        self.technical_analysis_agent = TechnicalAnalysisAgent()
        self.news_sentiment_agent = NewsSentimentAgent()
        self.esg_analysis_agent = ESGAnalysisAgent()
        self.risk_assessment_agent = RiskAssessmentAgent()
        self.rag_knowledge_agent = RAGKnowledgeAgent()
        self.regime_agent = RegimeAgent()
        
        self.consensus_engine = WeightedConsensusEngine()
        self.llm_decision_agent = LLMDecisionAgent()

        self.specialists = [
            self.market_forecasting_agent,
            self.technical_analysis_agent,
            self.news_sentiment_agent,
            self.esg_analysis_agent,
            self.risk_assessment_agent,
            self.rag_knowledge_agent,
            self.regime_agent
        ]

    def run(self, context: Dict[str, Any], custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Runs the entire 7-step multi-agent RAG-LLM framework."""
        # 1. Execute all specialist agents
        results = [agent.run(context) for agent in self.specialists]
        
        # 2. Extract RAG context
        rag_res = next((r for r in results if r.agent_name == AgentName.RAG.value), None)
        rag_context = "\n".join(rag_res.evidence) if rag_res else ""
        regime_res = next((r for r in results if r.agent_name == AgentName.REGIME.value), None)
        
        # 3. Decision Engine: Weighted Consensus Engine
        consensus_context = {
            "results": results,
            "weights": custom_weights,
            "regime": regime_res.metadata if regime_res else {}
        }
        consensus_result = self.consensus_engine.run(consensus_context)
        
        # 4. Decision Engine: LLM Decision Agent
        llm_context = {
            "results": results,
            "consensus_data": {
                "recommendation": consensus_result.metadata.get("recommendation", "HOLD"),
                "consensus_score": consensus_result.score,
                "confidence": consensus_result.confidence,
                "confidence_details": consensus_result.metadata.get("confidence_details", {}),
                "regime": consensus_result.metadata.get("regime", {})
            },
            "rag_context": rag_context
        }
        explanation_result = self.llm_decision_agent.run(llm_context)

        return {
            "results": results,
            "consensus": consensus_result,
            "explanation": explanation_result
        }
