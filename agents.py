"""
========================================================================================
ESG-AWARE MULTI-AGENT FRAMEWORK: AGENTS ENTRY POINT (agents.py)
========================================================================================
This file serves as the primary master entry point containing all autonomous agent definitions
for the ESG-aware Multi-Agent RAG-LLM Framework based on the 7-step Workflow Diagram:

1. Autonomous Agents:
   - MarketForecastingAgent    : Price & trend prediction (LSTM / GRU / NeuralProphet)
   - TechnicalAnalysisAgent   : RSI, MACD, Bollinger Bands, Moving Average crossover signals
   - NewsSentimentAgent       : FinBERT / Financial news sentiment scoring
   - ESGAnalysisAgent         : Carbon, Renewables, Governance, ESG score computation
   - RiskAssessmentAgent      : Volatility, Drawdown, Beta, VaR, Sharpe ratio calculation
   - RAGKnowledgeAgent        : SentenceTransformers embedding & FAISS vector context retrieval

2. Decision Engine Agents:
   - WeightedConsensusEngine  : Dynamic weight consensus scoring engine
   - LLMDecisionAgent         : Llama 3 / FinGPT evidence-grounded recommendation generator

3. Framework Orchestrator:
   - MultiAgentOrchestrator   : End-to-end multi-agent workflow executor
========================================================================================
"""

from agents.market_forecasting_agent import MarketForecastingAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.esg_analysis_agent import ESGAnalysisAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.rag_knowledge_agent import RAGKnowledgeAgent
from agents.llm_decision_agent import LLMDecisionAgent
from agents.weighted_consensus_engine import WeightedConsensusEngine
from agents.orchestrator import MultiAgentOrchestrator
from agents import BaseAgent, AgentResult


# List of all available autonomous specialist agents
ALL_AUTONOMOUS_AGENTS = [
    MarketForecastingAgent,
    TechnicalAnalysisAgent,
    NewsSentimentAgent,
    ESGAnalysisAgent,
    RiskAssessmentAgent,
    RAGKnowledgeAgent
]

__all__ = [
    "BaseAgent",
    "AgentResult",
    "MarketForecastingAgent",
    "TechnicalAnalysisAgent",
    "NewsSentimentAgent",
    "ESGAnalysisAgent",
    "RiskAssessmentAgent",
    "RAGKnowledgeAgent",
    "LLMDecisionAgent",
    "WeightedConsensusEngine",
    "MultiAgentOrchestrator",
    "ALL_AUTONOMOUS_AGENTS"
]


if __name__ == "__main__":
    print("=========================================================================")
    print("ESG Multi-Agent System - All Agents Initialized Successfully")
    print("=========================================================================")
    print(f"Registered Autonomous Agents: {[agent.__name__ for agent in ALL_AUTONOMOUS_AGENTS]}")
    print(f"Decision Engine Agents      : [WeightedConsensusEngine, LLMDecisionAgent]")
    print(f"Orchestrator Engine         : MultiAgentOrchestrator")
    print("=========================================================================")
