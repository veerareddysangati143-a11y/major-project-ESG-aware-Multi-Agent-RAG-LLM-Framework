"""
Base Agent Module & Package Exports for ESG Multi-Agent Stock Analysis System.
Defines standard AgentResult, BaseAgent interface, and re-exports all autonomous agents.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from utils.constants import AgentName, SignalType
from utils.helpers import convert_numpy_types
from utils.logger import setup_logger


@dataclass
class AgentResult:
    """Standardized output schema for all agents in the multi-agent system."""
    agent_name: str
    signal: Union[SignalType, str] = SignalType.NEUTRAL
    score: float = 0.0               # Range: -1.0 to +1.0 (or 0-100 for ESG/Risk)
    confidence: float = 0.0          # Range: 0.0 to 1.0
    evidence: Union[List[Any], Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        sig_str = self.signal.value if isinstance(self.signal, SignalType) else str(self.signal)
        raw_dict = {
            "agent": self.agent_name,
            "signal": sig_str,
            "score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "evidence": self.evidence,
            "metadata": self.metadata,
            "execution_time_ms": round(float(self.execution_time_ms), 2),
            "error": self.error
        }
        return convert_numpy_types(raw_dict)


class BaseAgent(ABC):
    """Abstract Base Class for all autonomous agents."""

    def __init__(self, name: Union[AgentName, str]):
        self.name = name.value if isinstance(name, AgentName) else str(name)
        self.logger = setup_logger(self.name)

    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError("Each agent must implement the analyze() method.")

    def run(self, context: Dict[str, Any]) -> AgentResult:
        self.logger.info(f"Executing agent: {self.name}...")
        start_time = time.time()
        
        try:
            result = self.analyze(context)
            elapsed = (time.time() - start_time) * 1000
            result.execution_time_ms = elapsed
            self.logger.info(
                f"Agent {self.name} completed in {elapsed:.2f}ms | "
                f"Signal: {result.signal} | Confidence: {result.confidence:.2f}"
            )
            return result
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.logger.error(f"Error executing agent {self.name}: {str(e)}", exc_info=True)
            
            return AgentResult(
                agent_name=self.name,
                signal=SignalType.UNAVAILABLE,
                score=0.0,
                confidence=0.0,
                evidence=[],
                metadata={"status": "failed"},
                execution_time_ms=elapsed,
                error=str(e)
            )


# Import concrete specialist agents for package export
from agents.market_forecasting_agent import MarketForecastingAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.esg_analysis_agent import ESGAnalysisAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.rag_knowledge_agent import RAGKnowledgeAgent
from agents.llm_decision_agent import LLMDecisionAgent
from agents.weighted_consensus_engine import WeightedConsensusEngine
from agents.orchestrator import MultiAgentOrchestrator

__all__ = [
    "AgentResult",
    "BaseAgent",
    "MarketForecastingAgent",
    "TechnicalAnalysisAgent",
    "NewsSentimentAgent",
    "ESGAnalysisAgent",
    "RiskAssessmentAgent",
    "RAGKnowledgeAgent",
    "LLMDecisionAgent",
    "WeightedConsensusEngine",
    "MultiAgentOrchestrator"
]
