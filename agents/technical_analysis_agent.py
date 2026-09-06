"""
Technical Analysis Agent (RSI, MACD, Bollinger Bands).
Generates technical trading signals.
"""

from typing import Dict, Any
from agents import BaseAgent, AgentResult
from utils.constants import AgentName
from utils.helpers import numeric_to_signal
from technical import analyze_technical_signals


class TechnicalAnalysisAgent(BaseAgent):
    """Autonomous agent for technical analysis using RSI, MACD, SMA."""
    
    def __init__(self):
        super().__init__(AgentName.TECHNICAL)

    def analyze(self, context: Dict[str, Any]) -> AgentResult:
        data = context["data"]
        result = analyze_technical_signals(data)
        score = result["score"]
        signal = numeric_to_signal(score)
        
        return AgentResult(
            agent_name=self.name,
            signal=signal,
            score=score,
            confidence=0.80,
            evidence=result["reasons"],
            metadata={
                "rsi": result["rsi"],
                "macd": result["macd"],
                "sma_20": result["sma_20"],
                "sma_50": result["sma_50"]
            }
        )
