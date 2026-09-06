"""
Risk Assessment Agent (Volatility, Drawdown, Beta, VaR, Sharpe Ratio).
Evaluates downside risk, Sharpe ratio, and risk score.
"""

from typing import Dict, Any
from agents import BaseAgent, AgentResult
from utils.constants import AgentName
from utils.helpers import numeric_to_signal
from risk import compute_risk_metrics


class RiskAssessmentAgent(BaseAgent):
    """Autonomous agent for financial risk assessment and drawdown monitoring."""
    
    def __init__(self):
        super().__init__(AgentName.RISK)

    def analyze(self, context: Dict[str, Any]) -> AgentResult:
        data = context["data"]
        metrics = compute_risk_metrics(data)
        score = metrics["agent_score"]
        signal = numeric_to_signal(score)
        
        return AgentResult(
            agent_name=self.name,
            signal=signal,
            score=score,
            confidence=0.85,
            evidence=metrics["evidence"],
            metadata={
                "volatility": metrics["volatility"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "var_95": metrics["var_95"],
                "beta": metrics["beta"],
                "risk_score": metrics["risk_score"]
            }
        )
