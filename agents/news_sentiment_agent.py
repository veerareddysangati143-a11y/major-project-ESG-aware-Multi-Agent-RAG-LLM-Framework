"""
News Sentiment Agent (FinBERT / Sentiment Analysis).
Analyzes sentiment in financial news articles.
"""

from typing import Dict, Any
from agents import BaseAgent, AgentResult
from utils.constants import AgentName
from utils.helpers import numeric_to_signal
from sentiment import analyze_news_sentiment


class NewsSentimentAgent(BaseAgent):
    """Autonomous agent for news sentiment analysis using FinBERT models."""
    
    def __init__(self):
        super().__init__(AgentName.SENTIMENT)

    def analyze(self, context: Dict[str, Any]) -> AgentResult:
        headlines = context.get("headlines", [])
        result = analyze_news_sentiment(headlines)
        score = result["score"]
        signal = numeric_to_signal(score)
        
        return AgentResult(
            agent_name=self.name,
            signal=signal,
            score=score,
            confidence=result["confidence"],
            evidence=result["evidence"],
            metadata={
                "headline_count": result["headline_count"],
                "breakdown": result["breakdown"],
                "model": "FinBERT / Keyword Engine"
            }
        )
