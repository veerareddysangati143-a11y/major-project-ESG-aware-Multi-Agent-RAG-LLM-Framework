"""
Constants and Enumerations for ESG Multi-Agent System.
"""

from enum import Enum


class SignalType(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNAVAILABLE = "UNAVAILABLE"


class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class AgentName(str, Enum):
    FORECASTING = "Market Forecasting Agent"
    TECHNICAL = "Technical Analysis Agent"
    SENTIMENT = "News Sentiment Agent"
    ESG = "ESG Analysis Agent"
    RISK = "Risk Assessment Agent"
    RAG = "RAG Knowledge Agent"
    REGIME = "Market Regime Agent"
    CONSENSUS = "Dynamic Consensus Engine"
    LLM_DECISION = "LLM Decision Agent"
