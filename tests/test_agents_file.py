"""
Unit tests to verify agents.py exports, individual agent outputs, and consensus calculation.
"""

import pytest
import pandas as pd
import numpy as np

from agents import (
    MarketForecastingAgent,
    TechnicalAnalysisAgent,
    NewsSentimentAgent,
    ESGAnalysisAgent,
    RiskAssessmentAgent,
    RAGKnowledgeAgent,
    LLMDecisionAgent,
    WeightedConsensusEngine,
    MultiAgentOrchestrator
)
import agents


def _sample_data():
    dates = pd.date_range("2024-01-01", periods=100)
    prices = np.linspace(100, 150, 100) + np.random.normal(0, 1, 100)
    df = pd.DataFrame({
        "Date": dates,
        "Open": prices - 1,
        "High": prices + 2,
        "Low": prices - 2,
        "Close": prices,
        "Volume": 1000000,
        "Daily_Return": pd.Series(prices).pct_change()
    })
    df["SMA 20"] = df["Close"].rolling(20).mean()
    df["SMA 50"] = df["Close"].rolling(50).mean()
    df["EMA 12"] = df["Close"].ewm(span=12).mean()
    df["EMA 26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA 12"] - df["EMA 26"]
    df["RSI"] = 55.0
    return df.dropna().reset_index(drop=True)


def test_agents_module_exports():
    assert hasattr(agents, "MarketForecastingAgent")
    assert hasattr(agents, "TechnicalAnalysisAgent")
    assert hasattr(agents, "NewsSentimentAgent")
    assert hasattr(agents, "ESGAnalysisAgent")
    assert hasattr(agents, "RiskAssessmentAgent")
    assert hasattr(agents, "RAGKnowledgeAgent")
    assert hasattr(agents, "LLMDecisionAgent")
    assert hasattr(agents, "WeightedConsensusEngine")
    assert hasattr(agents, "MultiAgentOrchestrator")


def test_individual_agents_execution():
    data = _sample_data()
    ctx = {
        "data": data,
        "ticker": "AAPL",
        "headlines": ["Company reports strong earnings growth"],
        "documents": ["Sustainability report: reduced emissions by 20%"]
    }

    forecast = MarketForecastingAgent().run(ctx)
    assert forecast.error is None
    assert "Forecasting" in forecast.agent_name

    tech = TechnicalAnalysisAgent().run(ctx)
    assert tech.error is None

    sent = NewsSentimentAgent().run(ctx)
    assert sent.error is None

    esg = ESGAnalysisAgent().run(ctx)
    assert esg.error is None

    risk = RiskAssessmentAgent().run(ctx)
    assert risk.error is None

    rag = RAGKnowledgeAgent().run(ctx)
    assert rag.error is None


def test_orchestrator_end_to_end():
    orchestrator = MultiAgentOrchestrator()
    data = _sample_data()
    pipeline = orchestrator.run({
        "data": data,
        "ticker": "RELIANCE.NS",
        "headlines": ["Positive market outlook"],
        "documents": ["ESG compliance score 80"]
    })

    assert len(pipeline["results"]) >= 6
    assert pipeline["consensus"].error is None
    assert pipeline["explanation"].error is None
