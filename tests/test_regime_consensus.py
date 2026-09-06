from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from adaptive_weights import calculate_adaptive_weights
from confidence_engine import calculate_decision_confidence
from consensus import compute_weighted_consensus
from regime_detection import detect_market_regime


def _results(scores):
    names = [
        "Market Forecasting Agent",
        "Technical Analysis Agent",
        "News Sentiment Agent",
        "ESG Analysis Agent",
        "Risk Assessment Agent",
    ]
    return [
        SimpleNamespace(
            agent_name=name,
            score=score,
            confidence=0.8,
            signal="BUY" if score > 0 else "SELL" if score < 0 else "HOLD",
            evidence=["evidence"],
            metadata={"risk_score": 30} if "Risk" in name else {},
            error=None,
        )
        for name, score in zip(names, scores)
    ]


def test_regime_detection_identifies_bull_trend():
    close = np.linspace(100, 125, 80)
    data = pd.DataFrame({"Close": close, "Daily_Return": pd.Series(close).pct_change().fillna(0)})
    result = detect_market_regime(data)
    assert result["regime"] == "BULL"
    assert 0.0 <= result["confidence"] <= 1.0


def test_adaptive_weights_are_normalized():
    weights = calculate_adaptive_weights("BEAR", base_weights={
        "forecast": 0.5, "technical": 0.5, "sentiment": 0.6, "esg": 0.75, "risk": 0.85
    })
    assert set(weights) == {"forecast", "technical", "sentiment", "esg", "risk"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["risk"] > weights["forecast"]


def test_consensus_reports_contributions_and_confidence_components():
    results = _results([0.8, 0.6, 0.0, 0.7, -0.2])
    output = compute_weighted_consensus(results, regime_context={"regime": "BULL", "confidence": 0.8})
    assert sum(row["weight"] for row in output["breakdown"]) == pytest.approx(1.0)
    assert sum(row["contribution"] for row in output["breakdown"]) == pytest.approx(output["consensus_score"])
    assert 0.0 <= output["confidence"] <= 1.0
    assert "agent_agreement" in output["confidence_details"]


def test_confidence_decreases_with_disagreement():
    results = _results([0.8, -0.8, 0.8, -0.8, 0.0])
    breakdown = [{"agent": result.agent_name, "weight": 0.2} for result in results]
    details = calculate_decision_confidence(results, breakdown, 0.0, 0.5)
    assert details["disagreement"] >= 0.4
    assert details["confidence"] < 0.6
