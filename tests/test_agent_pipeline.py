"""Offline regression test for the multi-agent stock pipeline."""

import numpy as np

from agents.orchestrator import MultiAgentOrchestrator
from data_collector import get_stock_data


def _analysis_data():
    data = get_stock_data("RELIANCE.NS", "2015-01-01", "2024-01-01", save_raw=False)
    close = data["Close"]
    data["SMA 20"] = close.rolling(20).mean()
    data["SMA 50"] = close.rolling(50).mean()
    data["EMA 12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA 26"] = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA 12"] - data["EMA 26"]
    gains = close.diff().clip(lower=0).rolling(14).mean()
    losses = -close.diff().clip(upper=0).rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + gains / losses.replace(0, np.nan)))
    return data.dropna(subset=["SMA 50", "RSI"]).reset_index(drop=True)


def test_all_specialist_agents_return_results():
    pipeline = MultiAgentOrchestrator().run({
        "data": _analysis_data(),
        "ticker": "RELIANCE.NS",
        "documents": [],
    })

    assert len(pipeline["results"]) == 7
    assert all(result.error is None for result in pipeline["results"])
    assert pipeline["consensus"].error is None
    assert pipeline["explanation"].error is None
