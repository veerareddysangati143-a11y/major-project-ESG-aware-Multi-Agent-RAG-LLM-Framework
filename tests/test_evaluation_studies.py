import numpy as np
import pandas as pd

from evaluation import compute_ablation_study, compute_backtest_results, compute_model_comparison


def _data(rows=140):
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    close = 100 + np.cumsum(np.full(rows, 0.15)) + np.sin(np.arange(rows))
    return pd.DataFrame({
        "Date": dates,
        "Close": close,
        "Daily_Return": pd.Series(close).pct_change().fillna(0.0),
    })


def test_model_comparison_uses_runtime_and_unavailable_labels():
    result = compute_model_comparison(_data())
    rows = {row["Model"]: row for row in result["rows"]}
    assert rows["Proposed/Ensemble"]["MAE"] is not None
    assert rows["LSTM"]["MAE"] is None
    assert "Not available" in rows["Transformer"]["Source"]


def test_backtest_returns_all_strategies_and_metrics():
    result = compute_backtest_results(_data())
    assert len(result["rows"]) == 5
    assert all("Cumulative_Return" in row for row in result["rows"])
    assert result["rows"][0]["Strategy"] == "Buy and Hold"


def test_ablation_marks_unavailable_historical_inputs_honestly():
    result = compute_ablation_study(_data())
    statuses = {row["Configuration"]: row["Status"] for row in result["rows"]}
    assert statuses["Forecast-only"] == "Evaluated"
    assert statuses["+ ESG"] == "Not yet evaluated"
    assert statuses["+ RAG"] == "Not yet evaluated"
