"""
Unit tests for Data Collection Module (Phase 2).
Tests downloading stock data via yfinance, cleaning, error handling, and return calculation.
"""

import pytest
import pandas as pd
from data_collector import get_stock_data, get_latest_market_data


def test_get_stock_data_valid_ticker():
    """Tests successful data retrieval for a valid ticker (AAPL)."""
    df = get_stock_data("AAPL", "2023-01-01", "2023-03-01", save_raw=False)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 10
    
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Daily_Return"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
        
    # Check no duplicate dates
    assert df["Date"].duplicated().sum() == 0
    
    # Check chronological ordering
    assert df["Date"].is_monotonic_increasing


def test_get_stock_data_invalid_ticker():
    """Tests that an invalid ticker raises ValueError or RuntimeError gracefully."""
    with pytest.raises((ValueError, RuntimeError)):
        get_stock_data("INVALID_TICKER_XYZ_9999", "2023-01-01", "2023-02-01", save_raw=False)


def test_get_stock_data_invalid_dates():
    """Tests that start_date after end_date raises ValueError."""
    with pytest.raises(ValueError):
        get_stock_data("AAPL", "2023-05-01", "2023-01-01", save_raw=False)


def test_latest_market_data_calculates_observed_change(monkeypatch):
    import data_collector

    dates = pd.date_range("2026-09-03", periods=2, freq="D")
    history = pd.DataFrame({"Date": dates, "Close": [100.0, 105.0]})

    class FakeTicker:
        def history(self, **kwargs):
            return history.set_index("Date")

    monkeypatch.setattr(data_collector.yf, "Ticker", lambda ticker: FakeTicker())
    result = get_latest_market_data("RELIANCE.NS")

    assert result["ok"] is True
    assert result["latest_observed_price"] == 105.0
    assert result["previous_close"] == 100.0
    assert result["price_change"] == 5.0
    assert result["price_change_percent"] == 5.0


def test_latest_market_data_reports_provider_failure(monkeypatch):
    import data_collector

    class BrokenTicker:
        def history(self, **kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(data_collector.yf, "Ticker", lambda ticker: BrokenTicker())
    result = get_latest_market_data("RELIANCE.NS")

    assert result["ok"] is False
    assert result["data_status"] == "API unavailable"
    assert result["latest_observed_price"] is None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
