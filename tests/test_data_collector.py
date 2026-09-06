"""
Unit tests for Data Collection Module (Phase 2).
Tests downloading stock data via yfinance, cleaning, error handling, and return calculation.
"""

import pytest
import pandas as pd
from data_collector import get_stock_data


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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
