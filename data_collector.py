"""
Stock Data Collection Module using Yahoo Finance (yfinance) and Local CSV Datasets.
Downloads, validates, cleans, calculates daily returns, and caches stock data.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf

from config import RAW_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("data_collector")


def get_latest_market_data(ticker: str) -> dict:
    """Return the latest available observed Yahoo Finance price for a ticker.

    This deliberately uses a short recent window and is separate from the
    historical analysis loader. Daily history may be delayed or closed-market
    data, so the response labels it as latest available rather than live.
    """
    ticker = ticker.strip().upper()
    try:
        history = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        if history.empty:
            raise ValueError("Yahoo Finance returned no recent observations")

        history = history.reset_index()
        date_column = "Datetime" if "Datetime" in history.columns else "Date"
        history[date_column] = pd.to_datetime(history[date_column], utc=True).dt.tz_localize(None)
        history = history.dropna(subset=["Close"]).sort_values(date_column).reset_index(drop=True)
        if history.empty:
            raise ValueError("Recent history contained no valid close values")

        latest = history.iloc[-1]
        previous = history.iloc[-2] if len(history) > 1 else None
        latest_price = float(latest["Close"])
        previous_close = float(previous["Close"]) if previous is not None else latest_price
        price_change = latest_price - previous_close
        price_change_percent = (price_change / previous_close * 100.0) if previous_close else 0.0
        timestamp = pd.Timestamp(latest[date_column])

        return {
            "ok": True,
            "ticker": ticker,
            "latest_observed_price": latest_price,
            "previous_close": previous_close,
            "price_change": price_change,
            "price_change_percent": price_change_percent,
            "latest_price_timestamp": timestamp.isoformat(),
            "data_date": timestamp.date().isoformat(),
            "latest_price_source": "Yahoo Finance",
            "market_status": "Latest available market data",
            "data_status": "Latest available",
            "error": None,
        }
    except Exception as error:
        logger.warning(f"Latest market data unavailable for {ticker}: {error}")
        return {
            "ok": False,
            "ticker": ticker,
            "latest_observed_price": None,
            "previous_close": None,
            "price_change": None,
            "price_change_percent": None,
            "latest_price_timestamp": None,
            "data_date": None,
            "latest_price_source": "Yahoo Finance",
            "market_status": "Unavailable",
            "data_status": "API unavailable",
            "error": str(error),
        }


def _load_local_candidate(path: Path) -> pd.DataFrame:
    """Read one local CSV and normalize its date column for range selection."""
    local_df = pd.read_csv(path)

    if "Price" in local_df.columns and not local_df.empty and local_df.iloc[0]["Price"] == "Date":
        local_df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        local_df = local_df.iloc[1:].reset_index(drop=True)

    date_col = [c for c in local_df.columns if "date" in str(c).lower() or "price" in str(c).lower()]
    if not date_col:
        raise ValueError("No date column found")

    local_df = local_df.rename(columns={date_col[0]: "Date"})
    local_df["Date"] = pd.to_datetime(local_df["Date"], errors="coerce")
    return local_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def get_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    save_raw: bool = True
) -> pd.DataFrame:
    """
    Downloads historical stock OHLCV data from Yahoo Finance or loads local CSV datasets.

    Args:
        ticker (str): Stock symbol (e.g., 'AAPL', 'MSFT', 'RELIANCE.NS').
        start_date (str): Start date string in 'YYYY-MM-DD' format.
        end_date (str): End date string in 'YYYY-MM-DD' format.
        save_raw (bool): Whether to cache raw CSV in data/raw/.

    Returns:
        pd.DataFrame: Cleaned pandas DataFrame containing columns:
                      ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily_Return'].
    """
    ticker = ticker.strip().upper()
    logger.info(f"Retrieving stock data for '{ticker}' from {start_date} to {end_date}...")

    # Sanity checks on date format
    try:
        dt_start = datetime.strptime(start_date, "%Y-%m-%d")
        dt_end = datetime.strptime(end_date, "%Y-%m-%d")
        if dt_start >= dt_end:
            raise ValueError(f"Start date ({start_date}) must be before end date ({end_date}).")
    except ValueError as ve:
        logger.error(f"Invalid date input for ticker {ticker}: {ve}")
        raise

    df = pd.DataFrame()
    local_is_complete = False

    # 1. Check if a local CSV matching the ticker exists in RAW_DATA_DIR or workspace
    clean_ticker = ticker.replace("^", "").replace("=", "_")
    local_files = list(RAW_DATA_DIR.glob(f"*{clean_ticker}*.csv")) + list(RAW_DATA_DIR.parent.parent.glob(f"*{clean_ticker}*.csv"))

    if local_files:
        candidates = []
        for local_path in sorted(set(local_files)):
            try:
                local_df = _load_local_candidate(local_path)
                mask = (local_df["Date"] >= dt_start) & (local_df["Date"] <= dt_end)
                filtered_df = local_df.loc[mask].copy()
                if len(filtered_df) > 5:
                    covers_start = local_df["Date"].min() <= pd.Timestamp(dt_start)
                    covers_end = local_df["Date"].max() >= pd.Timestamp(dt_end)
                    candidates.append((covers_start and covers_end, len(filtered_df), local_df["Date"].max(), local_path, filtered_df))
            except Exception as ex:
                logger.warning(f"Failed loading local dataset file ({local_path}): {ex}")

        if candidates:
            local_is_complete, _, _, local_path, df = max(candidates, key=lambda item: (item[0], item[1], item[2]))
            logger.info(f"Loaded {len(df)} rows from best-covered local dataset: {local_path}")
            df.attrs["data_source"] = f"Local dataset: {local_path.name}"

    # 2. Download from yfinance when local data is missing or does not cover the request.
    if df.empty or not local_is_complete:
        try:
            stock = yf.Ticker(ticker)
            online_df = stock.history(start=start_date, end=end_date, auto_adjust=False)

            if online_df.empty:
                logger.warning(f"Ticker history empty via Ticker API for {ticker}. Retrying with yf.download...")
                online_df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if online_df.empty and df.empty:
                raise ValueError(f"No stock data found for ticker '{ticker}'. Please check if symbol is valid.")

            if not online_df.empty:
                online_df = online_df.reset_index()
                if df.empty:
                    df = online_df
                    df.attrs["data_source"] = "Yahoo Finance"
                else:
                    df = pd.concat([df, online_df], ignore_index=True)
                    df.attrs["data_source"] = "Local dataset + Yahoo Finance extension"
        except Exception as e:
            if df.empty:
                logger.error(f"Network error or invalid ticker while downloading '{ticker}': {e}")
                raise RuntimeError(f"Failed to retrieve data for ticker '{ticker}': {e}")
            logger.warning(f"Could not extend local data from Yahoo Finance: {e}. Using local coverage.")

    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize column names
    rename_dict = {}
    for col in df.columns:
        c_lower = str(col).lower()
        if "date" in c_lower:
            rename_dict[col] = "Date"
        elif "open" in c_lower:
            rename_dict[col] = "Open"
        elif "high" in c_lower:
            rename_dict[col] = "High"
        elif "low" in c_lower:
            rename_dict[col] = "Low"
        elif "close" in c_lower and "adj" not in c_lower:
            rename_dict[col] = "Close"
        elif "adj" in c_lower or "adjusted" in c_lower:
            rename_dict[col] = "Adj Close"
        elif "vol" in c_lower:
            rename_dict[col] = "Volume"

    df = df.rename(columns=rename_dict)

    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # Verify required OHLCV columns exist
    required_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Stock data missing required columns: {missing}")

    # Data Cleaning & Formatting
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.drop_duplicates(subset=["Date"], keep="first")
    df = df.sort_values(by="Date").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0.0)

    # Calculate Daily Percentage Returns
    df["Daily_Return"] = df["Close"].pct_change().fillna(0.0)

    logger.info(f"Successfully processed {len(df)} trading days for '{ticker}'. Latest Close: ${df['Close'].iloc[-1]:.2f}")

    # Cache raw CSV if requested
    if save_raw:
        try:
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            out_filename = RAW_DATA_DIR / f"{clean_ticker}_{start_date}_to_{end_date}.csv"
            df.to_csv(out_filename, index=False)
            logger.info(f"Saved stock data cache to: {out_filename}")
        except Exception as e:
            logger.warning(f"Could not save stock data cache: {e}")

    return df


if __name__ == "__main__":
    # Test Data Collector with local RELIANCE.NS dataset
    test_df = get_stock_data("RELIANCE.NS", "2015-01-01", "2024-01-01")
    print(test_df.head())
    print("\nData summary:")
    print(test_df.info())
