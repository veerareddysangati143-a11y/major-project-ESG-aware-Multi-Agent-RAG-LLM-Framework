"""
Technical Analysis Module.
Calculates RSI, MACD, Bollinger Bands, and Moving Averages.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes RSI, MACD, Bollinger Bands, and Moving Averages on OHLCV data."""
    data = df.copy()
    close = data["Close"]
    
    # Moving Averages
    data["SMA 20"] = close.rolling(20).mean()
    data["SMA 50"] = close.rolling(50).mean()
    data["EMA 12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA 26"] = close.ewm(span=26, adjust=False).mean()
    
    # MACD
    data["MACD"] = data["EMA 12"] - data["EMA 26"]
    data["MACD Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD Hist"] = data["MACD"] - data["MACD Signal"]
    
    # RSI
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    std_20 = close.rolling(20).std()
    data["BB Upper"] = data["SMA 20"] + (2 * std_20)
    data["BB Lower"] = data["SMA 20"] - (2 * std_20)
    data["BB Width"] = (data["BB Upper"] - data["BB Lower"]) / data["SMA 20"]
    
    return data


def analyze_technical_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates composite technical signal and breakdown."""
    data = compute_technical_indicators(df)
    latest = data.iloc[-1]
    
    score = 0.0
    reasons = []
    
    if latest["Close"] > latest["SMA 20"]:
        score += 0.35
        reasons.append("Price above 20-day SMA")
    else:
        score -= 0.35
        reasons.append("Price below 20-day SMA")
        
    if latest["SMA 20"] > latest["SMA 50"]:
        score += 0.35
        reasons.append("20-day SMA above 50-day SMA (Golden trend)")
    else:
        score -= 0.35
        reasons.append("20-day SMA below 50-day SMA (Bearish trend)")
        
    rsi = float(latest["RSI"]) if not np.isnan(latest["RSI"]) else 50.0
    if rsi < 30:
        score += 0.20
        reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi > 70:
        score -= 0.20
        reasons.append(f"RSI overbought ({rsi:.1f})")
    else:
        reasons.append(f"RSI neutral ({rsi:.1f})")
        
    macd = float(latest["MACD"]) if not np.isnan(latest["MACD"]) else 0.0
    macd_sig = float(latest["MACD Signal"]) if not np.isnan(latest["MACD Signal"]) else 0.0
    if macd > macd_sig:
        score += 0.10
        reasons.append("MACD above Signal line")
    else:
        score -= 0.10
        reasons.append("MACD below Signal line")
        
    return {
        "score": float(np.clip(score, -1.0, 1.0)),
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_sig,
        "sma_20": float(latest["SMA 20"]),
        "sma_50": float(latest["SMA 50"]),
        "reasons": reasons,
        "data": data
    }
