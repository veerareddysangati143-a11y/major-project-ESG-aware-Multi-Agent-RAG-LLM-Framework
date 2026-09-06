"""Measurable market-regime detection for the stock analysis pipeline."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from utils.constants import MarketRegime


def detect_market_regime(data: pd.DataFrame, volatility_threshold: float = 0.45, trend_threshold: float = 0.08) -> Dict[str, Any]:
    """Classify market conditions using trend, momentum, drawdown, and volatility."""
    if len(data) < 2 or "Close" not in data:
        return {
            "regime": MarketRegime.UNKNOWN.value,
            "confidence": 0.0,
            "volatility": 0.0,
            "trend_strength": 0.0,
            "drawdown": 0.0,
            "features": {},
        }

    close = data["Close"].astype(float)
    returns = data.get("Daily_Return", close.pct_change()).dropna()
    recent_returns = returns.tail(min(20, len(returns)))
    recent_close = close.tail(min(50, len(close)))
    volatility = float(recent_returns.std(ddof=0) * np.sqrt(252)) if len(recent_returns) > 1 else 0.0
    trend_strength = float(recent_close.iloc[-1] / recent_close.iloc[0] - 1.0) if len(recent_close) > 1 else 0.0
    drawdown = float((close / close.cummax() - 1.0).iloc[-1])

    if volatility >= volatility_threshold:
        regime = MarketRegime.HIGH_VOLATILITY
    elif trend_strength >= trend_threshold:
        regime = MarketRegime.BULL
    elif trend_strength <= -trend_threshold:
        regime = MarketRegime.BEAR
    else:
        regime = MarketRegime.SIDEWAYS

    trend_signal = min(1.0, abs(trend_strength) / max(trend_threshold, 1e-9))
    volatility_signal = min(1.0, volatility / max(volatility_threshold, 1e-9))
    confidence = float(np.clip(0.5 * trend_signal + 0.5 * volatility_signal, 0.0, 1.0))
    if regime == MarketRegime.SIDEWAYS:
        confidence = float(np.clip(1.0 - abs(trend_strength) / max(trend_threshold, 1e-9), 0.0, 1.0))

    return {
        "regime": regime.value,
        "confidence": confidence,
        "volatility": volatility,
        "trend_strength": trend_strength,
        "drawdown": drawdown,
        "features": {
            "recent_return_50": trend_strength,
            "annualized_volatility_20": volatility,
            "current_drawdown": drawdown,
        },
    }
