"""
Helper Functions and Utilities for ESG Multi-Agent System.
Includes math operations, type conversions, and signal transformations.
"""

import json
from enum import Enum
from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from utils.constants import SignalType


def signal_to_numeric(signal: Union[SignalType, str]) -> float:
    """
    Converts a string or SignalType enum into a numeric value for weighted consensus.
    BUY / POSITIVE -> +1.0
    HOLD / NEUTRAL -> 0.0
    SELL / NEGATIVE -> -1.0
    """
    if isinstance(signal, SignalType):
        sig_str = signal.value.upper()
    else:
        sig_str = str(signal).upper()

    if sig_str in ["BUY", "POSITIVE", "BULLISH"]:
        return 1.0
    elif sig_str in ["SELL", "NEGATIVE", "BEARISH"]:
        return -1.0
    elif sig_str in ["HOLD", "NEUTRAL"]:
        return 0.0
    else:
        return 0.0


def numeric_to_signal(score: float, buy_threshold: float = 0.20, sell_threshold: float = -0.20) -> SignalType:
    """
    Converts a numeric score [-1.0, +1.0] back into a BUY / HOLD / SELL signal.
    """
    if score >= buy_threshold:
        return SignalType.BUY
    elif score <= sell_threshold:
        return SignalType.SELL
    else:
        return SignalType.HOLD


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively converts numpy data types, NaNs, and pandas objects to standard Python types
    to allow clean JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return 0.0
    return obj


def format_currency(val: float, currency_symbol: str = "$") -> str:
    """Formats float value as currency string."""
    try:
        return f"{currency_symbol}{val:,.2f}"
    except (ValueError, TypeError):
        return f"{currency_symbol}0.00"


def format_percentage(val: float) -> str:
    """Formats float value (e.g. 0.045) as percentage string ('4.50%')."""
    try:
        return f"{val * 100:.2f}%"
    except (ValueError, TypeError):
        return "0.00%"


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts value to float, handling None, NaN, inf."""
    try:
        if val is None:
            return default
        f_val = float(val)
        if np.isnan(f_val) or np.isinf(f_val):
            return default
        return f_val
    except (ValueError, TypeError):
        return default
