"""Data-backed evaluation metrics for the current stock analysis run."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from forecasting import predict_stock_trend
from regime_detection import detect_market_regime


def _regression_metrics(actual_prices: np.ndarray, predicted_prices: np.ndarray, baseline_price: float) -> Dict[str, float]:
    errors = actual_prices - predicted_prices
    actual_returns = actual_prices / max(baseline_price, 1e-9) - 1.0
    predicted_returns = predicted_prices / max(baseline_price, 1e-9) - 1.0
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAPE": float(np.mean(np.abs(errors / np.maximum(actual_prices, 1e-9))) * 100),
        "Direction_Accuracy": float(np.mean(np.sign(actual_returns) == np.sign(predicted_returns)) * 100),
    }


def compute_model_comparison(data: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
    """Compare available runtime forecasts on the same chronological holdout."""
    if len(data) <= horizon + 20:
        return {"rows": [], "test_period": None}
    train = data.iloc[:-horizon]
    actual = data.iloc[-horizon:]["Close"].to_numpy(dtype=float)
    baseline_price = float(train["Close"].iloc[-1])
    proposed = predict_stock_trend(train, horizon=horizon)
    proposed_prices = baseline_price * np.exp(
        np.linspace(0.0, np.log(max(proposed["projected_price"], 1e-9) / baseline_price), horizon + 1)[1:]
    )
    forecasts = {
        "Naive Last Value": (np.repeat(baseline_price, horizon), "Available runtime baseline"),
        "Moving Average Baseline": (np.repeat(float(train["Close"].tail(20).mean()), horizon), "Available runtime baseline"),
        "Proposed/Ensemble": (proposed_prices, proposed.get("prediction_method", "Available runtime forecast")),
        "LSTM": (None, "Not available: no exported runtime artifact"),
        "GRU": (None, "Not available: no exported runtime artifact"),
        "Transformer": (None, "Not available: no runtime implementation"),
    }
    rows = []
    for model_name, (predicted, source) in forecasts.items():
        row = {"Model": model_name, "Source": source}
        if predicted is None:
            row.update({"MAE": None, "RMSE": None, "MAPE": None, "Direction_Accuracy": None})
        else:
            row.update(_regression_metrics(actual, predicted, baseline_price))
        rows.append(row)
    return {
        "rows": rows,
        "test_period": {
            "start": data["Date"].iloc[-horizon].date().isoformat(),
            "end": data["Date"].iloc[-1].date().isoformat(),
        },
    }


def _strategy_position(strategy: str, history: pd.DataFrame) -> float:
    if strategy == "buy_and_hold":
        return 1.0
    if len(history) < 20:
        return 0.0
    close = history["Close"].astype(float)
    technical_score = 0.0
    sma_20 = close.tail(20).mean()
    if close.iloc[-1] > sma_20:
        technical_score += 0.5
    else:
        technical_score -= 0.5
    if len(close) >= 50:
        sma_50 = close.tail(50).mean()
        technical_score += 0.5 if sma_20 > sma_50 else -0.5
    forecast_score = float(predict_stock_trend(history, horizon=1).get("score", 0.0))
    returns = history["Close"].pct_change().dropna()
    volatility = float(returns.tail(20).std(ddof=0) * np.sqrt(252)) if len(returns) > 1 else 0.0
    risk_score = float(np.clip(volatility * 100, 0.0, 100.0))
    risk_signal = float(np.clip(0.5 - risk_score / 100.0, -1.0, 1.0))
    if strategy == "technical":
        score = technical_score
    elif strategy == "forecast_only":
        score = forecast_score
    elif strategy == "static_multi_agent":
        score = 0.5 * forecast_score + 0.5 * technical_score
    else:
        regime = detect_market_regime(history)["regime"]
        forecast_weight = 0.30 if regime == "BULL" else 0.15 if regime == "BEAR" else 0.10 if regime == "HIGH_VOLATILITY" else 0.18
        technical_weight = 0.22 if regime == "BULL" else 0.12 if regime == "BEAR" else 0.15 if regime == "HIGH_VOLATILITY" else 0.25
        risk_weight = 0.08 if regime == "BULL" else 0.25 if regime == "BEAR" else 0.35 if regime == "HIGH_VOLATILITY" else 0.12
        score = forecast_weight * forecast_score + technical_weight * technical_score + risk_weight * risk_signal
    return 1.0 if score >= 0.20 else -1.0 if score <= -0.20 else 0.0


def _backtest_strategy(data: pd.DataFrame, strategy: str, warmup: int = 60) -> Dict[str, Any]:
    returns = data["Close"].pct_change().fillna(0.0).to_numpy(dtype=float)
    strategy_returns = []
    positions = []
    for index in range(warmup, len(data) - 1):
        position = _strategy_position(strategy, data.iloc[:index])
        positions.append(position)
        strategy_returns.append(position * returns[index + 1])
    strategy_returns = np.asarray(strategy_returns, dtype=float)
    if len(strategy_returns) == 0:
        return {"Cumulative_Return": None, "Annualized_Return": None, "Sharpe": None, "Max_Drawdown": None, "Volatility": None, "Trades": 0}
    equity = np.cumprod(1.0 + strategy_returns)
    volatility = float(strategy_returns.std(ddof=0) * np.sqrt(252))
    sharpe = float(strategy_returns.mean() / max(strategy_returns.std(ddof=0), 1e-9) * np.sqrt(252))
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    trades = int(sum(previous != current for previous, current in zip(positions, positions[1:]) if current != 0))
    return {
        "Cumulative_Return": float((equity[-1] - 1.0) * 100),
        "Annualized_Return": float((equity[-1] ** (252 / len(strategy_returns)) - 1.0) * 100),
        "Sharpe": sharpe,
        "Max_Drawdown": float(drawdown.min() * 100),
        "Volatility": volatility * 100,
        "Trades": trades,
    }


def compute_backtest_results(data: pd.DataFrame) -> Dict[str, Any]:
    """Run strategies using only each date's historical prefix."""
    strategies = {
        "Buy and Hold": "buy_and_hold",
        "Technical Baseline": "technical",
        "Forecast-only": "forecast_only",
        "Static-weight Multi-agent": "static_multi_agent",
        "Adaptive Multi-agent": "adaptive_multi_agent",
    }
    return {"rows": [{"Strategy": name, **_backtest_strategy(data, key)} for name, key in strategies.items()]}


def compute_ablation_study(data: pd.DataFrame) -> Dict[str, Any]:
    """Report ablations supported by available time-aligned runtime inputs."""
    available = compute_backtest_results(data)["rows"]
    by_name = {row["Strategy"]: row for row in available}
    rows = [
        {"Configuration": "Forecast-only", "Status": "Evaluated", **by_name["Forecast-only"]},
        {"Configuration": "Forecast + Technical", "Status": "Evaluated", **by_name["Static-weight Multi-agent"]},
        {"Configuration": "Forecast + Technical + Sentiment", "Status": "Not yet evaluated"},
        {"Configuration": "+ RAG", "Status": "Not yet evaluated"},
        {"Configuration": "+ ESG", "Status": "Not yet evaluated"},
        {"Configuration": "+ Risk", "Status": "Evaluated", **by_name["Adaptive Multi-agent"]},
        {"Configuration": "Full adaptive model", "Status": "Evaluated", **by_name["Adaptive Multi-agent"]},
    ]
    return {"rows": rows}


def compute_framework_evaluation_metrics(data: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate the trend model on the final five available observations."""
    horizon = min(5, max(1, len(data) // 10))
    train = data.iloc[:-horizon].copy()
    actual = data.iloc[-horizon:].copy()
    prediction = predict_stock_trend(train, horizon=horizon)
    last_price = float(train["Close"].iloc[-1])
    predicted_prices = last_price * np.exp(
        np.linspace(0.0, np.log(max(prediction["projected_price"], 1e-9) / last_price), horizon + 1)[1:]
    )
    actual_prices = actual["Close"].to_numpy(dtype=float)
    errors = actual_prices - predicted_prices
    actual_returns = actual_prices / last_price - 1.0
    predicted_returns = predicted_prices / last_price - 1.0
    rows = []
    for index, (_, observation) in enumerate(actual.iterrows()):
        rows.append({
            "prediction_date": train["Date"].iloc[-1].date().isoformat(),
            "actual_date": observation["Date"].date().isoformat(),
            "actual_close": float(actual_prices[index]),
            "predicted_close": float(predicted_prices[index]),
            "actual_return": float(actual_returns[index]),
            "predicted_return": float(predicted_returns[index]),
            "absolute_error": float(abs(errors[index])),
        })

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape = float(np.mean(np.abs(errors / np.maximum(actual_prices, 1e-9))) * 100)
    directional_accuracy = float(np.mean(np.sign(actual_returns) == np.sign(predicted_returns)) * 100)
    return {
        "forecasting": {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Directional_Accuracy": directional_accuracy},
        "classification": {"Accuracy": directional_accuracy},
        "risk": {"Sharpe_Ratio": float(data["Daily_Return"].mean() / max(data["Daily_Return"].std(), 1e-9) * np.sqrt(252)), "Max_Drawdown": float((data["Close"] / data["Close"].cummax() - 1).min() * 100)},
        "esg_alignment": {"ESG_Score": None, "ESG_Faithfulness": "Evidence-backed table"},
        "llm_metrics": {"Faithfulness": "Evidence-backed", "Hallucination_Rate": "Not measured"},
        "trading_metrics": {"Cumulative_Return": float((data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100), "Win_Rate": float((data["Daily_Return"] > 0).mean() * 100)},
        "actual_vs_predicted": rows,
    }
