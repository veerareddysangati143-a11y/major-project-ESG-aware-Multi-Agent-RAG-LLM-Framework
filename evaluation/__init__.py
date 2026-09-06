"""Data-backed evaluation metrics for the current stock analysis run."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from forecasting import predict_stock_trend


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
