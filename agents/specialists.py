"""Deterministic specialist agents for the stock intelligence pipeline."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from agents import AgentResult, BaseAgent
from utils.constants import AgentName, SignalType
from utils.helpers import numeric_to_signal
from regime_detection import detect_market_regime


def _data(context: Dict[str, Any]) -> pd.DataFrame:
    return context["data"]


class ForecastingAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.FORECASTING)

    def analyze(self, context):
        data = _data(context)
        close = data["Close"]
        window = min(20, len(close))
        slope = np.polyfit(np.arange(window), close.tail(window), 1)[0]
        expected_return = (slope * 5) / close.iloc[-1]
        score = float(np.clip(expected_return * 20, -1, 1))
        return AgentResult(self.name, numeric_to_signal(score), score, min(1.0, 0.45 + abs(score) * 0.5), [f"20-session slope: {slope:.4f}", f"5-session projected return: {expected_return:.2%}"], {"method": "linear trend baseline", "horizon_days": 5})


class TechnicalAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.TECHNICAL)

    def analyze(self, context):
        row = _data(context).iloc[-1]
        signals = []
        score = 0.0
        if row["Close"] > row["SMA 20"]:
            score += 0.35; signals.append("price above SMA 20")
        else:
            score -= 0.35; signals.append("price below SMA 20")
        if row["SMA 20"] > row["SMA 50"]:
            score += 0.35; signals.append("SMA 20 above SMA 50")
        else:
            score -= 0.35; signals.append("SMA 20 below SMA 50")
        if row["RSI"] < 30:
            score += 0.2; signals.append("RSI oversold")
        elif row["RSI"] > 70:
            score -= 0.2; signals.append("RSI overbought")
        else:
            signals.append("RSI neutral")
        return AgentResult(self.name, numeric_to_signal(score), score, 0.75, signals, {"rsi": float(row["RSI"]), "macd": float(row["MACD"])})


class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.SENTIMENT)

    def analyze(self, context):
        headlines = context.get("headlines", [])
        positive = {"growth", "profit", "upgrade", "strong", "surge", "green", "award"}
        negative = {"loss", "downgrade", "weak", "fall", "risk", "fraud", "pollution"}
        values = []
        for headline in headlines:
            words = set(str(headline).lower().split())
            values.append((len(words & positive) - len(words & negative)) / max(1, len(words & positive | words & negative)))
        score = float(np.clip(np.mean(values) if values else 0.0, -1, 1))
        return AgentResult(self.name, numeric_to_signal(score), score, 0.35 if not headlines else 0.6, headlines or ["No news feed supplied; sentiment is neutral."], {"headline_count": len(headlines), "mode": "keyword baseline"})


class ESGAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.ESG)

    def analyze(self, context):
        text = " ".join(context.get("documents", [])).lower()
        categories = {"environmental": ["emission", "renewable", "carbon", "water"], "social": ["employee", "community", "safety", "diversity"], "governance": ["board", "audit", "ethics", "compliance"]}
        scores = {name: min(100, 50 + 12 * sum(text.count(term) for term in terms)) for name, terms in categories.items()}
        overall = sum(scores.values()) / len(scores)
        signal = SignalType.BUY if overall >= 65 else SignalType.HOLD if overall >= 45 else SignalType.SELL
        evidence = [f"{name.title()}: {value:.0f}/100" for name, value in scores.items()]
        evidence.append("Scores use available local documents and are not a substitute for audited ESG research.")
        return AgentResult(self.name, signal, (overall - 50) / 50, 0.3 if not text else 0.55, evidence, {"scores": scores, "document_count": len(context.get("documents", []))})


class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.RISK)

    def analyze(self, context):
        returns = _data(context)["Daily_Return"].dropna()
        volatility = float(returns.std() * np.sqrt(252))
        drawdown = float((_data(context)["Close"] / _data(context)["Close"].cummax() - 1).min())
        sharpe = float((returns.mean() * 252 - 0.045) / volatility) if volatility else 0.0
        risk_score = float(np.clip(volatility * 100 + abs(drawdown) * 50, 0, 100))
        score = float(np.clip(0.5 - risk_score / 100, -1, 1))
        signal = numeric_to_signal(score)
        return AgentResult(self.name, signal, score, 0.8, [f"Annualized volatility: {volatility:.2%}", f"Maximum drawdown: {drawdown:.2%}", f"Sharpe ratio: {sharpe:.2f}"], {"risk_score": risk_score, "volatility": volatility, "max_drawdown": drawdown, "sharpe": sharpe})


class RAGAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.RAG)

    def analyze(self, context):
        documents = context.get("documents", [])
        evidence = [str(document)[:240] for document in documents[:5]] or ["No local ESG or filing documents were indexed."]
        return AgentResult(self.name, SignalType.NEUTRAL, 0.0, 0.25 if not documents else 0.5, evidence, {"retrieved_chunks": len(evidence), "mode": "local document retrieval"})


class RegimeAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.REGIME)

    def analyze(self, context):
        result = detect_market_regime(_data(context))
        regime = result["regime"]
        score = {"BULL": 1.0, "BEAR": -1.0, "SIDEWAYS": 0.0, "HIGH_VOLATILITY": 0.0}.get(regime, 0.0)
        return AgentResult(
            self.name,
            regime,
            score,
            result["confidence"],
            [
                f"Regime: {regime}",
                f"50-session trend strength: {result['trend_strength']:.2%}",
                f"20-session annualized volatility: {result['volatility']:.2%}",
                f"Current drawdown: {result['drawdown']:.2%}",
            ],
            result,
        )


class LLMDecisionAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentName.LLM_DECISION)

    def analyze(self, context):
        results = context.get("results", [])
        signals = [result.signal.value if isinstance(result.signal, SignalType) else str(result.signal) for result in results]
        buy = signals.count("BUY")
        sell = signals.count("SELL")
        signal = SignalType.BUY if buy > sell else SignalType.SELL if sell > buy else SignalType.HOLD
        score = (buy - sell) / max(1, len(signals))
        rationale = f"Rule-based explanation: {buy} bullish votes, {sell} bearish votes, and {len(signals)} specialist outputs were considered."
        return AgentResult(self.name, signal, score, 0.5, [rationale], {"provider": "local deterministic fallback", "buy_votes": buy, "sell_votes": sell})


SPECIALIST_AGENTS = [ForecastingAgent, TechnicalAgent, SentimentAgent, ESGAgent, RiskAgent, RAGAgent, RegimeAgent]
