"""Regime-aware adaptive weight calculation."""

from typing import Any, Dict, Optional

import numpy as np

from config import BASE_AGENT_WEIGHTS, REGIME_WEIGHT_MAP


VOTING_AGENTS = ("forecast", "technical", "sentiment", "esg", "risk")


def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
    cleaned = {key: max(0.0, float(weights.get(key, 0.0))) for key in VOTING_AGENTS}
    total = sum(cleaned.values()) or 1.0
    return {key: value / total for key, value in cleaned.items()}


def calculate_adaptive_weights(
    regime: str,
    agent_scores: Optional[Dict[str, float]] = None,
    market_features: Optional[Dict[str, Any]] = None,
    base_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return normalized regime-aware weights.

    Regime profiles are configurable starting policies, not claims of optimality.
    Optional scores/features are accepted so a later optimizer can learn adjustments
    without changing the consensus interface.
    """
    del agent_scores, market_features
    base = _normalise(base_weights or BASE_AGENT_WEIGHTS)
    profile = _normalise(REGIME_WEIGHT_MAP.get(str(regime).upper(), BASE_AGENT_WEIGHTS))
    adapted = {key: base[key] * profile[key] for key in VOTING_AGENTS}
    return _normalise(adapted)
