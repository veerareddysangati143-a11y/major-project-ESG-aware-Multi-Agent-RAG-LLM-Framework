"""Transparent regime-aware consensus engine."""

from typing import Any, Dict, List, Optional

import numpy as np

from adaptive_weights import calculate_adaptive_weights
from confidence_engine import calculate_decision_confidence
from config import BUY_THRESHOLD, SELL_THRESHOLD


def compute_weighted_consensus(
    agent_results: List[Any],
    custom_weights: Optional[Dict[str, float]] = None,
    regime_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate ``sum(normalized_weight * agent_score)`` transparently."""
    regime_context = regime_context or {}
    regime = regime_context.get("regime", "SIDEWAYS")
    adaptive_weights = calculate_adaptive_weights(
        regime,
        agent_scores={getattr(result, "agent_name", ""): float(getattr(result, "score", 0.0)) for result in agent_results},
        market_features=regime_context.get("features", {}),
        base_weights=custom_weights,
    )

    breakdown = []
    valid_results = [result for result in agent_results if getattr(result, "error", None) is None]
    for result in valid_results:
        agent_name = getattr(result, "agent_name", "")
        agent_key = next((key for key in adaptive_weights if key in agent_name.lower()), None)
        if agent_key is None:
            continue
        score = float(np.clip(getattr(result, "score", 0.0), -1.0, 1.0))
        weight = adaptive_weights[agent_key]
        breakdown.append({
            "agent": agent_name,
            "signal": str(getattr(result, "signal", "NEUTRAL")),
            "score": score,
            "confidence": float(np.clip(getattr(result, "confidence", 0.0), 0.0, 1.0)),
            "weight": weight,
            "normalized_weight": weight,
            "configured_weight": (custom_weights or {}).get(agent_key, weight),
            "contribution": score * weight,
        })

    composite_score = float(np.clip(sum(row["contribution"] for row in breakdown), -1.0, 1.0))
    confidence_details = calculate_decision_confidence(
        valid_results,
        breakdown,
        composite_score,
        float(regime_context.get("confidence", 0.0)),
    )
    if composite_score >= BUY_THRESHOLD and confidence_details["confidence"] >= 0.40:
        recommendation = "BUY"
    elif composite_score <= SELL_THRESHOLD and confidence_details["confidence"] >= 0.40:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    return {
        "recommendation": recommendation,
        "consensus_score": composite_score,
        "confidence": confidence_details["confidence"],
        "total_agents": len(breakdown),
        "breakdown": breakdown,
        "normalized_weights": adaptive_weights,
        "confidence_details": confidence_details,
        "regime": regime_context,
    }
