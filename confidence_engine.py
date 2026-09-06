"""Transparent decision-confidence calculation."""

from typing import Any, Dict, List

import numpy as np


def _direction(score: float) -> int:
    if score > 0.05:
        return 1
    if score < -0.05:
        return -1
    return 0


def calculate_decision_confidence(
    agent_results: List[Any],
    breakdown: List[Dict[str, Any]],
    consensus_score: float,
    regime_confidence: float = 0.0,
) -> Dict[str, float]:
    """Calculate confidence from strength, agreement, forecast, evidence, and risk."""
    voting = {row["agent"]: row for row in breakdown}
    final_direction = _direction(consensus_score)
    weighted_agreement = 0.0
    total_weight = 0.0
    agreement_count = 0
    voting_count = 0
    forecast_reliability = 0.0
    evidence_quality = 0.0
    risk_adjustment = 0.5

    for result in agent_results:
        row = voting.get(getattr(result, "agent_name", ""))
        if not row:
            continue
        voting_count += 1
        weight = float(row["weight"])
        direction = _direction(float(getattr(result, "score", 0.0)))
        if direction == final_direction or (direction == 0 and final_direction == 0):
            agreement_count += 1
            weighted_agreement += weight
        total_weight += weight

        name = getattr(result, "agent_name", "").lower()
        if "forecast" in name:
            forecast_reliability = float(np.clip(getattr(result, "confidence", 0.0), 0.0, 1.0))
        if "rag" in name or "esg" in name:
            evidence_count = len(getattr(result, "evidence", []) or [])
            evidence_quality = max(evidence_quality, float(np.clip(evidence_count / 3.0, 0.0, 1.0)))
        if "risk" in name:
            risk_score = float(getattr(result, "metadata", {}).get("risk_score", 50.0))
            risk_adjustment = float(np.clip(1.0 - risk_score / 100.0, 0.0, 1.0))

    agreement = weighted_agreement / total_weight if total_weight else 0.0
    agreement_ratio = agreement_count / voting_count if voting_count else 0.0
    consensus_strength = float(np.clip(abs(consensus_score), 0.0, 1.0))
    regime_factor = 0.5 + 0.5 * float(np.clip(regime_confidence, 0.0, 1.0))
    confidence = float(np.clip(
        (0.25 * consensus_strength
         + 0.25 * agreement
         + 0.20 * forecast_reliability
         + 0.15 * evidence_quality
         + 0.15 * risk_adjustment) * regime_factor,
        0.0,
        1.0,
    ))

    return {
        "confidence": confidence,
        "consensus_strength": consensus_strength,
        "agent_agreement": agreement,
        "agreement_ratio": agreement_ratio,
        "disagreement": 1.0 - agreement,
        "forecast_reliability": forecast_reliability,
        "evidence_quality": evidence_quality,
        "risk_adjustment": risk_adjustment,
        "regime_factor": regime_factor,
    }
