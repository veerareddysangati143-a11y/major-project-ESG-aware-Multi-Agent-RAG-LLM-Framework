"""
Weighted Consensus Engine Module.
Combines autonomous specialist agent outputs with dynamic dynamic weights to compute final consensus score & confidence.
"""

from typing import List, Dict, Any
import numpy as np


def compute_weighted_consensus(agent_results: List[Any], custom_weights: Dict[str, float] = None) -> Dict[str, Any]:
    """Combines agent signals into a unified dynamic consensus score."""
    if custom_weights is None:
        custom_weights = {
            "forecast": 0.25,
            "technical": 0.20,
            "sentiment": 0.15,
            "esg": 0.25,
            "risk": 0.15
        }

    weight_total = sum(max(0.0, float(weight)) for weight in custom_weights.values()) or 1.0
    normalized_weights = {
        key: max(0.0, float(weight)) / weight_total
        for key, weight in custom_weights.items()
    }
        
    valid_results = [r for r in agent_results if getattr(r, "error", None) is None]
    
    total_weight = 0.0
    weighted_score = 0.0
    confidence_sum = 0.0
    
    breakdown = []
    for res in valid_results:
        agent_name = getattr(res, "agent_name", "").lower()
        score = float(getattr(res, "score", 0.0))
        confidence = float(getattr(res, "confidence", 0.5))
        
        weight_key = next((key for key in normalized_weights if key in agent_name), None)
        if weight_key is None:
            continue
        weight = normalized_weights[weight_key]
                
        effective_weight = weight * confidence
        weighted_score += score * effective_weight
        total_weight += effective_weight
        confidence_sum += confidence
        
        breakdown.append({
            "agent": getattr(res, "agent_name", "Unknown"),
            "signal": str(getattr(res, "signal", "NEUTRAL")),
            "score": score,
            "confidence": confidence,
            "weight": weight,
            "configured_weight": custom_weights.get(weight_key, 0.0),
            "normalized_weight": weight,
            "effective_weight": effective_weight
        })
        
    final_score = float(np.clip(weighted_score / total_weight, -1.0, 1.0)) if total_weight > 0 else 0.0
    final_confidence = min(1.0, confidence_sum / max(1, len(valid_results)))
    
    if final_score >= 0.20:
        recommendation = "BUY"
    elif final_score <= -0.20:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
        
    return {
        "recommendation": recommendation,
        "consensus_score": final_score,
        "confidence": final_confidence,
        "total_agents": len(valid_results),
        "breakdown": breakdown,
        "normalized_weights": normalized_weights
    }
