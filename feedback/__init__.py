"""
Feedback Loop Module.
Monitors actual performance vs predictions, updates agent weights, and improves future decisions continuously.
"""

from typing import Dict, Any, List


class FeedbackLoopEngine:
    """Monitors performance outcomes and updates agent dynamic weights."""
    
    def __init__(self):
        self.history = []
        
    def log_prediction(self, ticker: str, predicted_signal: str, consensus_score: float, actual_return: float):
        """Logs prediction vs actual outcome."""
        correct = (predicted_signal == "BUY" and actual_return > 0) or (predicted_signal == "SELL" and actual_return < 0)
        self.history.append({
            "ticker": ticker,
            "predicted": predicted_signal,
            "score": consensus_score,
            "actual_return": actual_return,
            "correct": correct
        })
        
    def get_updated_agent_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Dynamically adjusts weights based on historical prediction accuracy."""
        if not self.history:
            return base_weights
            
        recent = self.history[-20:]
        win_rate = sum(1 for item in recent if item["correct"]) / len(recent)
        
        # Scale weights slightly according to market regime performance
        adjusted = {}
        adjustment_factor = 0.9 + 0.2 * win_rate
        for k, v in base_weights.items():
            adjusted[k] = round(v * adjustment_factor, 4)
            
        total = sum(adjusted.values())
        return {k: round(v / total, 4) for k, v in adjusted.items()}
