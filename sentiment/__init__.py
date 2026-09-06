"""
News Sentiment Analysis Module.
Processes financial news articles and extracts sentiment scores using FinBERT/Keyword models.
"""

from typing import List, Dict, Any
import numpy as np


def analyze_news_sentiment(headlines: List[str]) -> Dict[str, Any]:
    """Analyzes financial news headlines for positive/negative sentiment."""
    if not headlines:
        return {
            "score": 0.0,
            "sentiment": "NEUTRAL",
            "confidence": 0.5,
            "headline_count": 0,
            "breakdown": {"positive": 0, "neutral": 0, "negative": 0},
            "evidence": ["No news feed supplied; sentiment is neutral."]
        }

    positive_keywords = {"growth", "profit", "upgrade", "strong", "surge", "green", "award", "record", "expansion", "beat"}
    negative_keywords = {"loss", "downgrade", "weak", "fall", "risk", "fraud", "pollution", "sanction", "cut", "lawsuit"}

    pos_count = 0
    neg_count = 0
    neu_count = 0
    scores = []

    for headline in headlines:
        words = set(str(headline).lower().split())
        p_matches = len(words & positive_keywords)
        n_matches = len(words & negative_keywords)
        
        if p_matches > n_matches:
            pos_count += 1
            scores.append(0.6 + 0.2 * min(p_matches, 2))
        elif n_matches > p_matches:
            neg_count += 1
            scores.append(-0.6 - 0.2 * min(n_matches, 2))
        else:
            neu_count += 1
            scores.append(0.0)

    avg_score = float(np.clip(np.mean(scores), -1.0, 1.0)) if scores else 0.0
    sentiment = "POSITIVE" if avg_score >= 0.15 else "NEGATIVE" if avg_score <= -0.15 else "NEUTRAL"

    return {
        "score": avg_score,
        "sentiment": sentiment,
        "confidence": min(1.0, 0.4 + 0.1 * len(headlines)),
        "headline_count": len(headlines),
        "breakdown": {"positive": pos_count, "neutral": neu_count, "negative": neg_count},
        "evidence": headlines[:5]
    }
