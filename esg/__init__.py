"""
ESG Analysis Module.
Extracts Environmental (Carbon, Renewables), Social, and Governance factors.
"""

from typing import List, Dict, Any

import pandas as pd

from config import ESG_DATA_DIR, ESG_WEIGHTS


def load_esg_data(ticker: str) -> pd.DataFrame:
    """Load dated ESG records for a ticker."""
    filename = f"{ticker.strip().lower().replace('.', '_')}_esg.csv"
    path = ESG_DATA_DIR / filename
    if not path.exists():
        path = ESG_DATA_DIR / "reliance_esg.csv"
    data = pd.read_csv(path)
    data["as_of_date"] = pd.to_datetime(data["as_of_date"])
    return data[data["ticker"].str.upper() == ticker.strip().upper()].sort_values("as_of_date")


def evaluate_esg_factors(documents: List[str], ticker: str = "RELIANCE.NS") -> Dict[str, Any]:
    """Extracts ESG criteria from sustainability reports and documents."""
    text = " ".join(documents).lower() if documents else ""
    
    categories = {
        "environmental": ["emission", "renewable", "carbon", "water", "clean", "solar", "greenhouse", "recycling"],
        "social": ["employee", "community", "safety", "diversity", "human rights", "inclusion", "health"],
        "governance": ["board", "audit", "ethics", "compliance", "transparency", "anti-corruption", "stakeholder"]
    }
    
    table = load_esg_data(ticker)
    latest = table.iloc[-1] if not table.empty else None
    scores = {
        "environmental": float(latest["environmental"]) if latest is not None else 50.0,
        "social": float(latest["social"]) if latest is not None else 50.0,
        "governance": float(latest["governance"]) if latest is not None else 50.0,
    }
    details = {}
    for cat_name, terms in categories.items():
        matches = sum(text.count(term) for term in terms) if text else 2 # Baseline
        scores[cat_name] = min(100.0, scores[cat_name] + min(12.0, 2.0 * matches))
        details[cat_name] = {"keyword_matches": matches, "score": scores[cat_name]}
        
    overall_score = sum(scores[key] * ESG_WEIGHTS[key] for key in scores)
    # ESG signal mapping: overall >= 65 -> BUY, >= 45 -> HOLD, < 45 -> SELL
    esg_signal = "BUY" if overall_score >= 65 else "HOLD" if overall_score >= 45 else "SELL"
    normalized_score = float((overall_score - 50.0) / 50.0) # -1.0 to +1.0
    
    return {
        "overall_score": float(overall_score),
        "normalized_score": normalized_score,
        "signal": esg_signal,
        "category_scores": scores,
        "details": details,
        "document_count": len(documents),
        "esg_record_count": len(table),
        "as_of_date": latest["as_of_date"].date().isoformat() if latest is not None else None,
        "source": latest["source"] if latest is not None else "No ESG table record",
        "source_evidence": latest["evidence"] if latest is not None else "No dated ESG evidence available.",
        "evidence": [f"{cat.title()} Score: {score:.1f}/100" for cat, score in scores.items()]
    }
