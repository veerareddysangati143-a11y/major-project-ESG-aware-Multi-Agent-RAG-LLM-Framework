"""Standalone JSON API for the ESG multi-agent stock analysis pipeline."""

from datetime import date, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="ESG Multi-Agent Stock Analysis API",
    version="1.0.0",
    description="JSON API for explainable stock signals and ESG-aware consensus analysis.",
)


@app.get("/")
def root() -> Dict[str, str]:
    """Return API links without importing the heavy analysis pipeline."""
    return {
        "service": "esg-multi-agent-stock-api",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "analysis": "/analyze",
    }


class AnalysisRequest(BaseModel):
    ticker: str = Field(default="RELIANCE.NS", min_length=1)
    start_date: date = Field(default_factory=lambda: date.today() - timedelta(days=365 * 5))
    end_date: date = Field(default_factory=date.today)
    headlines: List[str] = Field(default_factory=list)
    documents: List[str] = Field(default_factory=list)
    weights: Optional[Dict[str, float]] = None


@app.get("/health")
def health() -> Dict[str, str]:
    """Return a lightweight liveness response for monitoring and mobile clients."""
    return {"status": "ok", "service": "esg-multi-agent-stock-api"}


@app.post("/analyze")
def analyze(request: AnalysisRequest) -> Dict:
    """Run the complete stock pipeline and return JSON-serializable results."""
    if request.start_date >= request.end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        # Keep cold-start health requests lightweight and load the pipeline only when used.
        from agents.orchestrator import MultiAgentOrchestrator
        from data_collector import get_stock_data
        from technical import compute_technical_indicators
        from evaluation import compute_framework_evaluation_metrics

        raw_data = get_stock_data(
            request.ticker,
            request.start_date.isoformat(),
            request.end_date.isoformat(),
            save_raw=False,
        )
        processed_data = compute_technical_indicators(raw_data).dropna(
            subset=["SMA 50", "RSI"]
        ).reset_index(drop=True)
        if len(processed_data) < 2:
            raise HTTPException(status_code=422, detail="Not enough usable trading days for analysis")

        context = {
            "data": processed_data,
            "ticker": request.ticker.strip().upper(),
            "headlines": request.headlines,
            "documents": request.documents,
        }
        output = MultiAgentOrchestrator().run(context, custom_weights=request.weights)
        consensus = output["consensus"]
        explanation = output["explanation"]
        evaluation = compute_framework_evaluation_metrics(processed_data)

        return {
            "ticker": context["ticker"],
            "data": {
                "rows": len(raw_data),
                "usable_rows": len(processed_data),
                "first_date": raw_data["Date"].min().date().isoformat(),
                "last_date": raw_data["Date"].max().date().isoformat(),
                "source": raw_data.attrs.get("data_source", "market data"),
                "latest_close": float(processed_data.iloc[-1]["Close"]),
            },
            "recommendation": consensus.metadata.get("recommendation", "HOLD"),
            "score": float(consensus.score),
            "confidence": float(consensus.confidence),
            "explanation": explanation.evidence[0] if explanation.evidence else "",
            "explanation_details": explanation.metadata,
            "agents": [result.to_dict() for result in output["results"]],
            "consensus_breakdown": consensus.metadata.get("breakdown", []),
            "regime": consensus.metadata.get("regime", {}),
            "confidence_details": consensus.metadata.get("confidence_details", {}),
            "evaluation": evaluation,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error