"""Context-aware dashboard assistant for explaining the current stock analysis."""

from typing import Any, Dict


def answer_user_question(question: str, analysis_context: Dict[str, Any]) -> str:
    """Answer dashboard questions from current analysis results without inventing facts."""
    question_lower = question.lower()
    ticker = analysis_context.get("ticker", "the selected stock")
    recommendation = analysis_context.get("recommendation", "HOLD")
    confidence = analysis_context.get("confidence", 0.0)
    score = analysis_context.get("score", 0.0)
    agents = analysis_context.get("agents", [])
    agent_lines = []
    for result in agents:
        signal = result.signal.value if hasattr(result.signal, "value") else str(result.signal)
        agent_lines.append(f"{result.agent_name}: {signal} ({result.score:+.2f})")

    if any(word in question_lower for word in ["esg", "environment", "social", "governance"]):
        esg = analysis_context.get("esg", {})
        category_scores = esg.get("category_scores", {})
        return (
            f"The ESG score is {esg.get('overall_esg_score', 'not available')}/100. "
            f"Environmental: {category_scores.get('environmental', 'N/A')}, "
            f"Social: {category_scores.get('social', 'N/A')}, "
            f"Governance: {category_scores.get('governance', 'N/A')}. "
            f"The evidence is from {esg.get('source', 'the configured ESG table')} "
            f"dated {esg.get('as_of_date', 'the latest available record')}."
        )

    if "risk" in question_lower:
        risk = analysis_context.get("risk", {})
        return (
            f"The risk agent reports annualized volatility of {risk.get('volatility', 0):.2%}, "
            f"maximum drawdown of {risk.get('max_drawdown', 0):.2%}, and a risk score of "
            f"{risk.get('risk_score', 0):.1f}/100. Review the Risk Assessment agent card for the full evidence."
        )

    if any(word in question_lower for word in ["forecast", "prediction", "price", "close", "target"]):
        forecast = analysis_context.get("forecast", {})
        current_price = forecast.get("current_price", "N/A")
        projected_price = forecast.get("projected_price", "N/A")
        projected_return = forecast.get("projected_return", 0)
        if "why" in question_lower or "method" in question_lower:
            return (
                f"The forecast uses a 60-day log-price trend, a 20-day short trend, and EMA momentum. "
                f"The trend contribution is {forecast.get('trend_return', 0):+.2%}, while EMA momentum contributes "
                f"{forecast.get('momentum_return', 0):+.2%}. Together they produce a {forecast.get('horizon_days', 5)}-day "
                f"projection of {projected_price} from the current close of {current_price} "
                f"({projected_return:+.2%})."
            )
        return (
            f"The current close for {ticker} is {current_price}. The {forecast.get('horizon_days', 5)}-day "
            f"projection is {projected_price} ({projected_return:+.2%}). "
            f"Method: {forecast.get('prediction_method', 'configured forecasting method')}."
        )

    if any(word in question_lower for word in ["rag", "evidence", "document"]):
        evidence = analysis_context.get("rag_evidence", [])
        if not evidence:
            return "No retrieved RAG evidence is available for this run. Open the RAG Pipeline tab to inspect retrieval status."
        return "The RAG pipeline retrieved these evidence items:\n\n" + "\n".join(
            f"{index}. {item}" for index, item in enumerate(evidence[:3], 1)
        )

    if any(word in question_lower for word in ["how", "dashboard", "tab", "help"]):
        return (
            "Use the tabs from left to right: Data & Preprocessing shows price history and indicators; "
            "RAG Pipeline shows retrieved evidence; Autonomous Agents shows individual reasoning; "
            "Decision Engine shows normalized weights; Output & Evaluation shows the final decision and holdout metrics."
        )

    if any(word in question_lower for word in ["why", "recommendation", "buy", "sell", "hold", "decision"]):
        return (
            f"The current decision for {ticker} is {recommendation} with {confidence:.0%} confidence "
            f"and a composite score of {score:+.2f}. The agent signals are: "
            f"{'; '.join(agent_lines)}. This is decision support, not a guaranteed price prediction. "
            "Open the Autonomous Agents and Decision Engine tabs to inspect each signal and weight."
        )

    return (
        f"I can explain the {ticker} analysis, recommendation reasons, ESG score, risk, forecast, "
        "RAG evidence, agent signals, or how to use the dashboard. Try asking: 'Why is the recommendation BUY?'"
    )
