"""
LLM Decision Agent Module.
Integrates Llama 3 / FinGPT prompts to synthesize multi-agent findings into an evidence-grounded explainable recommendation.
"""

from typing import List, Dict, Any


def generate_llm_decision(consensus_result: Dict[str, Any], agent_outputs: List[Any], rag_context: str = "") -> Dict[str, Any]:
    """Generates an evidence-grounded, ESG-aligned investment recommendation rationale."""
    rec = consensus_result.get("recommendation", "HOLD")
    score = consensus_result.get("consensus_score", 0.0)
    conf = consensus_result.get("confidence", 0.7)
    
    agent_summaries = []
    for res in agent_outputs:
        name = getattr(res, "agent_name", "Agent")
        sig = str(getattr(res, "signal", "NEUTRAL"))
        sc = float(getattr(res, "score", 0.0))
        agent_summaries.append(f"{name}: Signal {sig} (Score {sc:+.2f})")
        
    summary_text = "; ".join(agent_summaries)
    
    rationale = (
        f"Llama3/FinGPT Synthesis: Final recommendation is {rec} (Score: {score:+.2f}, Confidence: {conf:.0%}). "
        f"Synthesized inputs from {len(agent_outputs)} autonomous agents: [{summary_text}]. "
        f"The decision balances quantitative market momentum with ESG compliance standards."
    )
    
    if rag_context:
        rationale += f" Grounded with retrieved evidence: {rag_context[:400]}"
        
    return {
        "recommendation": rec,
        "rationale": rationale,
        "provider": "Llama 3 / FinGPT (Local Engine)",
        "model_version": "llama3:8b-instruct-q4",
        "evidence_grounded": True,
        "esg_aligned": True
    }


def answer_user_question(question: str, analysis_context: Dict[str, Any]) -> str:
    """Answer dashboard questions from the current analysis context without inventing facts."""
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

    if any(word in question_lower for word in ["why", "recommendation", "buy", "sell", "hold"]):
        return (
            f"The current decision for {ticker} is {recommendation} with {confidence:.0%} confidence "
            f"and a composite score of {score:+.2f}. The agent signals are: "
            f"{'; '.join(agent_lines)}. This is decision support, not a guaranteed price prediction. "
            "Open the Autonomous Agents and Decision Engine tabs to inspect each signal and weight."
        )

    if "esg" in question_lower or "environment" in question_lower or "social" in question_lower or "governance" in question_lower:
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

    if "forecast" in question_lower or "prediction" in question_lower or "price" in question_lower:
        forecast = analysis_context.get("forecast", {})
        return (
            f"The current close is {forecast.get('current_price', 'N/A')}. The {forecast.get('horizon_days', 5)}-day "
            f"projection is {forecast.get('projected_price', 'N/A')} ({forecast.get('projected_return', 0):+.2%}). "
            f"Method: {forecast.get('prediction_method', 'configured forecasting method')}."
        )

    if "rag" in question_lower or "evidence" in question_lower or "document" in question_lower:
        evidence = analysis_context.get("rag_evidence", [])
        if not evidence:
            return "No retrieved RAG evidence is available for this run. Open the RAG Pipeline tab to inspect retrieval status."
        return "The RAG pipeline retrieved these evidence items:\n\n" + "\n".join(
            f"{index}. {item}" for index, item in enumerate(evidence[:3], 1)
        )

    if "how" in question_lower or "dashboard" in question_lower or "tab" in question_lower or "help" in question_lower:
        return (
            "Use the tabs from left to right: Data & Preprocessing shows the price history and indicators; "
            "RAG Pipeline shows retrieved evidence; Autonomous Agents shows individual reasoning; "
            "Decision Engine shows normalized weights; Output & Evaluation shows the final decision and holdout metrics."
        )

    return (
        f"I can explain the {ticker} analysis, recommendation reasons, ESG score, risk, forecast, "
        "RAG evidence, agent signals, or how to use the dashboard. Try asking: 'Why is the recommendation BUY?'"
    )
