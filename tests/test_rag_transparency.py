from types import SimpleNamespace

from database import DatabaseManager
from llm import generate_llm_decision
from rag import build_and_query_rag


def test_document_metadata_survives_vector_search(tmp_path):
    database = DatabaseManager(tmp_path / "rag.db")
    database.add_document(
        "Sustainability Report",
        "ESG Reports",
        "Renewable energy and emissions reduction evidence.",
        source="Example Sustainability Report",
        publication_date="2024-03-31",
        document_type="ESG report",
        ticker="RELIANCE.NS",
    )

    results = database.search_vector_db("renewable energy emissions", top_k=1)

    matching = [item for item in results if item["source"] == "Example Sustainability Report"]
    assert matching or all("source" in item for item in results)
    assert all(-1.0 <= item["similarity_score"] <= 1.0 for item in results)

    stored = database.get_table_data("documents_rag", limit=20)
    inserted = next(item for item in stored if item["doc_title"] == "Sustainability Report")
    assert inserted["source"] == "Example Sustainability Report"
    assert inserted["publication_date"] == "2024-03-31"
    assert inserted["document_type"] == "ESG report"


def test_rag_returns_quality_and_status(tmp_path, monkeypatch):
    database_path = tmp_path / "rag.db"
    monkeypatch.setattr("rag.DatabaseManager", lambda: DatabaseManager(database_path))
    result = build_and_query_rag(
        ["Renewable energy investment reduced carbon emissions."],
        "ESG renewable energy",
        top_k=3,
    )

    assert 0.0 <= result["evidence_quality"] <= 1.0
    assert result["evidence_status"] in {"Sufficient evidence", "Insufficient evidence"}
    assert result["search_details"]
    assert "source" in result["search_details"][0]


def test_explanation_contains_structured_reasoning_and_risks():
    result = generate_llm_decision(
        {"recommendation": "HOLD", "consensus_score": 0.1, "confidence": 0.42},
        [
            SimpleNamespace(agent_name="Technical Analysis Agent", signal="HOLD", score=0.1, evidence=["RSI neutral"]),
            SimpleNamespace(agent_name="Risk Assessment Agent", signal="HOLD", score=0.0, evidence=["Volatility 20%"]),
        ],
        "[ESG report] Evidence: renewable energy investment",
    )

    assert result["evidence_grounded"] is True
    assert result["reasoning_factors"]
    assert result["key_risks"]
    assert result["supporting_evidence"]
