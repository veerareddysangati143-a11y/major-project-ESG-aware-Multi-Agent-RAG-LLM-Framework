"""
========================================================================================
ESG-AWARE MULTI-AGENT RAG-LLM FRAMEWORK STREAMLIT DASHBOARD (app.py)
========================================================================================
Interactive Dashboard implementing the 7-Step Workflow Diagram:
1. Data Collection
2. Data Preprocessing
3. RAG Knowledge Pipeline
4. Autonomous Agents (Forecasting, Technical, Sentiment, ESG, Risk)
5. Decision Engine (Weighted Consensus Engine + LLM Decision Agent)
6. Output & Evaluation (BUY/HOLD/SELL, Explainable Recommendation, Metrics Suite)
7. Feedback Loop (Performance Monitor & Weight Adaptation)
8. Persistent Database & Vector Store Inspector (SQLite & FAISS Store)
========================================================================================
"""

from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import RAW_DATA_DIR, MODEL_BENCHMARKS
from data_collector import get_stock_data
from agents.orchestrator import MultiAgentOrchestrator
from technical import compute_technical_indicators
from evaluation import compute_framework_evaluation_metrics
from feedback import FeedbackLoopEngine
from database import DatabaseManager
from utils.constants import AgentName

# Streamlit Page Setup
st.set_page_config(
    page_title="ESG Multi-Agent RAG-LLM Framework",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    :root {
        --primary-dark: #0f172a;
        --accent-teal: #0d9488;
        --accent-green: #16a34a;
        --accent-red: #dc2626;
        --accent-gold: #d97706;
        --card-bg: #ffffff;
    }
    .stApp { background-color: #f8fafc; color: #1e293b; }
    
    .workflow-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .workflow-header h1 {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .workflow-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }
    
    .recommendation-banner {
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .banner-buy { background: linear-gradient(135deg, #15803d 0%, #16a34a 100%); }
    .banner-hold { background: linear-gradient(135deg, #b45309 0%, #d97706 100%); }
    .banner-sell { background: linear-gradient(135deg, #b91c1c 0%, #dc2626 100%); }
    
    .agent-card {
        background: white;
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def get_db():
    return DatabaseManager()


db = get_db()

@st.cache_data(show_spinner=False)
def load_market_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = get_stock_data(ticker, start, end, save_raw=True)
    db.insert_stock_data(ticker, df)
    return df


# Initialize Session State
if "feedback_engine" not in st.session_state:
    st.session_state.feedback_engine = FeedbackLoopEngine()

if "custom_news" not in st.session_state:
    st.session_state.custom_news = [
        "Company announces major investments in renewable solar infrastructure and net-zero target",
        "Quarterly earnings report shows strong financial growth exceeding market estimates",
        "New compliance audit confirms high corporate governance standards"
    ]

if "custom_esg_docs" not in st.session_state:
    st.session_state.custom_esg_docs = [
        "Sustainability Report 2024: Reduced carbon emissions by 25%. Increased renewable energy utilization across production plants.",
        "Governance Filing 10-K: Independent audit committee established with strict ethical compliance guidelines.",
        "SEC EDGAR 10-Q: Capital expenditure focused on clean technologies and employee diversity initiatives."
    ]


# Header Banner
st.markdown(
    """
    <div class="workflow-header">
        <h1>ESG-aware Multi-Agent RAG-LLM Framework</h1>
        <p>Sustainable Stock Market Prediction & Investment Decisions Architecture</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Execution Controls")
    ticker = st.text_input("Stock Ticker", value="RELIANCE.NS").strip().upper()
    
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365 * 5))
    with col_e:
        end_date = st.date_input("End Date", value=date.today())
        
    st.subheader("⚖️ Decision Weights")
    w_forecast = st.slider("Market Forecast Weight", 0.0, 1.0, 0.25, 0.05)
    w_technical = st.slider("Technical Analysis Weight", 0.0, 1.0, 0.20, 0.05)
    w_sentiment = st.slider("News Sentiment Weight", 0.0, 1.0, 0.15, 0.05)
    w_esg = st.slider("ESG Factor Weight", 0.0, 1.0, 0.25, 0.05)
    w_risk = st.slider("Risk Assessment Weight", 0.0, 1.0, 0.15, 0.05)

    weight_values = {
        "Market Forecast": w_forecast,
        "Technical Analysis": w_technical,
        "News Sentiment": w_sentiment,
        "ESG Factor": w_esg,
        "Risk Assessment": w_risk,
    }
    weight_total = sum(weight_values.values()) or 1.0
    st.caption("Normalized voting weights")
    st.dataframe(
        pd.DataFrame([
            {"Agent": name, "Configured": value, "Normalized": value / weight_total}
            for name, value in weight_values.items()
        ]),
        hide_index=True,
        use_container_width=True,
    )
    
    run_btn = st.button("🚀 Run Multi-Agent Framework", type="primary", use_container_width=True)

if "analysis_data" not in st.session_state or run_btn:
    try:
        with st.spinner(f"Collecting market data for {ticker}..."):
            df = load_market_data(ticker, start_date.isoformat(), end_date.isoformat())
            st.session_state.analysis_data = df
    except Exception as err:
        st.error(f"Error retrieving data for {ticker}: {err}")
        st.stop()

df = st.session_state.analysis_data
processed_data = compute_technical_indicators(df).dropna(subset=["SMA 50", "RSI"]).reset_index(drop=True)

if len(processed_data) < 2:
    st.error("The selected date range does not contain enough usable trading days. Choose a wider range.")
    st.stop()

if len(processed_data) < 50:
    st.warning(f"Only {len(processed_data)} usable trading days are available. Forecast confidence may be limited.")

# Persist the same feature-rich rows that the user sees in the dashboard.
db.insert_stock_data(ticker, processed_data)

# Run Orchestrator Pipeline
custom_weights_map = {
    "forecast": w_forecast,
    "technical": w_technical,
    "sentiment": w_sentiment,
    "esg": w_esg,
    "risk": w_risk
}

orchestrator = MultiAgentOrchestrator()
pipeline_output = orchestrator.run({
    "data": processed_data,
    "ticker": ticker,
    "headlines": st.session_state.custom_news,
    "documents": st.session_state.custom_esg_docs
}, custom_weights=custom_weights_map)

results = pipeline_output["results"]
consensus = pipeline_output["consensus"]
explanation = pipeline_output["explanation"]
metrics_data = compute_framework_evaluation_metrics(processed_data)

# Save one prediction per explicit analysis run, rather than on every Streamlit rerun.
rec_signal = consensus.metadata.get("recommendation", "HOLD")
run_key = f"{ticker}:{start_date}:{end_date}:{rec_signal}:{round(consensus.score, 4)}"
if st.session_state.get("last_logged_run") != run_key:
    db.log_prediction(ticker, rec_signal, consensus.score, consensus.confidence, explanation.evidence[0])
    st.session_state.last_logged_run = run_key

latest_row = processed_data.iloc[-1]
prev_row = processed_data.iloc[-2]
price_change_pct = (latest_row["Close"] / prev_row["Close"] - 1) * 100

# Top Overview Metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Latest Price", f"₹{latest_row['Close']:,.2f}", f"{price_change_pct:+.2f}%")
m2.metric("Consensus Signal", rec_signal)
m3.metric("Confidence Score", f"{consensus.confidence:.0%}")
m4.metric("RSI (14)", f"{latest_row['RSI']:.1f}")
m5.metric("Database Entries", sum(db.get_all_table_counts().values()))

st.markdown("---")

# Main 7-Stage Workflow Tabs + Database Tab
tabs = st.tabs([
    "📥 1 & 2. Data & Preprocessing",
    "🔍 3. RAG Pipeline",
    "🤖 4. Autonomous Agents",
    "⚖️ 5. Decision Engine",
    "🎯 6. Output & Evaluation",
    "🔄 7. Feedback Loop",
    "🗄️ Database & Vector Store"
])

# --------------------------------------------------------------------------------------
# TAB 1 & 2: Data Collection & Data Preprocessing
# --------------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("What is happening in this analysis?")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Trading days loaded", f"{len(df):,}")
    summary_cols[1].metric("Analysis days", f"{len(processed_data):,}")
    summary_cols[2].metric("Latest close", f"₹{latest_row['Close']:,.2f}")
    summary_cols[3].metric("Data source", df.attrs.get("data_source", "Local or cached data"))
    st.info("This is AI-generated investment decision support, not guaranteed financial advice. The system reads historical prices, ESG records, documents, and indicators, asks specialist agents for independent signals, and combines them into one recommendation.")
    st.caption(f"Dataset coverage: {df['Date'].min().date().isoformat()} to {df['Date'].max().date().isoformat()} | These dates come from the loaded rows.")

    st.subheader("1. Data Collection Layer")
    dc1, dc2, dc3, dc4, dc5 = st.columns(5)
    with dc1:
        st.markdown("**Yahoo Finance**\n- Prices & Volume\n- OHLCV Feed: ✅ Active")
    with dc2:
        st.markdown("**Reuters / News**\n- Financial Articles\n- Feed Count: " + str(len(st.session_state.custom_news)))
    with dc3:
        st.markdown("**ESG Reports**\n- Sustainability Docs\n- Document Count: " + str(len(st.session_state.custom_esg_docs)))
    with dc4:
        st.markdown("**SEC EDGAR**\n- 10-K, 10-Q Filings\n- Status: Parsed")
    with dc5:
        st.markdown("**World Bank**\n- Macro Indicators\n- Inflation & GDP: Loaded")

    st.markdown("---")
    st.subheader("2. Data Preprocessing & Feature Engineering")
    st.write("Noise removal, feature scaling, RSI, MACD, and Bollinger Bands calculation.")

    # Price & Moving Averages Chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=processed_data["Date"], open=processed_data["Open"], high=processed_data["High"],
        low=processed_data["Low"], close=processed_data["Close"], name="Price"
    ))
    fig_price.add_trace(go.Scatter(x=processed_data["Date"], y=processed_data["SMA 20"], name="SMA 20", line=dict(color="#0284c7")))
    fig_price.add_trace(go.Scatter(x=processed_data["Date"], y=processed_data["SMA 50"], name="SMA 50", line=dict(color="#d97706")))
    fig_price.update_layout(title=f"{ticker} Historical OHLCV Price & Moving Averages", height=450, template="plotly_white", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_price, use_container_width=True)

# --------------------------------------------------------------------------------------
# TAB 3: RAG Knowledge Pipeline
# --------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("3. RAG Knowledge Pipeline")
    rag_col1, rag_col2 = st.columns([1, 2])
    
    with rag_col1:
        st.markdown("### RAG Architecture")
        st.markdown("- **Document Collection**: Sustainability reports & 10-K filings")
        st.markdown("- **Chunking**: Overlapping sliding window (600 tokens)")
        st.markdown("- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2`")
        st.markdown("- **Vector Store**: SQLite / FAISS Vector DB")
        st.markdown("- **Top-K Retrieval**: Context Top 3 Chunks")
        
    with rag_col2:
        st.markdown("### Top-K Retrieved Vector Document Context")
        rag_res = next((r for r in results if r.agent_name == AgentName.RAG.value), None)
        if rag_res:
            st.write("**User query:**", rag_res.metadata.get("query", "ESG sustainability, regulatory compliance, and risk"))
        if rag_res and rag_res.evidence:
            for idx, chunk in enumerate(rag_res.evidence, 1):
                st.info(f"**Retrieved Chunk #{idx}**:\n{chunk}")
        else:
            st.warning("No RAG chunks retrieved.")

# --------------------------------------------------------------------------------------
# TAB 4: Autonomous Agents
# --------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("4. Autonomous Agents Performance")
    agent_cols = st.columns(3)
    
    for idx, res in enumerate(results):
        col_idx = idx % 3
        with agent_cols[col_idx]:
            sig_val = res.signal.value if hasattr(res.signal, "value") else str(res.signal)
            badge_color = "#16a34a" if sig_val == "BUY" else "#dc2626" if sig_val == "SELL" else "#d97706"
            
            st.markdown(
                f"""
                <div class="agent-card">
                    <h4 style="margin:0; color:#0f172a;">{res.agent_name.replace('_', ' ').title()}</h4>
                    <p style="margin:0.2rem 0; font-weight:bold; color:{badge_color};">Signal: {sig_val} (Score: {res.score:+.2f})</p>
                    <p style="margin:0; font-size:0.85rem; color:#64748b;">Confidence: {res.confidence:.0%} | Execution: {res.execution_time_ms:.1f}ms</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.expander(f"Inspect {res.agent_name} details"):
                if res.metadata.get("analysis"):
                    st.write("**Agent analysis:**", res.metadata["analysis"])
                st.write("**Evidence:**", res.evidence)
                st.write("**Metadata:**", res.metadata)

# --------------------------------------------------------------------------------------
# TAB 5: Decision Engine
# --------------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("5. Decision Engine")
    de_col1, de_col2 = st.columns(2)
    
    with de_col1:
        st.markdown("### Weighted Consensus Engine")
        st.write("Dynamic agent signal aggregation. Voting weights are normalized to 100%; RAG and regime provide evidence/context rather than extra votes.")
        breakdown_data = consensus.metadata.get("breakdown", [])
        if breakdown_data:
            st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)
        else:
            st.write("Consensus Score:", f"{consensus.score:+.3f}")
            st.write("Confidence:", f"{consensus.confidence:.0%}")
            
    with de_col2:
        st.markdown("### LLM Decision Agent (Llama 3 / FinGPT)")
        st.success(f"**Final LLM Rationale:**\n\n{explanation.evidence[0]}")
        st.json(explanation.metadata)

# --------------------------------------------------------------------------------------
# TAB 6: Output & Evaluation Dashboard
# --------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("6. Output & Evaluation Dashboard")
    banner_class = "banner-buy" if rec_signal == "BUY" else "banner-sell" if rec_signal == "SELL" else "banner-hold"
    
    st.markdown(
        f"""
        <div class="recommendation-banner {banner_class}">
            <h2 style="margin:0; font-size:2.5rem; color:white;">FINAL RECOMMENDATION: {rec_signal}</h2>
            <p style="margin:0.5rem 0 0 0; font-size:1.2rem; color:white;">Confidence Score: {consensus.confidence:.0%} | Composite Score: {consensus.score:+.3f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("Comprehensive Evaluation Metrics Suite")
    ev1, ev2, ev3 = st.columns(3)
    
    with ev1:
        st.markdown("#### 📉 Forecasting Metrics")
        st.dataframe(pd.DataFrame([metrics_data["forecasting"]]), use_container_width=True)
        st.markdown("#### 🎯 Classification Metrics")
        st.dataframe(pd.DataFrame([metrics_data["classification"]]), use_container_width=True)
        
    with ev2:
        st.markdown("#### 🛡️ Risk Metrics")
        st.dataframe(pd.DataFrame([metrics_data["risk"]]), use_container_width=True)
        st.markdown("#### 🌿 ESG Alignment Metrics")
        st.dataframe(pd.DataFrame([metrics_data["esg_alignment"]]), use_container_width=True)
        
    with ev3:
        st.markdown("#### 🤖 LLM Metrics")
        st.dataframe(pd.DataFrame([metrics_data["llm_metrics"]]), use_container_width=True)
        st.markdown("#### 📈 Trading Performance Metrics")
        st.dataframe(pd.DataFrame([metrics_data["trading_metrics"]]), use_container_width=True)

    st.markdown("#### Model Performance Reference")
    st.caption("The holdout metrics above are calculated from the current dataset. The table below contains recorded notebook benchmark results and is labelled separately from the live holdout evaluation.")
    model_rows = [
        {"Model": model_name, "Accuracy": values["Accuracy"], "RMSE": values["RMSE"], "MAE": values["MAE"], "R2": values["R2"], "Source": "Notebook benchmark"}
        for model_name, values in MODEL_BENCHMARKS.items()
    ]
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True)

    st.markdown("#### Actual Price vs Predicted Price")
    evaluation_rows = pd.DataFrame(metrics_data["actual_vs_predicted"])
    st.dataframe(evaluation_rows, use_container_width=True)
    if not evaluation_rows.empty:
        evaluation_fig = go.Figure()
        evaluation_fig.add_trace(go.Scatter(x=evaluation_rows["actual_date"], y=evaluation_rows["actual_close"], name="Actual Close"))
        evaluation_fig.add_trace(go.Scatter(x=evaluation_rows["actual_date"], y=evaluation_rows["predicted_close"], name="Predicted Close"))
        evaluation_fig.update_layout(title="Holdout Evaluation: Actual vs Predicted Close", height=350, template="plotly_white")
        st.plotly_chart(evaluation_fig, use_container_width=True)

    esg_result = next((r for r in results if r.agent_name == AgentName.ESG.value), None)
    if esg_result:
        st.markdown("#### ESG Evidence Used by the Decision Engine")
        esg_scores = esg_result.metadata.get("category_scores", {})
        st.dataframe(pd.DataFrame([esg_scores]), use_container_width=True)
        st.caption(f"Source: {esg_result.metadata.get('source', 'Unavailable')} | As of: {esg_result.metadata.get('as_of_date', 'Unavailable')}")
        st.info(esg_result.metadata.get("source_evidence", "No ESG evidence available."))

# --------------------------------------------------------------------------------------
# TAB 7: Feedback Loop
# --------------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("7. Feedback Loop & Continuous Adaptation")
    fb1, fb2 = st.columns(2)
    
    with fb1:
        st.markdown("### Performance Monitor")
        actual_rtn = st.number_input("Simulate Actual Market Outcome Return (%)", value=2.5, step=0.5) / 100.0
        
        if st.button("Log Prediction Outcome"):
            st.session_state.feedback_engine.log_prediction(ticker, rec_signal, consensus.score, actual_rtn)
            st.success("Outcome logged successfully into continuous learning engine.")
            
        history = st.session_state.feedback_engine.history
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True)
        else:
            st.info("No outcomes logged yet in this session.")
            
    with fb2:
        st.markdown("### Dynamic Weight Adaptation")
        updated_weights = st.session_state.feedback_engine.get_updated_agent_weights(custom_weights_map)
        st.write("Reinforced Dynamic Agent Weights:")
        st.json(updated_weights)

# --------------------------------------------------------------------------------------
# TAB 8: Database & Vector Store Viewer
# --------------------------------------------------------------------------------------
with tabs[6]:
    st.subheader("🗄️ Persistent Database & Vector Store Inspector")
    st.write(f"**Database Location:** `{db.db_path}`")
    
    counts = db.get_all_table_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stock OHLCV Rows", counts.get("stocks_data", 0))
    c2.metric("RAG Vector Docs", counts.get("documents_rag", 0))
    c3.metric("Prediction Logs", counts.get("agent_predictions", 0))
    c4.metric("Feedback Records", counts.get("feedback_history", 0))
    
    st.markdown("---")
    db_table = st.selectbox("Select Database Table to Inspect", ["documents_rag", "agent_predictions", "stocks_data", "feedback_history"])
    table_data = db.get_table_data(db_table, limit=50)
    
    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.info(f"No records found in table `{db_table}`.")
        
    st.markdown("---")
    st.subheader("🔍 Vector Similarity Query Tester")
    v_query = st.text_input("Enter Query for Vector DB Search", value="carbon emissions renewable solar energy")
    if v_query:
        vector_results = db.search_vector_db(v_query, top_k=3)
        st.dataframe(pd.DataFrame(vector_results), use_container_width=True)

st.caption("ESG-aware Multi-Agent RAG-LLM Framework | System Architecture v2.0")
