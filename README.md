# ESG-Aware Multi-Agent RAG-LLM Framework for Sustainable Stock Market Prediction and Green Investment Decision Making

An enterprise-grade, multi-agent financial intelligence and decision-support system designed to predict stock market movement and generate explainable **BUY / HOLD / SELL** recommendations by fusing quantitative forecasting, technical indicators, financial news NLP, corporate ESG extraction, document retrieval (RAG), financial risk modeling, dynamic regime detection, and LLM reasoning.

---

## 🌟 System Architecture

```
                                  +---------------------------------------+
                                  |         Streamlit User Interface      |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      Multi-Agent Orchestrator         |
                                  +---------------------------------------+
                                                      |
      +-------------------+-------------------+-------+-------+-------------------+-------------------+
      |                   |                   |               |                   |                   |
      v                   v                   v               v                   v                   v
+------------+     +------------+     +---------------+ +------------+     +------------+     +------------+
| Market     |     | Technical  |     | FinBERT News  | | ESG        |     | Risk       |     | RAG        |
| Forecast   |     | Analysis   |     | Sentiment     | | Analysis   |     | Assessment |     | Document   |
| (LSTM/GRU/ |     | (RSI/MACD/ |     | (ProsusAI/    | | (E/S/G     |     | (Sharpe/   |     | Knowledge  |
| Prophet)   |     | BB/MAs)    |     | FinBERT)      | | Extractor) |     | VaR/DD)    |     | (FAISS)    |
+------------+     +------------+     +---------------+ +------------+     +------------+     +------------+
      |                   |                   |               |                   |                   |
      +-------------------+-------------------+-------+-------+-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       Market Regime Detector          |
                                  |   (BULL / BEAR / SIDEWAYS / VOLATILE) |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   Dynamic Weighted Consensus Engine   |
                                  |     (Confidence-Weighted Signal)      |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       LLM Reasoning Agent             |
                                  |       (Ollama / Llama 3)              |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   Explainable Recommendation &        |
                                  |       Evaluation Dashboard            |
                                  +---------------------------------------+
```

---

## 🤖 Specialized Agents

1. **Market Forecasting Agent**: Combines deep learning (Keras LSTM & GRU) with NeuralProphet time-series models to predict price trends and expected returns.
2. **Technical Analysis Agent**: Computes SMA (20/50), EMA (12/26), RSI (14), MACD, Bollinger Bands, and ATR to identify momentum and reversal signals.
3. **News Sentiment Agent**: Employs `ProsusAI/finbert` to analyze financial news headlines and quantify market sentiment (-1 to +1).
4. **ESG Analysis Agent**: Parses annual & sustainability PDF reports to extract Environmental, Social, and Governance metrics and evidence passages.
5. **Risk Assessment Agent**: Computes Sharpe Ratio, Maximum Drawdown, Value at Risk (VaR 95%), and Volatility to produce a 0-100 normalized risk score.
6. **RAG Knowledge Agent**: Uses `SentenceTransformers` and `FAISS` vector database to index and retrieve corporate filings for factual groundedness.
7. **Market Regime Agent**: Detects current macro-state (`BULL`, `BEAR`, `SIDEWAYS`, `HIGH_VOLATILITY`) to dynamically recalibrate agent weights.
8. **Dynamic Weighted Consensus Engine**: Aggregates signals using regime-adjusted, confidence-weighted mathematical voting.
9. **LLM Decision Agent**: Leverages Ollama (Llama 3) to synthesize all agent outputs into structured, evidence-grounded investment rationales.

---

## 📂 Folder Structure

```
esg_multi_agent_stock_system/
│
├── app.py                      # Main Streamlit Dashboard Application
├── config.py                   # Centralized Configuration & Hyperparameters
├── requirements.txt            # System Dependencies
├── .env.example                # Environment Variables Template
├── README.md                   # System Documentation
│
├── data/                       # Data Store (Raw, Processed, News, ESG, Vector DB)
├── models/                     # Trained Models (LSTM, GRU, NeuralProphet, Scalers)
├── agents/                     # Autonomous Multi-Agent Implementations
├── forecasting/                # Time-Series Preprocessing & Deep Learning Models
├── technical/                  # Technical Indicators & Signal Generation
├── sentiment/                  # FinBERT NLP Pipeline & News Loaders
├── esg/                        # PDF Text Parsers & ESG Metric Extractors
├── rag/                        # Chunking, Embeddings, FAISS Vector Store, Retriever
├── risk/                       # Financial Risk Analytics & Score Calculators
├── consensus/                  # Regime-Aware Dynamic Weight Allocation Engine
├── llm/                        # Prompt Templates, Ollama Client & Explanation Generator
├── evaluation/                 # Metrics, Backtester & Performance Verification
├── utils/                      # Logging, Constants, Helpers & Data Serialization
└── tests/                      # Unit & Integration Test Suite
```

The dashboard now runs the complete local multi-agent pipeline. The specialist implementations live in `agents/specialists.py` and are coordinated by `agents/orchestrator.py`. The pipeline includes forecasting, technical analysis, news sentiment, ESG scoring, risk assessment, local document retrieval, market regime detection, confidence-weighted consensus, and a local explanation agent. Heavy model downloads and Ollama are optional; the default path is deterministic and can run from the cached stock dataset.

---

## ⚡ Installation & Setup

### 1. Prerequisites
- Python 3.11 or 3.12
- Git
- Ollama (for local Llama 3 LLM execution)

### 2. Virtual Environment Setup

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Setup Ollama (Local Llama 3)
Download and install Ollama from [https://ollama.com](https://ollama.com).
Pull the Llama 3 model:
```bash
ollama pull llama3
```

---

## 🚀 Running the Application

To launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The project includes Streamlit server configuration in `.streamlit/config.toml`, so it listens on all local network interfaces. Open your browser at `http://localhost:8501` on the host computer.

### Sharing on the same Wi-Fi/network

Find the host computer's IPv4 address with PowerShell:

```powershell
ipconfig
```

Send this address to another computer on the same network:

```text
http://YOUR_IPV4_ADDRESS:8501
```

Allow Python/Streamlit through Windows Firewall if the other computer cannot connect. The recipient must be on the same Wi-Fi/network; `localhost` must not be shared because it refers to the recipient's own computer.

### Sharing over the internet

A private `10.x.x.x` address cannot be opened from outside your network. Use a hosted deployment or a tunnel such as Cloudflare Tunnel/ngrok. With Cloudflare Tunnel installed, start the app and run:

```powershell
cloudflared tunnel --url http://localhost:8501
```

Send the generated `https://...trycloudflare.com` URL. Keep both the Streamlit process and tunnel running while the link is in use.

### Continuous JSON API

The project also includes a REST API for mobile apps, websites, and other clients. Start it from the project directory:

```powershell
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

API URLs on the host computer:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

On the same Wi-Fi/network, replace `localhost` with the host computer's IPv4 address. The analysis endpoint is `POST /analyze` and accepts JSON:

```json
{
      "ticker": "RELIANCE.NS",
      "start_date": "2023-01-01",
      "end_date": "2024-01-01",
      "headlines": ["Company reports strong growth"],
      "documents": ["Sustainability report confirms renewable energy investment"]
}
```

To make the API public, expose port `8000` through Cloudflare Tunnel or deploy the project to a cloud host. A LAN URL such as `http://10.x.x.x:8000` cannot work on a phone using mobile data.

---

## ⚠️ Research & Safety Disclaimer

> **IMPORTANT**: This application is an academic and research prototype designed for demonstration and testing of multi-agent AI architectures in financial contexts. The generated stock market forecasts, signals, ESG scores, and BUY/HOLD/SELL recommendations **do not constitute personalized financial advice**. Always conduct independent due diligence before making investment decisions.
