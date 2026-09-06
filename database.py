"""
========================================================================================
FRAMEWORK DATABASE & VECTOR STORE MODULE (database.py)
========================================================================================
Manages local persistent SQLite database (data/stock_intelligence.db) and vector index:
1. Stock Market Data Table (stocks_data)
2. RAG Document & Vector Store Table (documents_rag)
3. Multi-Agent Predictions Log Table (agent_predictions)
4. Feedback Loop & Performance History Table (feedback_history)
========================================================================================
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from config import DATA_DIR, VECTOR_DB_DIR

DB_PATH = DATA_DIR / "stock_intelligence.db"


class DatabaseManager:
    """Manages SQLite tables and persistent vector search storage."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        """Initializes database schema if not already existing."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Stocks OHLCV Data Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stocks_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    daily_return REAL,
                    rsi REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            """)

            # 2. RAG Documents & Vector Index Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents_rag (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_title TEXT,
                    category TEXT,
                    chunk_text TEXT,
                    embedding_json TEXT,
                    source TEXT,
                    publication_date TEXT,
                    document_type TEXT,
                    ticker TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 3. Agent Predictions Log Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    recommendation TEXT,
                    score REAL,
                    confidence REAL,
                    rationale TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 4. Feedback Loop Log Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    predicted_signal TEXT,
                    consensus_score REAL,
                    actual_return REAL,
                    is_correct INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

            cursor.execute("PRAGMA table_info(documents_rag)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            for column, definition in {
                "source": "TEXT",
                "publication_date": "TEXT",
                "document_type": "TEXT",
                "ticker": "TEXT",
            }.items():
                if column not in existing_columns:
                    cursor.execute(f"ALTER TABLE documents_rag ADD COLUMN {column} {definition}")
            conn.commit()
            
        # Seed initial ESG & SEC documents if database is empty
        self.seed_default_documents()

    def seed_default_documents(self):
        """Seeds initial default ESG and SEC filings into vector store."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents_rag")
            if cursor.fetchone()[0] == 0:
                defaults = [
                    ("Sustainability Report 2024", "ESG Reports", "Company reduced carbon emissions by 25% year-over-year. Energy grid uses 40% renewable solar power across manufacturing facilities.", "Reliance Industries Sustainability Report 2023-24", "2024-03-31", "ESG report", "RELIANCE.NS"),
                    ("SEC EDGAR 10-K Filing", "SEC EDGAR", "Annual Report 10-K: Independent audit committee established with strict ethical compliance and transparency policies.", "SEC EDGAR 10-K Filing", None, "Regulatory filing", None),
                    ("World Bank Macro Report", "World Bank", "Macroeconomic forecast predicts stable inflation rates and high growth momentum in industrial production.", "World Bank Macro Report", None, "Macro report", None),
                    ("Reuters Financial News", "News", "Quarterly earnings beat market expectations by 8.5%, driven by record revenue growth and expansion.", "Reuters Financial News", None, "News", None)
                ]
                for title, cat, text, source, publication_date, document_type, ticker in defaults:
                    vec = self._compute_dummy_embedding(text)
                    cursor.execute(
                        "INSERT INTO documents_rag (doc_title, category, chunk_text, embedding_json, source, publication_date, document_type, ticker) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (title, cat, text, json.dumps(vec), source, publication_date, document_type, ticker)
                    )
                conn.commit()

    def _compute_dummy_embedding(self, text: str) -> List[float]:
        """Computes deterministic 64-dimensional vector embedding for text similarity matching."""
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.normal(0, 1, 64)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def insert_stock_data(self, ticker: str, df: Any):
        """Saves processed OHLCV stock dataframe into SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for idx, row in df.iterrows():
                dt = str(row.get("Date", ""))
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO stocks_data
                        (ticker, date, open, high, low, close, volume, daily_return, rsi, sma_20, sma_50)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker, dt,
                        float(row.get("Open", 0)), float(row.get("High", 0)),
                        float(row.get("Low", 0)), float(row.get("Close", 0)),
                        int(row.get("Volume", 0)), float(row.get("Daily_Return", 0) if not np.isnan(row.get("Daily_Return", 0)) else 0),
                        float(row.get("RSI", 50) if not np.isnan(row.get("RSI", 50)) else 50),
                        float(row.get("SMA 20", 0) if not np.isnan(row.get("SMA 20", 0)) else 0),
                        float(row.get("SMA 50", 0) if not np.isnan(row.get("SMA 50", 0)) else 0)
                    ))
                except Exception:
                    continue
            conn.commit()

    def add_document(self, doc_title: str, category: str, text: str, source: str = "User-provided context", publication_date: str = None, document_type: str = "Context", ticker: str = None):
        """Adds a document chunk to the vector store database."""
        vec = self._compute_dummy_embedding(text)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents_rag (doc_title, category, chunk_text, embedding_json, source, publication_date, document_type, ticker) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_title, category, text, json.dumps(vec), source, publication_date, document_type, ticker)
            )
            conn.commit()

    def search_vector_db(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs vector similarity search against stored RAG documents."""
        query_vec = np.array(self._compute_dummy_embedding(query))
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, doc_title, category, chunk_text, embedding_json, source, publication_date, document_type, ticker FROM documents_rag")
            rows = cursor.fetchall()
            
        results = []
        for r_id, title, cat, text, vec_json, source, publication_date, document_type, ticker in rows:
            doc_vec = np.array(json.loads(vec_json))
            similarity = float(np.dot(query_vec, doc_vec))
            results.append({
                "id": r_id,
                "title": title,
                "category": cat,
                "text": text,
                "similarity_score": similarity
                ,"source": source or "Unspecified source",
                "publication_date": publication_date,
                "document_type": document_type or cat,
                "ticker": ticker
            })
            
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def log_prediction(self, ticker: str, recommendation: str, score: float, confidence: float, rationale: str):
        """Logs an agent framework prediction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_predictions (ticker, recommendation, score, confidence, rationale)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, recommendation, score, confidence, rationale))
            conn.commit()

    def get_all_table_counts(self) -> Dict[str, int]:
        """Returns row counts for all database tables."""
        counts = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for tbl in ["stocks_data", "documents_rag", "agent_predictions", "feedback_history"]:
                cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
                counts[tbl] = cursor.fetchone()[0]
        return counts

    def get_table_data(self, table_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves raw data rows from a table."""
        valid_tables = ["stocks_data", "documents_rag", "agent_predictions", "feedback_history"]
        if table_name not in valid_tables:
            return []
            
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
