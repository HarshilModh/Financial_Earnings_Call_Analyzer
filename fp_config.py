# Python 3.11+
# Stevens Username: harshilmodh
#
# Phase 1 — Shared configuration for the Financial Earnings Call Analyzer.
#
# Every other script imports constants from this module. Do not define
# file paths or model names anywhere else.

import os
from pathlib import Path

from dotenv import load_dotenv

# .env lives one level up (the FE 524 course folder)
load_dotenv(Path(__file__).parent.parent / ".env")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise SystemExit("Fatal: OPENAI_API_KEY not found in ../.env")

# ── Models ───────────────────────────────────────────────────────────────────

OPENAI_MODEL       = "gpt-4o-mini"   # generation
OPENAI_JUDGE_MODEL = "gpt-4o-mini"   # LLM-as-judge in evaluation
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"  # sentence-transformers

# ── Companies & filings ──────────────────────────────────────────────────────

COMPANIES: dict[str, str] = {
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
}

FILING_TYPES: list[str] = ["10-K", "10-Q"]

# Date range that captures FY2024 10-K and Q1–Q3 2024 10-Qs for all three
# companies (fiscal years differ: AAPL ends Sep, MSFT ends Jun, GOOGL ends Dec)
FILING_AFTER  = "2023-06-01"
FILING_BEFORE = "2025-06-01"
FILINGS_PER_TYPE = {"10-K": 1, "10-Q": 3}

# ── Chunking ─────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 1500   # characters ≈ 375 tokens for all-MiniLM-L6-v2
CHUNK_OVERLAP = 150    # characters

# ── Retrieval ────────────────────────────────────────────────────────────────

TOP_K = 5   # chunks returned per query

# ── ChromaDB ─────────────────────────────────────────────────────────────────

CHROMA_COLLECTION = "sec_filings"

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data" / "filings"        # sec-edgar-downloader output root
CHROMA_DIR     = BASE_DIR / "data" / "chroma_db"      # persistent ChromaDB storage
GOLDEN_QA_PATH = BASE_DIR / "golden_qa.json"
EVAL_OUT_PATH  = BASE_DIR / "eval_results.json"

# ── SEC EDGAR identity ───────────────────────────────────────────────────────
# SEC ToS requires identifying the requester in the User-Agent header.

SEC_COMPANY_NAME = "Stevens-FE524-Final"
SEC_EMAIL        = "harshilmodh77@gmail.com"
