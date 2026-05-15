# Config for the Financial Earnings Analyzer project.
# All other modules import constants from here.

import os
from pathlib import Path
from dotenv import load_dotenv

# .env is one folder up (the FE 524 course directory)
load_dotenv(Path(__file__).parent.parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not found — add it to ../.env")

# Models
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Companies we're analyzing
COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
}

FILING_TYPES = ["10-K", "10-Q"]

# Wide date range so we catch FY2024 filings for all three companies
# (their fiscal years end at different months)
FILING_AFTER = "2023-06-01"
FILING_BEFORE = "2025-06-01"
FILINGS_PER_TYPE = {"10-K": 1, "10-Q": 3}

# Chunking settings
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

TOP_K = 5

CHROMA_COLLECTION = "sec_filings"

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "filings"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
GOLDEN_QA_PATH = BASE_DIR / "golden_qa.json"
EVAL_OUT_PATH = BASE_DIR / "eval_results.json"

# SEC EDGAR requires a user-agent with contact info
SEC_COMPANY_NAME = "Stevens-FE524-Final"
SEC_EMAIL = "harshilmodh77@gmail.com"
