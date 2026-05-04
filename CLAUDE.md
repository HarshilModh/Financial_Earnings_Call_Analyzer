# FE 524 Final Project — Financial Earnings Call Analyzer
## A RAG-Powered Research Assistant

**Course:** FE 524 — Prompt Engineering Lab for Business Applications
**Due:** May 14, 2026
**Stevens Username:** harshilmodh

---

## Project Overview

Build a Retrieval-Augmented Generation (RAG) system that lets financial analysts query SEC filings
(10-K + 10-Q) for Apple (AAPL), Microsoft (MSFT), and Alphabet (GOOGL) using natural language.
The system retrieves relevant document chunks and synthesizes cited, accurate answers via OpenAI GPT.

---

## File Naming Convention

All project files use the prefix `fp_` (fp = final project). Example:
- `fp_download.py`
- `fp_build_kb.py`
- `fp_rag.py`
- `fp_evaluate.py`
- `fp_app.py`

---

## Environment

- **Python:** 3.11+
- **`.env` file location:** one directory up — `../env` or `../.env` (the FE 524 folder root)
- **Required env vars:** `OPENAI_API_KEY`
- Load with: `load_dotenv(Path(__file__).parent.parent / ".env")`

---

## Tech Stack

| Component       | Technology                                      |
|-----------------|-------------------------------------------------|
| PDF Parsing     | `pdfplumber` / `pymupdf` (fitz)                 |
| HTML Parsing    | `beautifulsoup4` + `lxml`                       |
| Chunking        | LangChain `RecursiveCharacterTextSplitter`      |
| Embeddings      | `sentence-transformers` (all-MiniLM-L6-v2)      |
| Vector DB       | `chromadb` (local PersistentClient)             |
| LLM / Generation| OpenAI GPT via `openai` Python SDK             |
| Evaluation      | Custom golden set + LLM-as-judge (GPT-4o)      |
| UI              | `streamlit`                                     |

---

## Document Corpus

| Company        | Ticker | Filing Types                        | Target # |
|----------------|--------|-------------------------------------|----------|
| Apple Inc.     | AAPL   | 10-Q (Q1–Q3 2024) + 10-K (FY2024)  | 4        |
| Microsoft Corp.| MSFT   | 10-Q (Q1–Q3 2024) + 10-K (FY2024)  | 4        |
| Alphabet Inc.  | GOOGL  | 10-Q (Q1–Q3 2024) + 10-K (FY2024)  | 4        |
| **Total**      |        |                                     | **12**   |

Data source: SEC EDGAR (`data.sec.gov`) — publicly available, no licensing restrictions.

---

## Chunk Metadata Schema

Every chunk stored in ChromaDB must carry this metadata:
```json
{
  "company":       "AAPL",
  "company_name":  "Apple Inc.",
  "filing_type":   "10-K",
  "period":        "FY2024",
  "section":       "Item 7 - MD&A",
  "chunk_index":   42,
  "source_file":   "primary-document.htm"
}
```

---

## Evaluation Targets

| Metric                    | Target     |
|---------------------------|------------|
| Factual accuracy          | ≥ 80%      |
| Retrieval precision (top-3)| ≥ 85%     |
| Hallucination rate        | ≤ 10%      |
| LLM-as-judge score        | ≥ 4.0 / 5.0|

---

## Phases

### Phase 1 — Config & Requirements
**File:** `fp_config.py`, `requirements.txt`

All shared constants live here. Every other script imports from `config.py`.

Constants to define:
- `OPENAI_API_KEY` — loaded from `../.env`
- `OPENAI_MODEL = "gpt-4o-mini"` — generation model
- `OPENAI_JUDGE_MODEL = "gpt-4o"` — evaluation judge model
- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` — sentence-transformers model
- `COMPANIES = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc."}`
- `FILING_TYPES = ["10-K", "10-Q"]`
- `CHUNK_SIZE = 1500` — characters (≈ 375 tokens for all-MiniLM-L6-v2)
- `CHUNK_OVERLAP = 150` — characters
- `TOP_K = 5` — chunks to retrieve per query
- `CHROMA_COLLECTION = "sec_filings"`
- `DATA_DIR = BASE_DIR / "data" / "filings"`
- `CHROMA_DIR = BASE_DIR / "data" / "chroma_db"`
- `GOLDEN_QA_PATH = BASE_DIR / "golden_qa.json"`
- `SEC_EMAIL = "harshilmodh77@gmail.com"`

**Status:** [ ] Not started

---

### Phase 2 — Data Ingestion & Chunking Pipeline
**File:** `fp_download.py`

Download SEC EDGAR filings using `sec-edgar-downloader`.
- For each ticker in `COMPANIES`: download 1 × 10-K and 3 × 10-Q
- Date range: `after="2023-06-01"`, `before="2025-06-01"` (covers FY2024 for all three companies)
- Output directory: `DATA_DIR` (files land in `DATA_DIR/sec-edgar-filings/<TICKER>/<FORM>/`)
- Print progress per company/form-type
- Skip tickers that already have files (idempotent re-runs)
- Respect SEC EDGAR rate limits: add `time.sleep(0.5)` between requests

**File:** `fp_build_kb.py`

Parse all downloaded filings, chunk, and store in ChromaDB. Run once after download.

Steps:
1. Walk `DATA_DIR/sec-edgar-filings/` recursively to find primary documents
2. Parse HTML with BeautifulSoup (`lxml` parser); parse PDF with `pdfplumber` if `.pdf`
3. Detect section headers (Item 1, Item 1A, Item 7, Item 8, etc.) to populate `section` metadata
4. Clean text: strip scripts/styles, collapse whitespace, convert tables to pipe-delimited text
5. Chunk using `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)`
6. Extract metadata from directory path: ticker → company, form type, accession date → period
7. Embed chunks using `SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")` via ChromaDB
8. Upsert into ChromaDB collection (idempotent — use chunk hash as document ID)
9. Print summary: N chunks indexed per company

**Status:** [ ] Not started

---

### Phase 3 — RAG Prompt Design & OpenAI API Integration
**File:** `fp_rag.py`

Core RAG module — imported by both the evaluation harness and the Streamlit UI.

Public API:
```python
def query(
    question: str,
    company_filter: list[str] | None = None,   # e.g. ["AAPL", "MSFT"]
    filing_filter: list[str] | None = None,    # e.g. ["10-K"]
    top_k: int = TOP_K,
) -> dict:
    # Returns:
    # {
    #   "answer": str,
    #   "sources": [{"company", "filing_type", "period", "section", "text_snippet"}],
    #   "retrieved_chunks": [full chunk dicts]
    # }
```

System prompt requirements (Harshil's role — RAG Prompt Design):
- Instruct the model to use ONLY the provided context excerpts
- Require inline citations: `[Company | Filing Type | Period | Section]`
- Require exact number/figure quoting from the filing
- Explicit instruction: if context is insufficient, say so — never fabricate
- Structured output: **Answer:** section followed by **Sources:** section with quoted excerpts
- Handle multi-company comparison questions by organizing response by company

**Status:** [ ] Not started

---

### Phase 4 — Evaluation Harness
**Files:** `fp_evaluate.py`, `golden_qa.json`

**`golden_qa.json`** — 25 manually curated Q&A pairs:
- 8–9 questions per company (AAPL, MSFT, GOOGL)
- Cover: gross margin, revenue segments, R&D spend, risk factors, forward guidance, YoY comparisons, cross-company comparisons
- Each entry:
  ```json
  {
    "id": "q01",
    "question": "...",
    "ground_truth": "...",
    "source_company": "AAPL",
    "source_filing": "10-K",
    "source_section": "Item 7 - MD&A"
  }
  ```

**`fp_evaluate.py`** — runs four metrics:

1. **Factual Accuracy** — LLM-as-judge: does the RAG answer match ground truth? Binary 0/1 per question.
2. **Retrieval Precision** — was `source_company` + `source_filing` present in the top-3 retrieved chunks? Binary 0/1.
3. **Hallucination Rate** — LLM-as-judge: does the answer contain any claim not supported by retrieved chunks? Binary.
4. **LLM-as-Judge Score** — GPT-4o rates answer vs ground truth 1–5 for correctness + completeness.

Output: prints a summary table and saves `eval_results.json` with per-question details.

**Status:** [ ] Not started

---

### Phase 5 — Streamlit UI
**File:** `fp_app.py`

Run with: `streamlit run fp_app.py`

Layout:
- **Sidebar:**
  - Company multi-select filter (AAPL / MSFT / GOOGL / All)
  - Filing type filter (10-K / 10-Q / All)
  - Top-K slider (3–10)
  - "Run Evaluation" button → shows metrics table
- **Main area:**
  - Title + brief description
  - Text input for natural language question
  - "Ask" button
  - Answer panel (formatted markdown)
  - Expandable "Sources" section showing retrieved chunks with metadata badges
  - Query history (last 5 Q&A pairs in session state)

**Status:** [ ] Not started

---

### Phase 6 — Agentic Financial Analyst (Tool-Use Loop)
**File:** `fp_agent.py`

An autonomous agent that uses OpenAI function calling to decompose complex questions
into tool calls, iterate until it has sufficient information, and produce cited answers.

Tools available to the agent:
- `search_filings(query, company, filing_type, num_results)` — ChromaDB vector search
- `calculate(expression, label)` — safe math evaluation for growth rates, margins, ratios
- `compare_companies(metric_name, data)` — structured markdown comparison table
- `plot_chart(chart_type, title, series, x_label, y_label)` — Plotly chart generation

Features:
- Multi-step reasoning: agent decomposes complex/comparison questions into sub-queries
- Conversational memory: follow-up questions work via conversation history
- Real-time streaming UI: each tool call and result is shown as it happens
- Safety: max 8 iterations to prevent runaway loops
- Chart rendering: Plotly charts displayed inline in Streamlit

**Status:** [ ] Not started

---

## Build Order

```
Phase 1 (config + requirements)
    ↓
Phase 2 (download → build_kb)
    ↓
Phase 3 (rag engine)
    ↓
Phase 4 (golden_qa.json → evaluate)
    ↓
Phase 5 (streamlit app)
    ↓
Phase 6 (agentic analyst)
```

Each phase is independently testable before moving to the next.

---

## Running the Full Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download SEC filings (~12 documents, may take a few minutes)
python fp_download.py

# 3. Parse, chunk, and index into ChromaDB
python fp_build_kb.py

# 4. (Optional) Run evaluation against golden Q&A set
python fp_evaluate.py

# 5. Launch the Streamlit UI
streamlit run fp_app.py
```
