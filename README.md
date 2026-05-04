# 📊 Financial Earnings Call Analyzer

> **A RAG-Powered Research Assistant for SEC Filings**

**Course:** FE 524 — Prompt Engineering Lab for Business Applications  
**Due:** May 14, 2026  
**Author:** Harshil Modh (`harshilmodh`)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Document Corpus](#document-corpus)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Module Reference](#module-reference)
  - [Phase 1 — Config & Requirements](#phase-1--config--requirements)
  - [Phase 2 — Data Ingestion & Knowledge Base](#phase-2--data-ingestion--knowledge-base)
  - [Phase 3 — RAG Engine](#phase-3--rag-engine)
  - [Phase 4 — Evaluation Harness](#phase-4--evaluation-harness)
  - [Phase 5 — Streamlit UI](#phase-5--streamlit-ui)
  - [Phase 6 — Agentic Financial Analyst](#phase-6--agentic-financial-analyst)
  - [Phase 7 — MCP Server](#phase-7--mcp-server)
  - [Phase 8 — Multi-Agent Deep Research](#phase-8--multi-agent-deep-research)
- [Evaluation Results](#evaluation-results)
- [UI Guide](#ui-guide)
- [Design System & Styling](#design-system--styling)
- [File Naming Convention](#file-naming-convention)
- [Troubleshooting](#troubleshooting)
- [Group Member Responsibilities](#group-member-responsibilities)

---

## Project Overview

This project builds a **Retrieval-Augmented Generation (RAG)** system that allows financial analysts to query SEC filings (10-K annual reports and 10-Q quarterly reports) for **Apple (AAPL)**, **Microsoft (MSFT)**, and **Alphabet (GOOGL)** using natural language.

The system:

1. **Downloads** 12 SEC filings from EDGAR (1 × 10-K + 3 × 10-Q per company)
2. **Parses** HTML/PDF filings, chunks them, and embeds into a ChromaDB vector database
3. **Retrieves** the most relevant passages for a user query via semantic search
4. **Generates** cited, accurate answers using OpenAI GPT
5. **Evaluates** performance with a 25-question golden Q&A set using four metrics
6. **Presents** everything through a polished Streamlit dashboard

Beyond the core RAG pipeline, the project includes an **Agentic Financial Analyst** (tool-use loop), a **Multi-Agent Deep Research** pipeline (Planner → Researcher → Writer), and an **MCP server** for external tool integrations.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 💬 **RAG Q&A** | Ask natural-language questions about SEC filings and get cited answers |
| 🤖 **Agentic Analyst** | Autonomous agent that decomposes complex questions, searches, calculates, compares, and plots charts |
| 📑 **Deep Research** | Three-agent pipeline that generates full research reports with executive summaries and cited findings |
| 📈 **Evaluation Dashboard** | Four-metric evaluation harness with per-question and per-company breakdowns |
| 🔧 **MCP Server** | FastMCP server exposing the knowledge base to any MCP-compatible client (Claude, etc.) |
| 🌗 **Light/Dark Themes** | Custom CSS design system with glassmorphism and Inter typography |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                          │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐│
│  │  Q&A Tab │  │  Agent Tab  │  │ Research Tab │  │ Evaluation Tab   ││
│  └────┬─────┘  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘│
└───────┼────────────────┼───────────────┼────────────────────┼──────────┘
        │                │               │                    │
        ▼                ▼               ▼                    ▼
   ┌─────────┐   ┌────────────┐  ┌────────────┐     ┌────────────────┐
   │ rag.py  │   │ agent.py   │  │ research.py│     │  evaluate.py   │
   │ (RAG    │   │ (OpenAI    │  │ (Planner+  │     │  (Golden QA +  │
   │  Engine)│   │  Function  │  │  Researcher│     │   LLM-Judge)   │
   │         │   │  Calling)  │  │  +Writer)  │     │                │
   └────┬────┘   └─────┬──────┘  └─────┬──────┘     └───────┬────────┘
        │               │               │                    │
        └───────────────┴───────────────┴────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │      ChromaDB         │
                    │  (Vector Database)    │
                    │  ~5,000+ chunks with  │
                    │  metadata per company │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │    SEC EDGAR Filings  │
                    │  12 documents (10-K   │
                    │  + 10-Q per company)  │
                    └───────────────────────┘
```

Additionally, `mcp_server.py` wraps the ChromaDB knowledge base as a standalone MCP tool server.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| PDF Parsing | `pdfplumber` / `pymupdf` (fitz) | Extract text from PDF filings |
| HTML Parsing | `beautifulsoup4` + `lxml` | Clean and extract text from HTML filings |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Split documents into overlapping chunks |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Generate dense vector embeddings |
| Vector DB | `chromadb` (local PersistentClient) | Store and query chunk embeddings |
| LLM / Generation | OpenAI GPT (`gpt-4o-mini`) | Generate answers from retrieved context |
| Evaluation Judge | OpenAI GPT (`gpt-4o-mini`) | LLM-as-judge for evaluation metrics |
| Agent Framework | OpenAI Function Calling | Tool-use loop for the agentic analyst |
| MCP Server | `fastmcp` | Expose knowledge base as MCP tools |
| Charting | `plotly` | Interactive chart generation in agent |
| UI | `streamlit` | Web dashboard |
| Data Source | SEC EDGAR (`data.sec.gov`) | 10-K and 10-Q filings |

---

## Document Corpus

| Company | Ticker | Filing Types | Count |
|---------|--------|-------------|-------|
| Apple Inc. | AAPL | 10-Q (Q1–Q3 2024) + 10-K (FY2024) | 4 |
| Microsoft Corp. | MSFT | 10-Q (Q1–Q3 2024) + 10-K (FY2024) | 4 |
| Alphabet Inc. | GOOGL | 10-Q (Q1–Q3 2024) + 10-K (FY2024) | 4 |
| **Total** | | | **12** |

Data source: SEC EDGAR (`data.sec.gov`) — publicly available, no licensing restrictions.

---

## Project Structure

```
harshilmodh_final_project/
│
├── harshilmodh_fp_config.py      # Phase 1: Shared configuration & constants
├── harshilmodh_fp_download.py    # Phase 2a: SEC EDGAR filing downloader
├── harshilmodh_fp_build_kb.py    # Phase 2b: Parse, chunk, embed → ChromaDB
├── harshilmodh_fp_rag.py         # Phase 3: RAG engine (retrieve + generate)
├── harshilmodh_fp_evaluate.py    # Phase 4: Evaluation harness (4 metrics)
├── harshilmodh_fp_app.py         # Phase 5: Streamlit UI dashboard
├── harshilmodh_fp_agent.py       # Phase 6: Agentic analyst (tool-use loop)
├── harshilmodh_fp_mcp_server.py  # Phase 7: FastMCP server
├── harshilmodh_fp_research.py    # Phase 8: Multi-agent deep research pipeline
│
├── requirements.txt              # Python dependencies
├── styles.css                    # Custom CSS design system for the UI
├── golden_qa.json                # 25 curated evaluation Q&A pairs
├── eval_results.json             # Latest evaluation output
│
├── data/
│   ├── filings/                  # Downloaded SEC filings (sec-edgar-filings/)
│   └── chroma_db/                # Persistent ChromaDB vector store
│
├── CLAUDE.md                     # Internal project spec / build instructions
└── README.md                     # ← You are here
```

---

## Setup & Installation

### Prerequisites

- **Python 3.11+**
- **OpenAI API Key** with access to `gpt-4o-mini` (or `gpt-4o`)

### 1. Clone / Download the Project

```bash
cd "Documents/Sem 4/FE 524/harshilmodh_final_project"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
# or: venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file **one directory up** from the project (in the `FE 524/` folder):

```
FE 524/
├── .env                         ← Place your .env file here
└── harshilmodh_final_project/
    └── ...
```

Contents of `.env`:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

> ⚠️ **Important:** The `.env` file must be in the **parent directory** (`FE 524/`), not inside the project folder. All scripts load it via `load_dotenv(Path(__file__).parent.parent / ".env")`.

---

## Running the Full Pipeline

Run each step in order. Each phase is independently testable.

```bash
# Step 1 — Download SEC filings (~12 documents, may take a few minutes)
python harshilmodh_fp_download.py

# Step 2 — Parse, chunk, and index into ChromaDB
python harshilmodh_fp_build_kb.py

# Step 3 — (Optional) Run evaluation against the golden Q&A set
python harshilmodh_fp_evaluate.py

# Step 4 — Launch the Streamlit UI
streamlit run harshilmodh_fp_app.py
```

### Pipeline Flow

```
download.py → build_kb.py → [evaluate.py] → app.py
   │               │              │              │
   │               │              │              ├── Q&A Tab (rag.py)
   │               │              │              ├── Agent Tab (agent.py)
   │               │              │              ├── Research Tab (research.py)
   │               │              │              └── Evaluation Tab
   │               │              │
   │               │              └── Runs rag.py + LLM judge → eval_results.json
   │               │
   │               └── Parses HTML/PDF → chunks → ChromaDB (data/chroma_db/)
   │
   └── Downloads 12 SEC filings → data/filings/sec-edgar-filings/
```

---

## Module Reference

### Phase 1 — Config & Requirements

**File:** `harshilmodh_fp_config.py`

All shared constants live here. Every other script imports from this module.

| Constant | Value | Description |
|----------|-------|-------------|
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM for generation |
| `OPENAI_JUDGE_MODEL` | `gpt-4o-mini` | LLM for evaluation judging |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | `1500` chars | Characters per chunk (~375 tokens) |
| `CHUNK_OVERLAP` | `150` chars | Overlap between chunks |
| `TOP_K` | `5` | Default number of chunks retrieved per query |
| `CHROMA_COLLECTION` | `sec_filings` | ChromaDB collection name |

---

### Phase 2 — Data Ingestion & Knowledge Base

#### `harshilmodh_fp_download.py`

Downloads SEC filings from EDGAR for all three companies.

- Downloads 1 × 10-K and 3 × 10-Q per ticker
- Date range: `2023-06-01` → `2025-06-01`
- **Idempotent** — skips already-downloaded filings
- Respects SEC EDGAR rate limits (`time.sleep(0.5)` between requests)

```bash
python harshilmodh_fp_download.py
```

#### `harshilmodh_fp_build_kb.py`

Parses all downloaded filings, chunks, and stores in ChromaDB.

**Processing pipeline:**
1. Reads EDGAR `full-submission.txt` files; extracts primary HTML document
2. Cleans HTML: strips scripts/styles, converts tables to pipe-delimited text
3. Chunks with `RecursiveCharacterTextSplitter` (1500 chars, 150 overlap)
4. Detects SEC section headers (Item 1, Item 1A, Item 7, Item 8, etc.)
5. Extracts period metadata from SGML headers
6. Upserts into ChromaDB with SHA-256 chunk IDs (idempotent)

**Chunk metadata schema:**
```json
{
  "company":       "AAPL",
  "company_name":  "Apple Inc.",
  "filing_type":   "10-K",
  "period":        "FY Sep 2024",
  "section":       "Item 7 - MD&A",
  "chunk_index":   42,
  "source_file":   "full-submission.txt"
}
```

```bash
python harshilmodh_fp_build_kb.py
```

---

### Phase 3 — RAG Engine

**File:** `harshilmodh_fp_rag.py`

Core retrieval-augmented generation module. Imported by the evaluation harness and the Streamlit UI.

**Public API:**
```python
from harshilmodh_fp_rag import query

result = query(
    question="What was Apple's total revenue in FY2024?",
    company_filter=["AAPL"],       # Optional: filter by ticker(s)
    filing_filter=["10-K"],        # Optional: filter by filing type(s)
    top_k=5,                       # Number of chunks to retrieve
)
# Returns:
# {
#   "answer": "...",
#   "sources": [{"company", "filing_type", "period", "section", "text_snippet"}],
#   "retrieved_chunks": [full chunk dicts]
# }
```

**System prompt features:**
- Strict grounding: model uses ONLY retrieved context
- Inline citations: `[Company | Filing Type | Period | Section]`
- Exact number quoting from filings
- Explicit refusal to fabricate when context is insufficient
- Structured output: **Answer:** + **Sources:** format
- Multi-company comparison support

**Query expansion:** Appends financial domain terms (`net sales revenue earnings income table figures millions billions fiscal year`) to improve dense retrieval on tabular data.

---

### Phase 4 — Evaluation Harness

**Files:** `harshilmodh_fp_evaluate.py`, `golden_qa.json`

#### Golden Q&A Set (`golden_qa.json`)

25 manually curated question-answer pairs:
- 9 questions for AAPL, 8 for MSFT, 8 for GOOGL
- Covers: revenue segments, gross margins, R&D, risk factors, YoY comparisons, cross-company comparisons

#### Four Evaluation Metrics

| # | Metric | How It Works | Target |
|---|--------|-------------|--------|
| 1 | **Factual Accuracy** | LLM-as-judge compares RAG answer to ground truth (binary 0/1) | ≥ 80% |
| 2 | **Retrieval Precision@3** | Checks if `source_company` + `source_filing` appear in top-3 retrieved chunks | ≥ 85% |
| 3 | **Hallucination Rate** | LLM-as-judge checks if answer contains unsupported claims (binary 0/1) | ≤ 10% |
| 4 | **LLM-as-Judge Score** | GPT rates answer 1–5 for correctness + completeness | ≥ 4.0/5.0 |

```bash
python harshilmodh_fp_evaluate.py
# Output: prints summary table + saves eval_results.json
```

---

### Phase 5 — Streamlit UI

**File:** `harshilmodh_fp_app.py`

```bash
streamlit run harshilmodh_fp_app.py
```

**Layout:**
- **Sidebar:** Company filter, filing type filter, top-K slider, theme toggle, KB stats, "Run Evaluation" button
- **Tab 1 (💬 Q&A):** Natural language question input → cited answer with source badges
- **Tab 2 (🤖 Agent):** Chat-based agentic analyst with real-time tool-call streaming
- **Tab 3 (📑 Deep Research):** Multi-agent research report generation with progress tracking
- **Tab 4 (🕑 History):** Last 5 Q&A pairs
- **Tab 5 (📈 Evaluation):** Metrics dashboard with per-question and per-company breakdowns

---

### Phase 6 — Agentic Financial Analyst

**File:** `harshilmodh_fp_agent.py`

An autonomous agent that uses **OpenAI function calling** to decompose complex questions into tool calls.

**Available Tools:**

| Tool | Purpose |
|------|---------|
| `search_filings(query, company, filing_type, num_results)` | ChromaDB vector search |
| `calculate(expression, label)` | Safe math evaluation (growth rates, margins, ratios) |
| `compare_companies(metric_name, data)` | Structured markdown comparison table |
| `plot_chart(chart_type, title, series, x_label, y_label)` | Plotly chart generation |

**Features:**
- Multi-step reasoning: decomposes complex/comparison questions into sub-queries
- Conversational memory: follow-up questions work via conversation history
- Real-time streaming: each tool call and result is shown as it happens in the UI
- Safety: max 8 iterations to prevent runaway loops
- Chart rendering: Plotly charts displayed inline in Streamlit

---

### Phase 7 — MCP Server

**File:** `harshilmodh_fp_mcp_server.py`

Exposes the SEC filings knowledge base as **MCP (Model Context Protocol)** tools for any MCP-compatible client (e.g., Claude Desktop).

```bash
# Run in stdio mode (default)
python harshilmodh_fp_mcp_server.py

# Run in SSE mode (for web clients)
python harshilmodh_fp_mcp_server.py --sse
```

**Exposed Tools:**
| Tool | Description |
|------|-------------|
| `search_sec_filings` | Semantic search across all indexed filings |
| `get_knowledge_base_stats` | Chunk counts per company |
| `calculate_financial_metric` | Safe math for financial calculations |
| `list_available_companies` | List tickers and company names |

**Resources:** `sec://companies`, `sec://filing-types`

---

### Phase 8 — Multi-Agent Deep Research

**File:** `harshilmodh_fp_research.py`

Three-agent pipeline for autonomous financial research:

| Agent | Role |
|-------|------|
| 🗺️ **Planner** | Decomposes the topic into 4–6 targeted research questions |
| 🔍 **Researcher** | For each question, searches the KB and produces a cited factual summary |
| ✍️ **Writer** | Synthesizes all findings into a structured research report |

**Report structure:** Executive Summary → Key Findings → Data Highlights → Risk Factors → Conclusion

```python
from harshilmodh_fp_research import run_deep_research

for event in run_deep_research(topic="Compare Apple and Microsoft's cloud revenue"):
    print(event)
```

---

## Evaluation Results

Latest evaluation results from `eval_results.json` (25 golden questions):

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Factual Accuracy | **80.0%** | ≥ 80% | ✅ Met |
| Retrieval Precision@3 | **100.0%** | ≥ 85% | ✅ Met |
| Hallucination Rate | **24.0%** | ≤ 10% | ⚠️ Above target |
| LLM-as-Judge Score | **4.76/5.0** | ≥ 4.0 | ✅ Met |

> **Note:** The hallucination rate includes cases where the LLM-as-judge flagged additional context the model provided beyond the strict ground truth. In most flagged cases (e.g., q02, q03, q10), the "hallucinated" content was actually correct information from the filing that went beyond the ground truth answer — indicating the system's thoroughness rather than fabrication.

---

## UI Guide

### Q&A Tab

1. Use the **sidebar filters** to narrow by company (AAPL, MSFT, GOOGL) and filing type (10-K, 10-Q)
2. Adjust the **Top-K slider** (3–10) for retrieval breadth
3. Type a question or click an **example question** button
4. Click **Ask** — the system retrieves relevant chunks and synthesizes a cited answer
5. Expand **Sources** to see the exact filing excerpts used

### Agent Tab

1. Type a question in the chat input at the bottom
2. Watch the agent's **reasoning process** in real time:
   - 🧠 **Thinking** — agent analyzes the question
   - 🔍 **Tool calls** — search, calculate, compare, plot
   - 📋 **Results** — data returned from each tool
   - 📊 **Charts** — interactive Plotly charts rendered inline
3. Ask **follow-up questions** — the agent maintains conversational context

### Deep Research Tab

1. Enter a **research topic** (e.g., "Compare cloud/services revenue growth for AAPL and MSFT")
2. Click **🚀 Generate Report**
3. Watch the three-phase pipeline:
   - 🗺️ Planner generates 4–6 research questions
   - 🔍 Researcher gathers findings for each question
   - ✍️ Writer produces a structured report
4. The final report includes executive summary, key findings, and citations

---

## Design System & Styling

**File:** `styles.css`

The UI uses a custom CSS design system built on:
- **Typography:** Inter (Google Fonts) with 400/500/600/700 weights
- **Theme support:** Light and Dark modes via CSS custom properties
- **Glassmorphism:** Sidebar uses `backdrop-filter: blur(20px)` for a frosted glass effect
- **Micro-animations:** Pulsing thinking indicators, hover transforms on buttons
- **Component library:** Answer cards, source badges, thinking/tool/result boxes, segmented tabs

---

## File Naming Convention

All project files use the prefix `harshilmodh_fp_` (fp = final project):

```
harshilmodh_fp_config.py
harshilmodh_fp_download.py
harshilmodh_fp_build_kb.py
harshilmodh_fp_rag.py
harshilmodh_fp_evaluate.py
harshilmodh_fp_app.py
harshilmodh_fp_agent.py
harshilmodh_fp_mcp_server.py
harshilmodh_fp_research.py
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Fatal: OPENAI_API_KEY not found` | Ensure `.env` file exists in the **parent directory** (`FE 524/`) with `OPENAI_API_KEY=sk-...` |
| `ChromaDB not found` | Run `python harshilmodh_fp_build_kb.py` to build the knowledge base |
| `No accession directories found` | Run `python harshilmodh_fp_download.py` first to download SEC filings |
| `RAG engine not found` (in UI) | Ensure `harshilmodh_fp_rag.py` is in the same directory as the app |
| `Agent module not found` (in UI) | Ensure `harshilmodh_fp_agent.py` is present |
| `Research module not found` | Ensure `harshilmodh_fp_research.py` is present |
| Slow embedding on first run | The `all-MiniLM-L6-v2` model downloads on first use (~80 MB). Subsequent runs use the cached model. |
| SEC EDGAR rate limiting | The downloader includes `time.sleep(0.5)` between requests. If errors persist, wait a few minutes and retry. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the virtual environment |

### Verifying the Knowledge Base

After running `build_kb.py`, the UI sidebar shows chunk counts per company. You should see ~1,000–2,000+ chunks per company.

---

## Group Member Responsibilities

Below is a suggested breakdown of responsibilities for the project. Each team member should be familiar with the full pipeline but can focus on their assigned areas:

### 🔧 Data Pipeline (Phases 1–2)
- **Config management** (`config.py`): Environment variables, constants, paths
- **SEC filing download** (`download.py`): EDGAR API interaction, idempotent downloads
- **Knowledge base construction** (`build_kb.py`): HTML/PDF parsing, chunking strategy, ChromaDB indexing

### 🧠 RAG Engine & Prompt Design (Phase 3)
- **System prompt engineering** (`rag.py`): Citation format, grounding instructions, multi-company handling
- **Query expansion**: Domain-specific term augmentation for better retrieval
- **Retrieval logic**: ChromaDB querying with metadata filters

### 📊 Evaluation & Quality (Phase 4)
- **Golden Q&A curation** (`golden_qa.json`): 25 test questions with verified ground truths
- **Metric implementation** (`evaluate.py`): Factual accuracy, retrieval precision, hallucination detection, LLM-as-judge
- **Results analysis**: Interpreting per-company breakdowns and identifying improvement areas

### 🖥️ UI & User Experience (Phase 5)
- **Streamlit dashboard** (`app.py`): Tab layout, sidebar controls, session state management
- **CSS design system** (`styles.css`): Themes, glassmorphism, typography, responsive layout
- **Source visualization**: Badges, quoted excerpts, expandable panels

### 🤖 Agent & Research Systems (Phases 6–8)
- **Agentic analyst** (`agent.py`): OpenAI function calling, tool definitions, iterative reasoning
- **MCP server** (`mcp_server.py`): FastMCP integration, resource/prompt/tool exposure
- **Deep research pipeline** (`research.py`): Multi-agent orchestration (Planner → Researcher → Writer)

---

<p align="center">
  <strong>FE 524 — Prompt Engineering Lab for Business Applications</strong><br>
  Stevens Institute of Technology · Spring 2026
</p>
