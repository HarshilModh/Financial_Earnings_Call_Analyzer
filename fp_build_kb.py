# Python 3.11+
# Stevens Username: harshilmodh
#
# Phase 2b — Parse SEC filings, chunk, embed, and index into ChromaDB.
#
# sec-edgar-downloader saves each accession as a single full-submission.txt
# (EDGAR SGML envelope). This script:
#   1. Reads full-submission.txt; extracts the period from the SGML header
#      and the primary HTML document from the first <TEXT>…</TEXT> block
#   2. Falls back to individual .htm/.pdf files if present
#   3. Cleans text (tables → pipe-delimited, scripts stripped)
#   4. Chunks with LangChain RecursiveCharacterTextSplitter
#   5. Tags each chunk with SEC section metadata via regex scan
#   6. Upserts into a local ChromaDB collection (SHA-256 chunk ID → idempotent)
#
# Run after fp_download.py:
#   python fp_build_kb.py

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

import chromadb
import pdfplumber
from bs4 import BeautifulSoup
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from fp_config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COMPANIES,
    DATA_DIR,
    EMBEDDING_MODEL,
    FILING_TYPES,
)

# ── ChromaDB batch size ───────────────────────────────────────────────────────

UPSERT_BATCH = 50

# ── SEC section-header patterns (checked in order; first match wins) ──────────
# Each entry: (friendly label, compiled regex)

_RAW_SECTIONS = [
    ("Item 1A - Risk Factors",                 r"\bitem\s+1a\b"),
    ("Item 1B - Unresolved Staff Comments",    r"\bitem\s+1b\b"),
    ("Item 1 - Business",                      r"\bitem\s+1\b"),
    ("Item 2 - Properties",                    r"\bitem\s+2\b"),
    ("Item 3 - Legal Proceedings",             r"\bitem\s+3\b"),
    ("Item 5 - Market for Registrant",         r"\bitem\s+5\b"),
    ("Item 6 - Selected Financial Data",       r"\bitem\s+6\b"),
    ("Item 7A - Quantitative Disclosures",     r"\bitem\s+7a\b"),
    ("Item 7 - MD&A",                          r"\bitem\s+7\b"),
    ("Item 8 - Financial Statements",          r"\bitem\s+8\b"),
    ("Item 9A - Controls and Procedures",      r"\bitem\s+9a\b"),
    ("Item 9 - Changes in Accountants",        r"\bitem\s+9\b"),
    ("Item 15 - Exhibits",                     r"\bitem\s+15\b"),
    ("Part II",                                r"\bpart\s+ii\b"),
    ("Part I",                                 r"\bpart\s+i\b"),
]
SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in _RAW_SECTIONS
]

# Patterns to extract the reporting period from the filing cover page
_PERIOD_RE = re.compile(
    r"(?:quarterly\s+period|annual\s+period|fiscal\s+year|quarter|year|"
    r"nine\s+months|six\s+months|three\s+months)\s+ended\s+"
    r"([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)


# ── Text extraction ───────────────────────────────────────────────────────────

def _table_to_text(table_tag) -> str:
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _html_to_text(html_bytes_or_str) -> str:
    if isinstance(html_bytes_or_str, str):
        soup = BeautifulSoup(html_bytes_or_str, "lxml")
    else:
        soup = BeautifulSoup(html_bytes_or_str, "lxml")

    for tag in soup(["script", "style", "head", "meta", "link", "noscript"]):
        tag.decompose()

    for table in soup.find_all("table"):
        table_text = _table_to_text(table)
        table.replace_with("\n" + table_text + "\n")

    text = "\n".join(soup.stripped_strings)
    return re.sub(r"\n{3,}", "\n\n", text)


def parse_html(path: Path) -> str:
    return _html_to_text(path.read_bytes())


def parse_pdf(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def parse_submission(path: Path) -> tuple[str, str]:
    """
    Parse an EDGAR full-submission.txt file.
    Returns (period_str, cleaned_text) by extracting the primary document
    from the first <TEXT>…</TEXT> SGML block.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Period from SGML header (format: YYYYMMDD)
    period = _period_from_sgml_header(raw, path.parent.parent.name)

    # Primary document lives in the first <TEXT>…</TEXT> block
    text_start = raw.find("<TEXT>")
    text_end   = raw.find("</TEXT>")
    if text_start == -1 or text_end == -1:
        return period, re.sub(r"\n{3,}", "\n\n", raw)

    html_fragment = raw[text_start + 6 : text_end]
    # Strip EDGAR XBRL wrapper tags (keep inner HTML intact)
    html_fragment = re.sub(r"</?XBRL[^>]*>", "", html_fragment, flags=re.IGNORECASE)

    return period, _html_to_text(html_fragment)


def _period_from_sgml_header(raw: str, form_type: str) -> str:
    m = re.search(r"CONFORMED PERIOD OF REPORT:\s+(\d{8})", raw)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
            suffix = "FY" if form_type == "10-K" else "Q ended"
            return f"{suffix} {dt.strftime('%b %Y')}"
        except ValueError:
            pass
    return "FY2024" if form_type == "10-K" else "2024"


def extract_text(path: Path) -> tuple[str, str]:
    """Returns (period, cleaned_text) for any supported file type."""
    if path.name == "full-submission.txt":
        return parse_submission(path)
    suffix = path.suffix.lower()
    if suffix in {".htm", ".html"}:
        text = parse_html(path)
    elif suffix == ".pdf":
        text = parse_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    # Fall back to text-based period detection for standalone HTML/PDF
    m = _PERIOD_RE.search(text[:8000])
    period = m.group(1).strip() if m else "2024"
    return period, text


# ── Metadata helpers ──────────────────────────────────────────────────────────


def detect_section(chunk_text: str) -> str:
    best_pos, best_label = -1, "General"
    for label, pattern in SECTION_PATTERNS:
        for m in pattern.finditer(chunk_text):
            if m.start() > best_pos:
                best_pos, best_label = m.start(), label
    return best_label


def chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


# ── Primary-document locator ──────────────────────────────────────────────────

def find_primary_doc(accession_dir: Path) -> Path | None:
    # Prefer the EDGAR full-submission envelope (always present from downloader)
    full_sub = accession_dir / "full-submission.txt"
    if full_sub.exists() and full_sub.stat().st_size > 50_000:
        return full_sub

    # Fall back to individual HTML / PDF files (e.g., manually placed filings)
    candidates = (
        list(accession_dir.glob("*.htm"))
        + list(accession_dir.glob("*.html"))
        + list(accession_dir.glob("*.pdf"))
    )
    candidates = [f for f in candidates if f.stat().st_size > 50_000]
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.stat().st_size)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_kb() -> None:
    filings_root = DATA_DIR / "sec-edgar-filings"
    if not filings_root.exists():
        raise SystemExit(
            f"Filings directory not found: {filings_root}\n"
            "Run  python fp_download.py  first."
        )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    summary: dict[str, int] = {t: 0 for t in COMPANIES}

    # Collect all accession dirs to process
    work_items: list[tuple[str, str, Path]] = []
    for ticker in COMPANIES:
        for form_type in FILING_TYPES:
            form_dir = filings_root / ticker / form_type
            if not form_dir.exists():
                continue
            for acc_dir in sorted(form_dir.iterdir()):
                if acc_dir.is_dir():
                    work_items.append((ticker, form_type, acc_dir))

    if not work_items:
        raise SystemExit("No accession directories found. Run the downloader first.")

    print(f"\nIndexing {len(work_items)} filing(s) into ChromaDB...")

    for ticker, form_type, acc_dir in tqdm(work_items, desc="Filings", unit="filing"):
        primary = find_primary_doc(acc_dir)
        if primary is None:
            tqdm.write(f"  WARN  No primary doc in {acc_dir.name} — skipping")
            continue

        tqdm.write(f"  {ticker} {form_type:5s}  {primary.name}  ({primary.stat().st_size // 1024} KB)")

        try:
            period, text = extract_text(primary)
        except Exception as exc:
            tqdm.write(f"  ERROR parsing {primary}: {exc}")
            continue
        chunks = splitter.split_text(text)

        ids, documents, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            doc_id = chunk_id(chunk)
            section = detect_section(chunk)
            meta = {
                "company":       ticker,
                "company_name":  COMPANIES[ticker],
                "filing_type":   form_type,
                "period":        period,
                "section":       section,
                "chunk_index":   i,
                "source_file":   primary.name,
            }
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append(meta)

        # Upsert in batches
        for start in range(0, len(ids), UPSERT_BATCH):
            collection.upsert(
                ids=ids[start : start + UPSERT_BATCH],
                documents=documents[start : start + UPSERT_BATCH],
                metadatas=metadatas[start : start + UPSERT_BATCH],
            )

        summary[ticker] += len(chunks)
        tqdm.write(f"         → {len(chunks)} chunks indexed  (period: {period})")

    print("\n── Knowledge Base Summary ──────────────────────────")
    total = 0
    for ticker, count in summary.items():
        print(f"  {ticker:6s} ({COMPANIES[ticker]}): {count:>5} chunks")
        total += count
    print(f"  {'TOTAL':6s}                   : {total:>5} chunks")
    print(f"\nChromaDB persisted at: {CHROMA_DIR}")


if __name__ == "__main__":
    build_kb()
