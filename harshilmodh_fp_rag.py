# Python 3.11+
# Stevens Username: harshilmodh
#
# Phase 3 — RAG engine for the Financial Earnings Call Analyzer.
#
# Public API: query()
# Imported by harshilmodh_fp_evaluate.py and harshilmodh_fp_app.py.

from __future__ import annotations

import textwrap
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

from harshilmodh_fp_config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
)

# ── Singletons (lazy-initialised once per process) ───────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None
_openai_client: Optional[OpenAI] = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        ef = SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_collection(
            CHROMA_COLLECTION, embedding_function=ef
        )
    return _collection


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _build_where_filter(
    company_filter: list[str] | None,
    filing_filter: list[str] | None,
) -> dict | None:
    """Build a ChromaDB $and/$or metadata filter from optional lists."""
    clauses: list[dict] = []

    if company_filter:
        if len(company_filter) == 1:
            clauses.append({"company": {"$eq": company_filter[0]}})
        else:
            clauses.append({"company": {"$in": company_filter}})

    if filing_filter:
        if len(filing_filter) == 1:
            clauses.append({"filing_type": {"$eq": filing_filter[0]}})
        else:
            clauses.append({"filing_type": {"$in": filing_filter}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _expand_query(question: str) -> str:
    """Append financial-domain search terms to improve dense retrieval on tabular data."""
    return (
        f"{question} "
        "net sales revenue earnings income table figures millions billions fiscal year"
    )


def retrieve(
    question: str,
    company_filter: list[str] | None = None,
    filing_filter: list[str] | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """Return top-k chunk dicts from ChromaDB for *question*."""
    col = _get_collection()
    where = _build_where_filter(company_filter, filing_filter)

    search_text = _expand_query(question)
    kwargs: dict = dict(query_texts=[search_text], n_results=top_k, include=["documents", "metadatas"])
    if where is not None:
        kwargs["where"] = where

    results = col.query(**kwargs)

    chunks: list[dict] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    for doc, meta in zip(docs, metas):
        chunks.append({**meta, "text": doc})
    return chunks


# ── Prompt construction ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a financial research assistant specialising in SEC filings (10-K and 10-Q).
    Your answers are grounded strictly in the context excerpts provided below.

    Rules you MUST follow:
    1. Use ONLY the information in the provided context excerpts. Do not draw on outside knowledge.
    2. Every factual claim must include an inline citation in the exact format:
       [Company | Filing Type | Period | Section]
       Example: [AAPL | 10-K | FY Sep 2024 | Item 7 - MD&A]
    3. When quoting specific numbers, percentages, or dollar figures, reproduce them exactly
       as they appear in the filing — do not paraphrase or round.
    4. If the context does not contain sufficient information to answer the question, say:
       "The provided context does not contain enough information to answer this question."
       Do NOT speculate or fabricate.
    5. For multi-company comparison questions, organise your response by company.
    6. Structure your response as follows:

       **Answer:**
       <your detailed, cited answer>

       **Sources:**
       <for each source used, one line:>
       - [Company | Filing Type | Period | Section]: "<exact short quote from that chunk>"
""")


def _build_user_message(question: str, chunks: list[dict]) -> str:
    parts = ["### Retrieved context excerpts\n"]
    for i, chunk in enumerate(chunks, 1):
        meta = (
            f"{chunk.get('company_name', chunk.get('company'))} | "
            f"{chunk.get('filing_type')} | "
            f"{chunk.get('period')} | "
            f"{chunk.get('section', 'Unknown section')}"
        )
        parts.append(f"[Excerpt {i} — {meta}]\n{chunk['text']}\n")
    parts.append(f"\n### Question\n{question}")
    return "\n".join(parts)


# ── Generation ────────────────────────────────────────────────────────────────

def _generate(question: str, chunks: list[dict]) -> str:
    client = _get_openai()
    user_msg = _build_user_message(question, chunks)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ── Source extraction ─────────────────────────────────────────────────────────

def _extract_sources(chunks: list[dict]) -> list[dict]:
    """Build the sources list from retrieved chunks (top-3 shown in UI)."""
    return [
        {
            "company":      chunk.get("company"),
            "company_name": chunk.get("company_name"),
            "filing_type":  chunk.get("filing_type"),
            "period":       chunk.get("period"),
            "section":      chunk.get("section"),
            "text_snippet": chunk["text"][:300].rstrip() + "…",
        }
        for chunk in chunks
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def query(
    question: str,
    company_filter: list[str] | None = None,
    filing_filter: list[str] | None = None,
    top_k: int = TOP_K,
) -> dict:
    """
    Run a RAG query against the SEC filings knowledge base.

    Returns
    -------
    {
        "answer":           str,
        "sources":          [{"company", "company_name", "filing_type",
                               "period", "section", "text_snippet"}],
        "retrieved_chunks": [full chunk dicts with "text" + all metadata]
    }
    """
    chunks = retrieve(question, company_filter, filing_filter, top_k)
    if not chunks:
        return {
            "answer": "No relevant documents were found for the given filters.",
            "sources": [],
            "retrieved_chunks": [],
        }

    answer = _generate(question, chunks)
    sources = _extract_sources(chunks)

    return {
        "answer":           answer,
        "sources":          sources,
        "retrieved_chunks": chunks,
    }


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_q = "What was Apple's total revenue for fiscal year 2024?"
    print(f"Question: {test_q}\n{'─'*60}")
    result = query(test_q, company_filter=["AAPL"], filing_filter=["10-K"])
    print(result["answer"])
    print(f"\n{'─'*60}\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  • {s['company']} | {s['filing_type']} | {s['period']} | {s['section']}")
        print(f"    {s['text_snippet'][:120]}…")
