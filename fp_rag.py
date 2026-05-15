# RAG engine — retrieve chunks from ChromaDB and generate answers via OpenAI.

from __future__ import annotations

import textwrap
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

from fp_config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
)


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



def _build_where_filter(
    company_filter: list[str] | None,
    filing_filter: list[str] | None,
) -> dict | None:
    """Build a ChromaDB metadata filter from optional lists."""
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
    """Append financial search terms to improve retrieval."""
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



_SYSTEM_PROMPT = (
    "You are a financial research assistant that answers questions about SEC filings "
    "(10-K annual reports and 10-Q quarterly reports).\n\n"
    "Important rules:\n"
    "- Only use the context excerpts provided. Do not use outside knowledge.\n"
    "- Cite every claim like this: [AAPL | 10-K | FY Sep 2024 | Item 7 - MD&A]\n"
    "- Copy numbers and percentages exactly as they appear in the filing, don't round.\n"
    "- If the context doesn't have enough info to answer, just say so. Don't make things up.\n"
    "- For questions comparing multiple companies, go company by company.\n\n"
    "Format your response as:\n"
    "**Answer:**\n"
    "<your answer with inline citations>\n\n"
    "**Sources:**\n"
    "- [Company | Filing Type | Period | Section]: \"<short quote from that chunk>\"\n"
)


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



if __name__ == "__main__":
    test_q = "What was Apple's total revenue for fiscal year 2024?"
    print(f"Question: {test_q}\n{'─'*60}")
    result = query(test_q, company_filter=["AAPL"], filing_filter=["10-K"])
    print(result["answer"])
    print(f"\n{'─'*60}\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  • {s['company']} | {s['filing_type']} | {s['period']} | {s['section']}")
        print(f"    {s['text_snippet'][:120]}…")
