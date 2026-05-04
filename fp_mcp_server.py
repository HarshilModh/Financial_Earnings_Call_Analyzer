# Python 3.11+
# Stevens Username: harshilmodh
#
# Phase 7 — FastMCP Server for the Financial Earnings Call Analyzer.
#
# Exposes the SEC filings knowledge base as MCP tools so that any
# MCP-compatible client (Claude, custom agents, etc.) can query it.
#
# Run:
#   python fp_mcp_server.py          (stdio mode)
#   python fp_mcp_server.py --sse     (SSE mode for web)

from __future__ import annotations

import json
import math
import re
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastmcp import FastMCP

from fp_config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    COMPANIES,
    EMBEDDING_MODEL,
    FILING_TYPES,
)

# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "SEC Filing Analyst",
    description=(
        "A financial research MCP server that provides tools to search and "
        "analyze SEC filings (10-K / 10-Q) for Apple, Microsoft, and Alphabet."
    ),
)

# ── ChromaDB singleton ────────────────────────────────────────────────────────

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        ef = SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(
            CHROMA_COLLECTION, embedding_function=ef
        )
    return _collection


# ── MCP Tools ─────────────────────────────────────────────────────────────────


@mcp.tool()
def search_sec_filings(
    query: str,
    company: Optional[str] = None,
    filing_type: Optional[str] = None,
    num_results: int = 5,
) -> str:
    """
    Search SEC filings (10-K and 10-Q) for Apple (AAPL), Microsoft (MSFT),
    and Alphabet (GOOGL). Returns the most relevant text chunks with metadata.

    Args:
        query: Search query — be specific with financial terms.
        company: Optional ticker filter (AAPL, MSFT, GOOGL).
        filing_type: Optional filing type filter (10-K or 10-Q).
        num_results: Number of chunks to retrieve (1-10, default 5).
    """
    col = _get_collection()
    num_results = min(max(num_results, 1), 10)

    clauses: list[dict] = []
    if company:
        clauses.append({"company": {"$eq": company}})
    if filing_type:
        clauses.append({"filing_type": {"$eq": filing_type}})

    where = None
    if len(clauses) == 1:
        where = clauses[0]
    elif len(clauses) > 1:
        where = {"$and": clauses}

    expanded = (
        f"{query} net sales revenue earnings income table figures "
        "millions billions fiscal year"
    )

    kwargs: dict = dict(
        query_texts=[expanded],
        n_results=num_results,
        include=["documents", "metadatas"],
    )
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    output_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        header = (
            f"[Chunk {i}] {meta.get('company_name', meta.get('company'))} | "
            f"{meta.get('filing_type')} | {meta.get('period')} | "
            f"{meta.get('section', 'General')}"
        )
        text = doc[:2000] if len(doc) > 2000 else doc
        output_parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(output_parts) if output_parts else "No results found."


@mcp.tool()
def get_knowledge_base_stats() -> str:
    """
    Get statistics about the SEC filings knowledge base — number of indexed
    chunks per company and total.
    """
    col = _get_collection()
    stats = {}
    total = 0
    for ticker, name in COMPANIES.items():
        res = col.get(where={"company": ticker}, include=[])
        count = len(res["ids"])
        stats[ticker] = {"name": name, "chunks": count}
        total += count

    lines = ["SEC Filings Knowledge Base Statistics", "=" * 45]
    for ticker, info in stats.items():
        lines.append(f"  {ticker:6s} ({info['name']}): {info['chunks']:,} chunks")
    lines.append(f"  {'TOTAL':6s}: {total:,} chunks")
    lines.append(f"\nCompanies: {', '.join(COMPANIES.keys())}")
    lines.append(f"Filing Types: {', '.join(FILING_TYPES)}")
    return "\n".join(lines)


@mcp.tool()
def calculate_financial_metric(expression: str, label: str = "") -> str:
    """
    Evaluate a mathematical expression for financial calculations.
    Use for growth rates, margins, ratios, etc.

    Args:
        expression: A math expression, e.g. '(391035 - 383285) / 383285 * 100'
        label: Human-readable label for the result.
    """
    allowed = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    }
    try:
        sanitized = re.sub(r"[^0-9+\-*/().,%e ]", "", expression)
        sanitized = sanitized.replace(",", "")
        result = eval(sanitized, {"__builtins__": {}}, allowed)  # noqa: S307
        if isinstance(result, float):
            formatted = f"{result:,.4f}".rstrip("0").rstrip(".")
        else:
            formatted = f"{result:,}"
        label_str = f" ({label})" if label else ""
        return f"Result{label_str}: {formatted}"
    except Exception as exc:
        return f"Calculation error: {exc}"


@mcp.tool()
def list_available_companies() -> str:
    """List all companies available in the knowledge base with their tickers."""
    lines = ["Available Companies:", "-" * 30]
    for ticker, name in COMPANIES.items():
        lines.append(f"  {ticker}: {name}")
    lines.append(f"\nFiling types available: {', '.join(FILING_TYPES)}")
    return "\n".join(lines)


# ── MCP Resources ─────────────────────────────────────────────────────────────


@mcp.resource("sec://companies")
def companies_resource() -> str:
    """JSON list of all companies in the knowledge base."""
    return json.dumps(
        [{"ticker": t, "name": n} for t, n in COMPANIES.items()],
        indent=2,
    )


@mcp.resource("sec://filing-types")
def filing_types_resource() -> str:
    """JSON list of available filing types."""
    return json.dumps(FILING_TYPES)


# ── Prompts ───────────────────────────────────────────────────────────────────


@mcp.prompt()
def financial_analysis_prompt(company: str, topic: str) -> str:
    """Generate a prompt for financial analysis of a specific company and topic."""
    return (
        f"You are a financial research analyst. Using the SEC filing search tools, "
        f"analyze {COMPANIES.get(company, company)}'s {topic}. "
        f"Search for relevant data in their 10-K and 10-Q filings, "
        f"cite specific numbers, and provide a thorough analysis."
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--sse" in sys.argv:
        mcp.run(transport="sse")
    else:
        mcp.run()
