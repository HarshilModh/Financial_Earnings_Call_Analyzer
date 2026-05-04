# Python 3.11+
# Stevens Username: harshilmodh
#
# Phase 8 — Multi-Agent Deep Research Pipeline.
#
# Three-agent pipeline for autonomous financial research:
#   1. Planner Agent  — decomposes the topic into research questions
#   2. Research Agent  — gathers data from the KB for each question
#   3. Writer Agent    — synthesizes a structured, cited research report
#
# Public API:
#   run_deep_research(topic, company_filter, filing_filter, top_k)
#   → yields ResearchEvent dicts for streaming UI updates

from __future__ import annotations

import json
import textwrap
from typing import Generator, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

from harshilmodh_fp_config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    COMPANIES,
    EMBEDDING_MODEL,
    FILING_TYPES,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
)

# ── Singletons ────────────────────────────────────────────────────────────────

_collection = None
_openai_client: Optional[OpenAI] = None


def _get_collection():
    global _collection
    if _collection is None:
        ef = SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(
            CHROMA_COLLECTION, embedding_function=ef
        )
    return _collection


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _llm(system: str, user: str, temperature: float = 0.0) -> str:
    client = _get_openai()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _search(query: str, company: str | None, filing: str | None, n: int) -> list[dict]:
    col = _get_collection()
    clauses = []
    if company:
        clauses.append({"company": {"$eq": company}})
    if filing:
        clauses.append({"filing_type": {"$eq": filing}})
    where = None
    if len(clauses) == 1:
        where = clauses[0]
    elif len(clauses) > 1:
        where = {"$and": clauses}

    expanded = f"{query} revenue earnings income table fiscal year figures"
    kwargs = dict(query_texts=[expanded], n_results=n, include=["documents", "metadatas"])
    if where:
        kwargs["where"] = where
    results = col.query(**kwargs)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({**meta, "text": doc})
    return chunks


# ── Event types ───────────────────────────────────────────────────────────────
# {"type": "phase",    "phase": "planning|researching|writing", "content": str}
# {"type": "question", "index": int, "total": int, "text": str}
# {"type": "finding",  "question": str, "summary": str, "sources": list}
# {"type": "report",   "content": str}
# {"type": "error",    "content": str}

ResearchEvent = dict


# ── Agent 1: Planner ──────────────────────────────────────────────────────────

_PLANNER_SYSTEM = textwrap.dedent("""\
    You are a financial research planner. Given a research topic about SEC filings,
    generate 4-6 specific, targeted research questions that would produce a
    comprehensive analysis.

    Rules:
    - Questions should be specific and answerable from 10-K/10-Q filings.
    - Cover different angles: revenue, profitability, risks, growth drivers, segments.
    - If the topic mentions specific companies, focus questions on those.
    - If it's a comparison topic, include cross-company questions.

    Respond with ONLY a JSON array of question strings. No markdown, no explanation.
    Example: ["What was...", "How did...", "What are..."]
""")


def _plan(topic: str, companies: list[str] | None) -> list[str]:
    company_note = ""
    if companies:
        names = [f"{t} ({COMPANIES[t]})" for t in companies if t in COMPANIES]
        company_note = f"\nFocus on: {', '.join(names)}"
    raw = _llm(_PLANNER_SYSTEM, f"Research topic: {topic}{company_note}")
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        questions = json.loads(raw)
        if isinstance(questions, list):
            return questions[:6]
    except json.JSONDecodeError:
        pass
    return [topic]  # fallback


# ── Agent 2: Researcher ──────────────────────────────────────────────────────

_RESEARCHER_SYSTEM = textwrap.dedent("""\
    You are a financial research analyst. Given a specific question and retrieved
    SEC filing excerpts, produce a concise factual summary (3-5 sentences).

    Rules:
    - Use ONLY information from the provided excerpts.
    - Include exact numbers, dollar figures, and percentages as they appear.
    - Cite sources as [Company | Filing | Period | Section].
    - If the excerpts don't contain relevant information, say so clearly.
    - Be precise and concise — this feeds into a larger report.
""")


def _research_question(
    question: str,
    company_filter: list[str] | None,
    filing_filter: list[str] | None,
    top_k: int,
) -> dict:
    """Research a single question. Returns {summary, sources, chunks}."""
    # For comparison questions, search each company separately
    all_chunks = []
    companies = company_filter or list(COMPANIES.keys())
    for comp in companies:
        filing = filing_filter[0] if filing_filter and len(filing_filter) == 1 else None
        chunks = _search(question, comp, filing, min(top_k, 5))
        all_chunks.extend(chunks)

    # Deduplicate by text
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        key = c["text"][:200]
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)
    unique_chunks = unique_chunks[:top_k]

    # Build context
    context_parts = []
    for i, c in enumerate(unique_chunks, 1):
        header = (
            f"[Excerpt {i} — {c.get('company_name', c.get('company'))} | "
            f"{c.get('filing_type')} | {c.get('period')} | "
            f"{c.get('section', 'General')}]"
        )
        context_parts.append(f"{header}\n{c['text'][:1500]}")

    context = "\n\n".join(context_parts)
    user_msg = f"Question: {question}\n\nRetrieved excerpts:\n{context}"
    summary = _llm(_RESEARCHER_SYSTEM, user_msg)

    sources = [
        {
            "company": c.get("company"),
            "filing_type": c.get("filing_type"),
            "period": c.get("period"),
            "section": c.get("section"),
        }
        for c in unique_chunks[:3]
    ]

    return {"summary": summary, "sources": sources, "chunks": unique_chunks}


# ── Agent 3: Writer ───────────────────────────────────────────────────────────

_WRITER_SYSTEM = textwrap.dedent("""\
    You are a senior financial analyst writing a research report. Given a topic
    and a set of research findings, produce a well-structured, professional report.

    Report structure:
    1. **Executive Summary** — 2-3 sentence overview of key findings
    2. **Key Findings** — Organized by theme, with inline citations [Company | Filing | Period]
    3. **Data Highlights** — Key numbers and metrics in a concise format
    4. **Risk Factors** — Any risks or concerns identified
    5. **Conclusion** — Brief forward-looking summary

    Rules:
    - Use markdown formatting for structure.
    - Every factual claim must have a citation.
    - Include exact numbers from the filings — do not round or estimate.
    - Be thorough but concise. Target 400-600 words.
    - Professional tone suitable for an institutional research note.
""")


def _write_report(topic: str, findings: list[dict]) -> str:
    findings_text = []
    for i, f in enumerate(findings, 1):
        sources_str = ", ".join(
            f"[{s['company']} | {s['filing_type']} | {s['period']}]"
            for s in f.get("sources", [])
        )
        findings_text.append(
            f"### Finding {i}: {f['question']}\n"
            f"{f['summary']}\n"
            f"Sources: {sources_str}"
        )

    user_msg = (
        f"Research Topic: {topic}\n\n"
        f"Research Findings:\n\n" + "\n\n".join(findings_text)
    )
    return _llm(_WRITER_SYSTEM, user_msg, temperature=0.1)


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def run_deep_research(
    topic: str,
    company_filter: list[str] | None = None,
    filing_filter: list[str] | None = None,
    top_k: int = TOP_K,
) -> Generator[ResearchEvent, None, None]:
    """
    Run the multi-agent deep research pipeline.
    Yields events for the UI to render in real time.
    """

    # ── Phase 1: Planning ─────────────────────────────────────────────────
    yield {
        "type": "phase",
        "phase": "planning",
        "content": "🗺️ Planning research — decomposing topic into targeted questions…",
    }

    try:
        questions = _plan(topic, company_filter)
    except Exception as exc:
        yield {"type": "error", "content": f"Planning failed: {exc}"}
        return

    for i, q in enumerate(questions):
        yield {"type": "question", "index": i + 1, "total": len(questions), "text": q}

    # ── Phase 2: Research ─────────────────────────────────────────────────
    yield {
        "type": "phase",
        "phase": "researching",
        "content": f"🔍 Researching {len(questions)} questions across SEC filings…",
    }

    findings: list[dict] = []
    for i, q in enumerate(questions):
        try:
            result = _research_question(q, company_filter, filing_filter, top_k)
            finding = {"question": q, **result}
            findings.append(finding)
            yield {
                "type": "finding",
                "question": q,
                "index": i + 1,
                "total": len(questions),
                "summary": result["summary"],
                "sources": result["sources"],
            }
        except Exception as exc:
            yield {"type": "error", "content": f"Research error on Q{i+1}: {exc}"}

    if not findings:
        yield {"type": "error", "content": "No findings gathered. Cannot generate report."}
        return

    # ── Phase 3: Writing ──────────────────────────────────────────────────
    yield {
        "type": "phase",
        "phase": "writing",
        "content": "✍️ Writing final research report…",
    }

    try:
        report = _write_report(topic, findings)
        yield {"type": "report", "content": report}
    except Exception as exc:
        yield {"type": "error", "content": f"Report generation failed: {exc}"}


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running deep research smoke test…\n")
    for event in run_deep_research("Apple's revenue and profitability in FY2024"):
        etype = event["type"]
        if etype == "phase":
            print(f"\n{'='*60}\n{event['content']}\n{'='*60}")
        elif etype == "question":
            print(f"  Q{event['index']}/{event['total']}: {event['text']}")
        elif etype == "finding":
            print(f"  ✓ Finding {event['index']}: {event['summary'][:100]}…")
        elif etype == "report":
            print(f"\n{'─'*60}\nFINAL REPORT:\n{'─'*60}\n{event['content']}")
        elif etype == "error":
            print(f"  ❌ {event['content']}")
