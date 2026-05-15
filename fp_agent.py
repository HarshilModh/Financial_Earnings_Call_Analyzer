# Agentic analyst — uses OpenAI function calling to search filings,
# calculate metrics, compare companies, and plot charts.

from __future__ import annotations

import json
import math
import re
import textwrap
from typing import Generator, Optional

import chromadb
import plotly.graph_objects as go
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

from fp_config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    COMPANIES,
    EMBEDDING_MODEL,
    FILING_TYPES,
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



TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": (
                "Search SEC filings (10-K and 10-Q) in the knowledge base. "
                "Returns the most relevant text chunks with metadata. "
                "Use this to find specific financial data, risk factors, revenue breakdowns, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query — be specific and include financial terms.",
                    },
                    "company": {
                        "type": "string",
                        "enum": list(COMPANIES.keys()),
                        "description": "Optional. Filter to a specific company ticker.",
                    },
                    "filing_type": {
                        "type": "string",
                        "enum": FILING_TYPES,
                        "description": "Optional. Filter to 10-K (annual) or 10-Q (quarterly).",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default 5, max 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression. Use for computing growth rates, "
                "margins, percentages, differences, ratios, etc. "
                "Examples: '(391035 - 383285) / 383285 * 100' for YoY growth, "
                "'170782 / 391035 * 100' for margin calculation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python math expression to evaluate (e.g. '(100-80)/80*100').",
                    },
                    "label": {
                        "type": "string",
                        "description": "A human-readable label for what this calculation represents.",
                    },
                },
                "required": ["expression", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_companies",
            "description": (
                "Generate a structured comparison table across companies for a specific metric. "
                "Provide the data you've already gathered from search_filings calls. "
                "Returns a formatted markdown table."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "The metric being compared (e.g. 'Total Revenue', 'Gross Margin %').",
                    },
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "company": {"type": "string"},
                                "value": {"type": "string"},
                                "period": {"type": "string"},
                                "notes": {"type": "string"},
                            },
                            "required": ["company", "value"],
                        },
                        "description": "Array of company data points for the comparison.",
                    },
                },
                "required": ["metric_name", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_chart",
            "description": (
                "Create a chart to visualize financial data. Use after gathering data "
                "via search_filings. Returns a Plotly chart specification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "pie"],
                        "description": "Type of chart to create.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title.",
                    },
                    "x_label": {
                        "type": "string",
                        "description": "Label for x-axis (bar/line charts).",
                    },
                    "y_label": {
                        "type": "string",
                        "description": "Label for y-axis (bar/line charts).",
                    },
                    "series": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Series/legend name."},
                                "labels": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Category labels (x-axis values or pie labels).",
                                },
                                "values": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "Numeric values corresponding to labels.",
                                },
                            },
                            "required": ["name", "labels", "values"],
                        },
                        "description": "One or more data series to plot.",
                    },
                },
                "required": ["chart_type", "title", "series"],
            },
        },
    },
]



def _tool_search_filings(
    query: str,
    company: str | None = None,
    filing_type: str | None = None,
    num_results: int = 5,
) -> str:
    """Execute a ChromaDB search and return formatted results."""
    col = _get_collection()
    num_results = min(max(num_results, 1), 10)

    # Build where filter
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

    # Expand query
    expanded = (
        f"{query} "
        "net sales revenue earnings income table figures millions billions fiscal year"
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
        # Truncate long chunks
        text = doc[:2000] if len(doc) > 2000 else doc
        output_parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(output_parts) if output_parts else "No results found."


def _tool_calculate(expression: str, label: str = "") -> str:
    """Safely evaluate a math expression."""
    # Only allow safe math ops
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    }
    try:
        # Sanitize input
        sanitized = re.sub(r"[^0-9+\-*/().,%e ]", "", expression)
        sanitized = sanitized.replace(",", "")  # remove thousand separators
        result = eval(sanitized, {"__builtins__": {}}, allowed_names)  # noqa: S307
        if isinstance(result, float):
            formatted = f"{result:,.4f}".rstrip("0").rstrip(".")
        else:
            formatted = f"{result:,}"
        label_str = f" ({label})" if label else ""
        return f"Result{label_str}: {formatted}"
    except Exception as exc:
        return f"Calculation error: {exc}. Expression was: {expression}"


def _tool_compare_companies(metric_name: str, data: list[dict]) -> str:
    """Build a markdown comparison table."""
    if not data:
        return "No data provided for comparison."

    rows = []
    for item in data:
        company = item.get("company", "—")
        value = item.get("value", "—")
        period = item.get("period", "—")
        notes = item.get("notes", "")
        rows.append(f"| {company} | {value} | {period} | {notes} |")

    table = (
        f"### {metric_name} — Company Comparison\n\n"
        f"| Company | {metric_name} | Period | Notes |\n"
        f"|---------|{'—' * len(metric_name)}|--------|-------|\n"
        + "\n".join(rows)
    )
    return table


def _tool_plot_chart(
    chart_type: str,
    title: str,
    series: list[dict],
    x_label: str = "",
    y_label: str = "",
) -> dict:
    """Create a Plotly figure dict (serializable). Rendered by the Streamlit UI."""
    fig = go.Figure()

    # Color palette
    colors = [
        "#2563eb", "#10b981", "#f59e0b", "#ef4444",
        "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
    ]

    for i, s in enumerate(series):
        color = colors[i % len(colors)]
        name = s.get("name", f"Series {i+1}")
        labels = s.get("labels", [])
        values = s.get("values", [])

        if chart_type == "bar":
            fig.add_trace(go.Bar(
                x=labels, y=values, name=name,
                marker_color=color, text=values, textposition="auto",
            ))
        elif chart_type == "line":
            fig.add_trace(go.Scatter(
                x=labels, y=values, name=name, mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ))
        elif chart_type == "pie":
            fig.add_trace(go.Pie(
                labels=labels, values=values, name=name,
                marker=dict(colors=colors[:len(labels)]),
                textinfo="label+percent",
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )

    # Return serialized figure
    return json.loads(fig.to_json())


# Dispatch
_TOOL_DISPATCH = {
    "search_filings": _tool_search_filings,
    "calculate": _tool_calculate,
    "compare_companies": _tool_compare_companies,
    "plot_chart": _tool_plot_chart,
}


_AGENT_SYSTEM = textwrap.dedent("""\
    You are an expert financial research agent specializing in SEC filings analysis.
    You have access to a knowledge base of 10-K (annual) and 10-Q (quarterly) filings
    for Apple (AAPL), Microsoft (MSFT), and Alphabet (GOOGL).

    Your workflow:
    1. ANALYZE the user's question to determine what data you need.
    2. USE TOOLS to gather information — search filings, perform calculations, build comparisons.
    3. For comparison questions, search EACH company separately for best results.
    4. Use the calculate tool when you need exact percentages, growth rates, or ratios.
    5. Use plot_chart to visualize data when it would help the user understand trends.
    6. Once you have sufficient information, provide a comprehensive, well-cited answer.

    Citation format: [Company | Filing Type | Period | Section]
    Example: [AAPL | 10-K | FY Sep 2024 | Item 7 - MD&A]

    Rules:
    - ONLY use information from the retrieved filing chunks. Never fabricate data.
    - Quote exact numbers from the filings — do not round or estimate.
    - If data is not available in the knowledge base, say so clearly.
    - For multi-company questions, organize your response by company.
    - Be thorough but concise. Financial analysts value precision.
""")


AgentEvent = dict



MAX_ITERATIONS = 8  # Safety limit to prevent infinite loops


def agent_run(
    question: str,
    company_filter: list[str] | None = None,
    filing_filter: list[str] | None = None,
    top_k: int = TOP_K,
    conversation_history: list[dict] | None = None,
) -> Generator[AgentEvent, None, None]:
    """
    Run the agentic analysis loop for a user question.

    Yields AgentEvent dicts as the agent thinks, calls tools, and produces output.
    The caller (Streamlit UI) renders each event in real time.
    """
    client = _get_openai()

    messages: list[dict] = [{"role": "system", "content": _AGENT_SYSTEM}]

    if conversation_history:

        for msg in conversation_history[-6:]:
            messages.append(msg)

    filter_note = ""
    if company_filter and set(company_filter) != set(COMPANIES.keys()):
        filter_note += f"Active company filter: {', '.join(company_filter)}. "
    if filing_filter and set(filing_filter) != set(FILING_TYPES):
        filter_note += f"Active filing filter: {', '.join(filing_filter)}. "

    user_content = question
    if filter_note:
        user_content += f"\n\n[System note: {filter_note}The user has set these filters in the sidebar.]"

    messages.append({"role": "user", "content": user_content})

    yield {"type": "thinking", "content": "Analyzing your question and planning tool calls..."}

    for iteration in range(MAX_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.0,
            )
        except Exception as exc:
            yield {"type": "error", "content": f"OpenAI API error: {exc}"}
            return

        choice = response.choices[0]
        assistant_msg = choice.message

        messages.append(assistant_msg.to_dict())

        if assistant_msg.tool_calls:
            for tool_call in assistant_msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                yield {
                    "type": "tool_call",
                    "name": fn_name,
                    "args": fn_args,
                    "id": tool_call.id,
                }

                # Run tool
                tool_fn = _TOOL_DISPATCH.get(fn_name)
                if tool_fn is None:
                    result = f"Unknown tool: {fn_name}"
                else:
                    try:
                        result = tool_fn(**fn_args)
                    except Exception as exc:
                        result = f"Tool error ({fn_name}): {exc}"

                # Charts get special handling
                if fn_name == "plot_chart" and isinstance(result, dict):
                    yield {"type": "chart", "figure": result}
                    tool_result_str = f"Chart '{fn_args.get('title', 'chart')}' created and displayed to the user."
                else:
                    tool_result_str = str(result)
                    # Truncate very large results for display
                    display_result = tool_result_str[:1500] + "…" if len(tool_result_str) > 1500 else tool_result_str
                    yield {
                        "type": "tool_result",
                        "name": fn_name,
                        "result": display_result,
                    }

                # Feed result back
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result_str[:6000],  # Stay within context limits
                })

        elif assistant_msg.content:
            yield {"type": "answer", "content": assistant_msg.content}
            return

        elif choice.finish_reason == "stop":
            yield {"type": "answer", "content": "Analysis complete. No additional information needed."}
            return

    # Max iterations safety
    yield {
        "type": "answer",
        "content": (
            "I've reached the maximum number of analysis steps. "
            "Here's what I found so far — please refine your question for more detail."
        ),
    }



if __name__ == "__main__":
    print("Running agent smoke test...\n")
    for event in agent_run("What was Apple's total revenue in FY2024?"):
        if event["type"] == "thinking":
            print(f"🧠 {event['content']}")
        elif event["type"] == "tool_call":
            print(f"🔧 Calling {event['name']}({json.dumps(event['args'], indent=2)})")
        elif event["type"] == "tool_result":
            print(f"📋 Result: {event['result'][:200]}...")
        elif event["type"] == "chart":
            print(f"📊 Chart generated")
        elif event["type"] == "answer":
            print(f"\n{'─'*60}\n📝 ANSWER:\n{event['content']}")
        elif event["type"] == "error":
            print(f"❌ {event['content']}")
