# Python 3.11+
# Stevens Username: harshilmodh
#
# Phase 5 + 6 — Streamlit UI for the Financial Earnings Call Analyzer.
#
# Layout:
#   Sidebar  — company / filing-type filters, top-k slider, "Run Evaluation" button, KB stats
#   Tab 1    — Q&A: question input, answer panel, cited sources
#   Tab 2    — 🤖 Agent: agentic AI analyst with tool use and chat interface
#   Tab 3    — History: last 5 Q&A pairs
#   Tab 4    — Evaluation: metrics table from eval_results.json
#
# Run:
#   streamlit run fp_app.py

from __future__ import annotations

import json
import time
import subprocess
import sys
from pathlib import Path

import chromadb
import plotly.io as pio
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from fp_config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    COMPANIES,
    EMBEDDING_MODEL,
    EVAL_OUT_PATH,
    FILING_TYPES,
    TOP_K,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Earnings Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS / Design System ──────────────────────────────────────────────────

with open("styles.css") as f:
    st.markdown(f"<style>\n{f.read()}\n</style>", unsafe_allow_html=True)

# ── ChromaDB connection (cached) ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base…")
def _get_collection():
    if not CHROMA_DIR.exists():
        return None
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    try:
        return client.get_collection(name=CHROMA_COLLECTION, embedding_function=ef)
    except Exception:
        return None


def _kb_stats(collection) -> dict[str, int]:
    stats: dict[str, int] = {}
    for ticker in COMPANIES:
        res = collection.get(where={"company": ticker}, include=[])
        stats[ticker] = len(res["ids"])
    return stats


# ── RAG backend ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_rag():
    try:
        from fp_rag import query as rag_query
        return rag_query
    except ImportError:
        return None


_RAG_QUERY = _load_rag()

# ── Agent backend ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_agent():
    try:
        from fp_agent import agent_run
        return agent_run
    except ImportError:
        return None


_AGENT_RUN = _load_agent()

# ── Research backend ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_research():
    try:
        from fp_research import run_deep_research
        return run_deep_research
    except ImportError:
        return None


_DEEP_RESEARCH = _load_research()


def run_query(question: str, company_filter, filing_filter, top_k: int) -> dict:
    if _RAG_QUERY is not None:
        return _RAG_QUERY(
            question=question,
            company_filter=company_filter or None,
            filing_filter=filing_filter or None,
            top_k=top_k,
        )
    return {
        "answer": "RAG engine not available. Please ensure fp_rag.py is present.",
        "sources": [],
        "retrieved_chunks": [],
    }


# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Agent chat state
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages: list[dict] = []

if "agent_conversation" not in st.session_state:
    st.session_state.agent_conversation: list[dict] = []

if "research_report" not in st.session_state:
    st.session_state.research_report = None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>📊 Financial Analyzer</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.9rem;'>FE 524 — harshilmodh</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    theme_mode = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True, label_visibility="collapsed")
    if theme_mode == "Dark":
        st.markdown("""<style>
        :root, [data-testid="stAppViewContainer"] {
            --background-color: #09090b !important;
            --secondary-background-color: #18181b !important;
            --text-color: #f8fafc !important;
            --primary-color: #6366f1 !important;
            --header-qa-bg1: #064e3b !important;
            --header-qa-bg2: #022c22 !important;
            --header-qa-border: #047857 !important;
            --header-qa-title: #34d399 !important;
            --header-qa-text: #a7f3d0 !important;
            
            --header-agent-bg1: #1e3a8a !important;
            --header-agent-bg2: #172554 !important;
            --header-agent-border: #1d4ed8 !important;
            --header-agent-title: #60a5fa !important;
            --header-agent-text: #bfdbfe !important;
            
            --header-research-bg1: #78350f !important;
            --header-research-bg2: #451a03 !important;
            --header-research-border: #b45309 !important;
            --header-research-title: #fbbf24 !important;
            --header-research-text: #fde68a !important;
        }
        </style>""", unsafe_allow_html=True)
    else:
        st.markdown("""<style>
        :root, [data-testid="stAppViewContainer"] {
            --background-color: #f8fafc !important;
            --secondary-background-color: #ffffff !important;
            --text-color: #0f172a !important;
            --primary-color: #3b82f6 !important;
            --header-qa-bg1: #f0fdf4 !important;
            --header-qa-bg2: #dcfce7 !important;
            --header-qa-border: #bbf7d0 !important;
            --header-qa-title: #166534 !important;
            --header-qa-text: #15803d !important;
            
            --header-agent-bg1: #eff6ff !important;
            --header-agent-bg2: #dbeafe !important;
            --header-agent-border: #bfdbfe !important;
            --header-agent-title: #1e3a8a !important;
            --header-agent-text: #3b82f6 !important;
            
            --header-research-bg1: #fef3c7 !important;
            --header-research-bg2: #fef08a !important;
            --header-research-border: #fbbf24 !important;
            --header-research-title: #92400e !important;
            --header-research-text: #a16207 !important;
        }
        </style>""", unsafe_allow_html=True)
        
    st.markdown("---")

    st.subheader("Company Filter")
    all_companies = list(COMPANIES.keys())
    selected_companies = st.multiselect(
        "Companies",
        options=all_companies,
        default=all_companies,
        label_visibility="collapsed",
    )
    company_filter = None if set(selected_companies) == set(all_companies) else selected_companies

    st.subheader("Filing Type Filter")
    selected_filings = st.multiselect(
        "Filing Types",
        options=FILING_TYPES,
        default=FILING_TYPES,
        label_visibility="collapsed",
    )
    filing_filter = None if set(selected_filings) == set(FILING_TYPES) else selected_filings

    st.subheader("Retrieval")
    top_k = st.slider("Top-K chunks", min_value=3, max_value=10, value=TOP_K)

    st.markdown("---")

    # Run evaluation button
    st.subheader("Evaluation")
    if st.button("Run Evaluation", use_container_width=True, type="secondary"):
        with st.spinner("Running evaluation (~2–3 min)…"):
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "fp_evaluate.py")],
                capture_output=True, text=True
            )
        if proc.returncode == 0:
            st.success("Evaluation complete! Check the Evaluation tab.")
        else:
            st.error(f"Evaluation failed:\n{proc.stderr[-500:]}")

    st.markdown("---")

    # KB stats
    collection = _get_collection()
    if collection:
        st.subheader("Knowledge Base")
        stats = _kb_stats(collection)
        for ticker, count in stats.items():
            st.metric(label=f"{ticker} — {COMPANIES[ticker]}", value=f"{count:,} chunks")
    else:
        st.warning("ChromaDB not found. Run build_kb first.", icon="⚠️")

    st.markdown("---")
    rag_status = "🟢 RAG engine loaded" if _RAG_QUERY else "🔴 RAG engine not found"
    agent_status = "🟢 Agent loaded" if _AGENT_RUN else "🔴 Agent not found"
    research_status = "🟢 Research pipeline loaded" if _DEEP_RESEARCH else "🔴 Research not found"
    st.caption(rag_status)
    st.caption(agent_status)
    st.caption(research_status)

# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_qa, tab_agent, tab_research, tab_history, tab_eval = st.tabs(["💬 Q&A", "🤖 Agent", "📑 Deep Research", "🕑 History", "📈 Evaluation"])

# ── Tab 1: Q&A ────────────────────────────────────────────────────────────────

with tab_qa:
    st.markdown(
        """
        <div class="header-card qa-header">
            <h3>📊 Financial Earnings Call Analyzer</h3>
            <p>
                Ask questions about <strong>Apple (AAPL)</strong>, <strong>Microsoft (MSFT)</strong>, or <strong>Alphabet (GOOGL)</strong> 
                SEC filings (10-K annual reports & 10-Q quarterly reports). 
                The system retrieves the most relevant passages and synthesizes a cited answer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("💡 Example questions", expanded=False):
        examples = [
            "What were Apple's iPhone net sales in fiscal year 2024?",
            "How did Apple's gross margin change between Products and Services in FY2024?",
            "What risk factors does Microsoft identify in its FY2024 10-K?",
            "What drove Google Search & other revenue growth in 2024?",
            "Compare the operating income growth of Apple and Alphabet in their most recent fiscal year.",
            "What are the components of Google Cloud revenues?",
            "What did Microsoft say about Activision Blizzard's impact on its financials?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.question_input = ex
                st.rerun()

    question = st.text_area(
        "Your question",
        placeholder="e.g. What drove Apple's Services revenue growth in FY2024?",
        height=90,
        label_visibility="collapsed",
        key="question_input",
    )

    ask_col, clear_col = st.columns([1, 6])
    with ask_col:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Clear", use_container_width=False):
            st.session_state.last_result = None
            st.session_state.question_input = ""
            st.rerun()

    if ask_clicked:
        q = question.strip()
        if q:
            with st.spinner("Retrieving and synthesizing answer…"):
                result = run_query(q, company_filter, filing_filter, top_k)
            entry = {"question": q, **result}
            st.session_state.last_result = entry
            st.session_state.history.insert(0, entry)
            # Keep only last 5
            st.session_state.history = st.session_state.history[:5]
        else:
            st.warning("Please enter a question.", icon="⚠️")

    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("#### Answer")
        st.markdown(
            f'<div class="answer-card">{res["answer"]}</div>',
            unsafe_allow_html=True,
        )

        sources = res.get("sources", [])
        if sources:
            with st.expander(f"📄 Sources ({len(sources)} retrieved chunks)", expanded=True):
                for i, src in enumerate(sources, 1):
                    badge_html = (
                        f'<span class="source-badge">{src.get("company","")}</span>'
                        f'<span class="source-badge">{src.get("filing_type","")}</span>'
                        f'<span class="source-badge">{src.get("period","")}</span>'
                        f'<span class="source-badge">{src.get("section","")}</span>'
                    )
                    st.markdown(f"**Source {i}** &nbsp; {badge_html}", unsafe_allow_html=True)
                    snippet = src.get("text_snippet", "")
                    st.markdown(
                        f'<div class="source-quote">"{snippet}"</div>',
                        unsafe_allow_html=True,
                    )
                    if i < len(sources):
                        st.markdown("<hr>", unsafe_allow_html=True)

# ── Tab 2: Agent ──────────────────────────────────────────────────────────────

_TOOL_ICONS = {
    "search_filings": "🔍",
    "calculate": "🧮",
    "compare_companies": "📊",
    "plot_chart": "📈",
}


def _render_agent_message(msg: dict):
    """Render a single agent message in the chat UI."""
    role = msg.get("role", "assistant")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])

    elif role == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            # Render events stored in the message
            for event in msg.get("events", []):
                etype = event["type"]

                if etype == "thinking":
                    st.markdown(
                        f'<div class="thinking-box">🧠 {event["content"]}</div>',
                        unsafe_allow_html=True,
                    )

                elif etype == "tool_call":
                    icon = _TOOL_ICONS.get(event["name"], "🔧")
                    args_str = json.dumps(event["args"], indent=2)
                    with st.expander(f"{icon} **{event['name']}**", expanded=False):
                        st.code(args_str, language="json")

                elif etype == "tool_result":
                    with st.expander(f"📋 Result from **{event['name']}**", expanded=False):
                        st.markdown(
                            f'<div class="tool-result-box"><pre>{event["result"]}</pre></div>',
                            unsafe_allow_html=True,
                        )

                elif etype == "chart":
                    fig = pio.from_json(json.dumps(event["figure"]))
                    st.plotly_chart(fig, use_container_width=True)

                elif etype == "answer":
                    st.markdown(event["content"])

                elif etype == "error":
                    st.error(event["content"])


with tab_agent:
    st.markdown(
        """
        <div class="header-card agent-header">
            <h3>🤖 Agentic Financial Analyst</h3>
            <p>
                An autonomous AI agent that <strong>decomposes your question</strong>, searches SEC filings, 
                performs calculations, builds comparison tables, and generates charts — all automatically. 
                Watch the agent's reasoning process in real time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render existing conversation
    for msg in st.session_state.agent_messages:
        _render_agent_message(msg)

    # Chat input
    agent_question = st.chat_input(
        "Ask the agent anything about AAPL, MSFT, or GOOGL filings…",
        key="agent_chat_input",
    )

    col_clear_agent, col_spacer = st.columns([1, 5])
    with col_clear_agent:
        if st.button("🗑️ Clear conversation", key="clear_agent"):
            st.session_state.agent_messages = []
            st.session_state.agent_conversation = []
            st.rerun()

    if agent_question and _AGENT_RUN:
        # Add user message
        user_msg = {"role": "user", "content": agent_question}
        st.session_state.agent_messages.append(user_msg)
        st.session_state.agent_conversation.append(
            {"role": "user", "content": agent_question}
        )

        # Render user message
        with st.chat_message("user"):
            st.markdown(agent_question)

        # Run agent and stream events
        events: list[dict] = []
        with st.chat_message("assistant", avatar="🤖"):
            for event in _AGENT_RUN(
                question=agent_question,
                company_filter=company_filter,
                filing_filter=filing_filter,
                top_k=top_k,
                conversation_history=st.session_state.agent_conversation,
            ):
                events.append(event)
                etype = event["type"]

                if etype == "thinking":
                    st.markdown(
                        f'<div class="thinking-box">🧠 {event["content"]}</div>',
                        unsafe_allow_html=True,
                    )

                elif etype == "tool_call":
                    icon = _TOOL_ICONS.get(event["name"], "🔧")
                    args_str = json.dumps(event["args"], indent=2)
                    with st.expander(f"{icon} **{event['name']}**", expanded=True):
                        st.code(args_str, language="json")

                elif etype == "tool_result":
                    with st.expander(f"📋 Result from **{event['name']}**", expanded=False):
                        st.markdown(
                            f'<div class="tool-result-box"><pre>{event["result"]}</pre></div>',
                            unsafe_allow_html=True,
                        )

                elif etype == "chart":
                    fig = pio.from_json(json.dumps(event["figure"]))
                    st.plotly_chart(fig, use_container_width=True)

                elif etype == "answer":
                    st.markdown(event["content"])

                elif etype == "error":
                    st.error(event["content"])

        # Save assistant message with all events
        assistant_msg = {"role": "assistant", "events": events}
        st.session_state.agent_messages.append(assistant_msg)

        # Save the final answer text for conversational context
        final_answer = ""
        for e in events:
            if e["type"] == "answer":
                final_answer = e["content"]
                break
        if final_answer:
            st.session_state.agent_conversation.append(
                {"role": "assistant", "content": final_answer}
            )

    elif agent_question and not _AGENT_RUN:
        st.error("Agent module not found. Ensure `fp_agent.py` is present.", icon="❌")

# ── Tab 3: Deep Research ──────────────────────────────────────────────────────

with tab_research:
    st.markdown(
        """
        <div class="header-card research-header">
            <h3>📑 Multi-Agent Deep Research</h3>
            <p>
                Enter a research topic and let <strong>three specialized agents</strong> collaborate:<br>
                🗺️ <strong>Planner</strong> → decomposes into research questions &nbsp;|&nbsp;
                🔍 <strong>Researcher</strong> → gathers data from filings &nbsp;|&nbsp;
                ✍️ <strong>Writer</strong> → produces a cited research report
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    research_topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Compare Apple and Microsoft's cloud/services revenue growth in FY2024",
        key="research_topic_input",
    )

    r_col1, r_col2 = st.columns([1, 5])
    with r_col1:
        run_research = st.button("🚀 Generate Report", type="primary", use_container_width=True)
    with r_col2:
        if st.button("Clear Report", key="clear_research"):
            st.session_state.research_report = None
            st.rerun()

    if run_research and research_topic.strip() and _DEEP_RESEARCH:
        st.session_state.research_report = None
        events_log = []

        progress_bar = st.progress(0, text="Starting research pipeline…")
        status_container = st.container()
        report_container = st.container()

        total_steps = 3  # plan, research, write
        current_step = 0

        for event in _DEEP_RESEARCH(
            topic=research_topic.strip(),
            company_filter=company_filter,
            filing_filter=filing_filter,
            top_k=top_k,
        ):
            events_log.append(event)
            etype = event["type"]

            if etype == "phase":
                current_step += 1
                progress_bar.progress(
                    min(current_step / total_steps, 0.95),
                    text=event["content"],
                )
                with status_container:
                    st.markdown(
                        f'<div class="thinking-box">{event["content"]}</div>',
                        unsafe_allow_html=True,
                    )

            elif etype == "question":
                with status_container:
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp; **Q{event['index']}/{event['total']}:** {event['text']}"
                    )

            elif etype == "finding":
                with status_container:
                    with st.expander(
                        f"✅ Finding {event['index']}/{event['total']}: {event['question'][:60]}…",
                        expanded=False,
                    ):
                        st.markdown(event["summary"])
                        if event.get("sources"):
                            badges = " ".join(
                                f'<span class="source-badge">{s["company"]} {s["filing_type"]} {s["period"]}</span>'
                                for s in event["sources"]
                            )
                            st.markdown(badges, unsafe_allow_html=True)

            elif etype == "report":
                progress_bar.progress(1.0, text="✅ Report complete!")
                st.session_state.research_report = event["content"]
                with report_container:
                    st.markdown("---")
                    st.markdown("### 📄 Research Report")
                    st.markdown(
                        f'<div class="answer-card">{event["content"]}</div>',
                        unsafe_allow_html=True,
                    )

            elif etype == "error":
                with status_container:
                    st.error(event["content"])

    elif run_research and not _DEEP_RESEARCH:
        st.error("Research module not found. Ensure `fp_research.py` is present.")

    # Show saved report on reruns
    if not run_research and st.session_state.research_report:
        st.markdown("---")
        st.markdown("### 📄 Research Report")
        st.markdown(
            f'<div class="answer-card">{st.session_state.research_report}</div>',
            unsafe_allow_html=True,
        )

# ── Tab 4: History ────────────────────────────────────────────────────────────

with tab_history:
    st.subheader("Recent Queries (last 5)")
    if not st.session_state.history:
        st.info("No queries yet. Ask something in the Q&A tab.", icon="ℹ️")
    else:
        for i, item in enumerate(st.session_state.history):
            label = f"Q{i+1}: {item['question'][:80]}{'…' if len(item['question']) > 80 else ''}"
            with st.expander(label, expanded=(i == 0)):
                st.markdown(
                    f'<div class="answer-card">{item["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                sources = item.get("sources", [])
                if sources:
                    st.caption("Sources: " + " · ".join(
                        f"{s.get('company','')} {s.get('filing_type','')} {s.get('period','')}"
                        for s in sources
                    ))
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ── Tab 4: Evaluation ─────────────────────────────────────────────────────────

with tab_eval:
    st.subheader("Evaluation Metrics")
    st.markdown(
        "Run the evaluation from the sidebar **Run Evaluation** button, "
        "or via `python fp_evaluate.py`. "
        "Results are read from `eval_results.json`."
    )

    if EVAL_OUT_PATH.exists():
        with open(EVAL_OUT_PATH) as f:
            eval_data = json.load(f)

        summary = eval_data.get("summary", {})

        # Summary metric cards
        col1, col2, col3, col4 = st.columns(4)
        metric_cfg = [
            ("factual_accuracy",    "Factual Accuracy",      0.80, "%",   False),
            ("retrieval_precision", "Retrieval Precision@3", 0.85, "%",   False),
            ("hallucination_rate",  "Hallucination Rate",    0.10, "%",   True),   # lower is better
            ("llm_judge_avg",       "LLM-as-Judge Score",    4.0,  "/5",  False),
        ]
        for col, (key, label, target, unit, lower_better) in zip(
            [col1, col2, col3, col4], metric_cfg
        ):
            val = summary.get(key)
            if val is not None:
                if unit == "%":
                    display = f"{val*100:.1f}%"
                    ok = (val <= target) if lower_better else (val >= target)
                else:
                    display = f"{val:.2f}{unit}"
                    ok = val >= target
                delta = "✓ Target met" if ok else "✗ Below target"
                col.metric(label=label, value=display, delta=delta,
                           delta_color="normal" if ok else "inverse")

        st.markdown("---")
        st.subheader("Per-Question Results")

        rows = eval_data.get("results", [])
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)[[
                "id", "question", "source_company", "source_filing",
                "factual_accuracy", "retrieval_precision", "hallucination", "judge_score"
            ]]
            df.columns = ["ID", "Question", "Company", "Filing",
                          "Factual ✓", "Retrieval ✓", "Hallucinated", "Score/5"]
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Per-company breakdown
        if rows:
            st.markdown("---")
            st.subheader("Per-Company Breakdown")
            companies = sorted({r["source_company"] for r in rows})
            summary_rows = []
            for co in companies:
                sub = [r for r in rows if r["source_company"] == co]
                ns = len(sub)
                summary_rows.append({
                    "Company": co,
                    "Questions": ns,
                    "Factual Accuracy": f"{sum(r['factual_accuracy'] for r in sub)/ns*100:.0f}%",
                    "Retrieval Precision": f"{sum(r['retrieval_precision'] for r in sub)/ns*100:.0f}%",
                    "Hallucination Rate": f"{sum(r['hallucination'] for r in sub)/ns*100:.0f}%",
                    "Avg Judge Score": f"{sum(r['judge_score'] for r in sub)/ns:.2f}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    else:
        st.info("Evaluation results not yet generated. Click **Run Evaluation** in the sidebar.", icon="ℹ️")
        import pandas as pd
        st.dataframe(pd.DataFrame([
            {"Metric": "Factual Accuracy",           "Target": "≥ 80%",   "Status": "Pending"},
            {"Metric": "Retrieval Precision (top-3)", "Target": "≥ 85%",   "Status": "Pending"},
            {"Metric": "Hallucination Rate",          "Target": "≤ 10%",   "Status": "Pending"},
            {"Metric": "LLM-as-Judge Score",          "Target": "≥ 4.0/5", "Status": "Pending"},
        ]), use_container_width=True, hide_index=True)
