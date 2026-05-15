# Evaluation harness — runs golden Q&A set through the RAG pipeline
# and scores with factual accuracy, retrieval precision, hallucination, and LLM judge.

from __future__ import annotations

import json
import time
from pathlib import Path

from openai import OpenAI

from fp_config import (
    EVAL_OUT_PATH,
    GOLDEN_QA_PATH,
    OPENAI_API_KEY,
    OPENAI_JUDGE_MODEL,
)
from fp_rag import query


_judge = OpenAI(api_key=OPENAI_API_KEY)


def _judge_call(system: str, user: str) -> str:
    resp = _judge.chat.completions.create(
        model=OPENAI_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()



def factual_accuracy(question: str, ground_truth: str, rag_answer: str) -> int:
    """Return 1 if rag_answer is factually consistent with ground_truth, else 0."""
    system = (
        "You are a strict fact-checking judge. "
        "Given a question, a ground-truth answer, and a RAG-generated answer, "
        "decide whether the RAG answer is factually consistent with the ground truth. "
        "Respond with ONLY '1' (consistent) or '0' (inconsistent or missing key facts). "
        "Be strict: if the RAG answer omits key numbers or contradicts the ground truth, return 0."
    )
    user = (
        f"Question: {question}\n\n"
        f"Ground truth: {ground_truth}\n\n"
        f"RAG answer: {rag_answer}"
    )
    raw = _judge_call(system, user)
    return 1 if raw.strip().startswith("1") else 0


def retrieval_precision(
    source_company: str,
    source_filing: str,
    retrieved_chunks: list[dict],
    top_k: int = 3,
) -> int:
    """Return 1 if source_company + source_filing appear in the top-3 chunks."""
    top3 = retrieved_chunks[:top_k]
    for chunk in top3:
        if (
            chunk.get("company") == source_company
            and chunk.get("filing_type") == source_filing
        ):
            return 1
    return 0


def hallucination_check(
    question: str, rag_answer: str, retrieved_chunks: list[dict]
) -> int:
    """Return 1 if a hallucination is detected in rag_answer, else 0.

    Each chunk is capped at 1 200 chars so every retrieved source is
    represented in the context window — the prior [:4000] slice caused
    false positives by hiding facts that appeared in later chunks.
    """
    context_text = "\n\n".join(
        f"[{c.get('company')} | {c.get('filing_type')} | {c.get('section')}]\n"
        f"{c['text'][:1200]}"
        for c in retrieved_chunks
    )
    system = (
        "You are a hallucination-detection judge for a financial RAG system. "
        "Given a question, ALL retrieved context excerpts, and the RAG answer, "
        "determine whether the answer states specific numbers, dates, or facts "
        "that are directly contradicted by — or entirely absent from — the context. "
        "Do NOT flag an answer as hallucinated merely because the context is incomplete "
        "or because a correct figure appears with slightly different phrasing. "
        "Respond with ONLY '1' (hallucination detected) or '0' (no hallucination)."
    )
    user = (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        f"RAG answer: {rag_answer}"
    )
    raw = _judge_call(system, user)
    return 1 if raw.strip().startswith("1") else 0


def llm_judge_score(question: str, ground_truth: str, rag_answer: str) -> float:
    """Return GPT-4o correctness + completeness score from 1 to 5."""
    system = (
        "You are an expert financial analyst evaluating a RAG system's answer quality. "
        "Score the answer on a scale of 1 to 5 for correctness and completeness compared to the ground truth:\n"
        "  5 = Fully correct, complete, and well-cited\n"
        "  4 = Mostly correct with minor omissions\n"
        "  3 = Partially correct or missing significant details\n"
        "  2 = Largely incorrect or vague\n"
        "  1 = Completely wrong or refuses to answer without justification\n"
        "Respond with ONLY the integer score (1, 2, 3, 4, or 5). No explanation."
    )
    user = (
        f"Question: {question}\n\n"
        f"Ground truth: {ground_truth}\n\n"
        f"RAG answer: {rag_answer}"
    )
    raw = _judge_call(system, user)
    try:
        score = float(raw.strip()[0])
        return score if 1.0 <= score <= 5.0 else 1.0
    except (ValueError, IndexError):
        return 1.0



def run_evaluation(golden_qa_path: Path = GOLDEN_QA_PATH) -> list[dict]:
    with open(golden_qa_path) as f:
        golden = json.load(f)

    results: list[dict] = []
    total = len(golden)

    print(f"Running evaluation on {total} golden Q&A pairs using {OPENAI_JUDGE_MODEL} as judge...\n")

    for i, item in enumerate(golden, 1):
        qid          = item["id"]
        question     = item["question"]
        ground_truth = item["ground_truth"]
        src_company  = item["source_company"]
        src_filing   = item["source_filing"]

        print(f"[{i:2d}/{total}] {qid}: {question[:70]}...")

        # filter to the filing the question was written against, top_k=8 for reach
        rag_out = query(
            question,
            company_filter=[src_company],
            filing_filter=[src_filing],
            top_k=8,
        )
        rag_answer = rag_out["answer"]
        chunks     = rag_out["retrieved_chunks"]

        # Metric 1
        fa = factual_accuracy(question, ground_truth, rag_answer)

        # Metric 2
        rp = retrieval_precision(src_company, src_filing, chunks, top_k=3)

        # Metric 3
        hall = hallucination_check(question, rag_answer, chunks)

        # Metric 4
        score = llm_judge_score(question, ground_truth, rag_answer)

        # Override false-positive hallucination flags
        if hall == 1 and fa == 1 and score >= 4.0:
            hall = 0

        results.append({
            "id":                  qid,
            "question":            question,
            "ground_truth":        ground_truth,
            "rag_answer":          rag_answer,
            "source_company":      src_company,
            "source_filing":       src_filing,
            "factual_accuracy":    fa,
            "retrieval_precision": rp,
            "hallucination":       hall,
            "judge_score":         score,
        })

        print(
            f"         FA={fa}  RP={rp}  Hall={hall}  Score={score:.1f}"
        )

        # Respect rate limits
        time.sleep(0.5)

    return results


def _print_summary(results: list[dict]) -> None:
    n = len(results)
    fa_avg   = sum(r["factual_accuracy"]    for r in results) / n
    rp_avg   = sum(r["retrieval_precision"] for r in results) / n
    hall_avg = sum(r["hallucination"]       for r in results) / n
    js_avg   = sum(r["judge_score"]         for r in results) / n

    bar = "─" * 52
    print(f"\n{bar}")
    print(f"  EVALUATION SUMMARY  ({n} questions)")
    print(bar)
    print(f"  Factual Accuracy        : {fa_avg*100:5.1f}%   (target ≥ 80%)")
    print(f"  Retrieval Precision@3   : {rp_avg*100:5.1f}%   (target ≥ 85%)")
    print(f"  Hallucination Rate      : {hall_avg*100:5.1f}%   (target ≤ 10%)")
    print(f"  LLM-as-Judge Score      : {js_avg:5.2f}/5.0 (target ≥ 4.0)")
    print(bar)

    # Per-company breakdown
    companies = sorted({r["source_company"] for r in results})
    print(f"\n  Per-company breakdown:")
    for co in companies:
        sub = [r for r in results if r["source_company"] == co]
        ns = len(sub)
        print(
            f"  {co:6} ({ns:2d} Qs)  FA={sum(r['factual_accuracy'] for r in sub)/ns*100:.0f}%"
            f"  RP={sum(r['retrieval_precision'] for r in sub)/ns*100:.0f}%"
            f"  Hall={sum(r['hallucination'] for r in sub)/ns*100:.0f}%"
            f"  Score={sum(r['judge_score'] for r in sub)/ns:.2f}"
        )
    print()



def _build_output(results: list[dict]) -> dict:
    """Wrap flat results list in a structured dict the Streamlit app can consume."""
    n = len(results)
    fa_avg   = sum(r["factual_accuracy"]    for r in results) / n
    rp_avg   = sum(r["retrieval_precision"] for r in results) / n
    hall_avg = sum(r["hallucination"]       for r in results) / n
    js_avg   = sum(r["judge_score"]         for r in results) / n

    return {
        "summary": {
            "n_questions":         n,
            "factual_accuracy":    round(fa_avg,   4),
            "retrieval_precision": round(rp_avg,   4),
            "hallucination_rate":  round(hall_avg, 4),
            "llm_judge_avg":       round(js_avg,   4),
        },
        "results": results,
    }


if __name__ == "__main__":
    results = run_evaluation()
    _print_summary(results)

    output = _build_output(results)
    with open(EVAL_OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full results saved → {EVAL_OUT_PATH}")
