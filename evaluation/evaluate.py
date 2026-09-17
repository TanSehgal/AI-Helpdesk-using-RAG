#!/usr/bin/env python3
"""Evaluation script for the AI Campus Helpdesk.

Runs the evaluation question set (evaluation/questions.json) against three
experimental configurations, so the research paper has real, comparable
numbers instead of invented ones:

  Experiment A: Plain LLM, no retrieval at all (tests raw model knowledge of
                a knowledge base it has never seen).
  Experiment B: LLM + RAG, but with NO department routing (single retrieval
                pool across all documents).
  Experiment C: Full system - Multi-Agent Router + department-scoped RAG.

Metrics computed (documented limitations below):
  - Routing accuracy (Experiment C only): did the router pick the expected
    department?
  - Retrieval success rate (Experiments B & C): does the expected source
    document appear among the retrieved chunks?
  - Answer correctness: a simple, automated KEYWORD/CONCEPT overlap check -
    the fraction of `expected_concepts` that appear (case-insensitive
    substring match) in the generated answer. This is NOT a semantic
    evaluator and will under- or over-count paraphrased answers. It is a
    lightweight proxy suitable for an intermediate student project, not a
    publication-grade evaluation.
  - Unsupported-answer rate: for the 2 deliberately out-of-domain questions
    (expected_source == "none"), whether the system correctly refused to
    answer instead of inventing information.
  - Average response time per question.

Usage:
    python evaluation/evaluate.py                 # runs all 3 experiments
    python evaluation/evaluate.py --experiment c   # runs a single experiment
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.router import get_agent_for_department, route
from app.llm.ollama_client import LLMUnavailableError, generate
from app.llm.prompts import RAG_ANSWER_PROMPT_TEMPLATE
from app.rag.retriever import retrieve
from app.utils.config import TOP_K
from app.utils.logger import get_logger

logger = get_logger("evaluate")

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

REFUSAL_MARKERS = ["could not find", "not available", "cannot find", "unable to find"]


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(marker in lower for marker in REFUSAL_MARKERS)


def concept_score(answer: str, expected_concepts: list[str]) -> float:
    if not expected_concepts:
        return 0.0
    lower = answer.lower()
    hits = sum(1 for c in expected_concepts if c.lower() in lower)
    return hits / len(expected_concepts)


# --- Experiment A: plain LLM, no RAG at all ---
def run_experiment_a(questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        start = time.perf_counter()
        try:
            answer = generate(f"Answer this student's question: {q['question']}")
        except LLMUnavailableError as exc:
            answer = f"[LLM_ERROR: {exc}]"
        elapsed = time.perf_counter() - start

        is_oos = q["expected_source"] == "none"
        rows.append(
            {
                "question": q["question"],
                "expected_department": q["expected_department"],
                "predicted_department": "",
                "routing_correct": "",
                "expected_source": q["expected_source"],
                "retrieval_hit": "",
                "answer": answer,
                "concept_score": round(concept_score(answer, q["expected_concepts"]), 2),
                "correctly_refused_oos": is_refusal(answer) if is_oos else "",
                "response_time_sec": round(elapsed, 3),
            }
        )
    return rows


# --- Experiment B: LLM + RAG, no department routing (single pool retrieval) ---
def run_experiment_b(questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        start = time.perf_counter()
        retrieved = retrieve(q["question"], department=None, top_k=TOP_K)
        if retrieved:
            context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in retrieved)
            prompt = RAG_ANSWER_PROMPT_TEMPLATE.format(
                agent_name="Campus Helpdesk Assistant",
                department="General",
                context=context,
                question=q["question"],
            )
            try:
                answer = generate(prompt)
            except LLMUnavailableError as exc:
                answer = f"[LLM_ERROR: {exc}]"
        else:
            answer = "I could not find this information in the current campus knowledge base."
        elapsed = time.perf_counter() - start

        sources = {c.source for c in retrieved}
        retrieval_hit = q["expected_source"] in sources if q["expected_source"] != "none" else (len(sources) == 0 or True)
        is_oos = q["expected_source"] == "none"
        rows.append(
            {
                "question": q["question"],
                "expected_department": q["expected_department"],
                "predicted_department": "",
                "routing_correct": "",
                "expected_source": q["expected_source"],
                "retrieval_hit": retrieval_hit if not is_oos else "",
                "answer": answer,
                "concept_score": round(concept_score(answer, q["expected_concepts"]), 2),
                "correctly_refused_oos": is_refusal(answer) if is_oos else "",
                "response_time_sec": round(elapsed, 3),
            }
        )
    return rows


# --- Experiment C: full multi-agent system (Router + department-scoped RAG) ---
def run_experiment_c(questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        start = time.perf_counter()
        predicted_department = route(q["question"])
        agent = get_agent_for_department(predicted_department)
        result = agent.answer(q["question"])
        elapsed = time.perf_counter() - start

        sources = {c.source for c in result.retrieved}
        is_oos = q["expected_source"] == "none"
        retrieval_hit = q["expected_source"] in sources if not is_oos else ""
        routing_correct = predicted_department == q["expected_department"]

        rows.append(
            {
                "question": q["question"],
                "expected_department": q["expected_department"],
                "predicted_department": predicted_department,
                "routing_correct": routing_correct,
                "expected_source": q["expected_source"],
                "retrieval_hit": retrieval_hit,
                "answer": result.answer,
                "concept_score": round(concept_score(result.answer, q["expected_concepts"]), 2),
                "correctly_refused_oos": is_refusal(result.answer) if is_oos else "",
                "response_time_sec": round(elapsed, 3),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], experiment_name: str) -> dict:
    total = len(rows)
    routing_rows = [r for r in rows if r["routing_correct"] != ""]
    retrieval_rows = [r for r in rows if r["retrieval_hit"] != ""]
    oos_rows = [r for r in rows if r["correctly_refused_oos"] != ""]

    routing_accuracy = (
        sum(1 for r in routing_rows if r["routing_correct"] is True) / len(routing_rows)
        if routing_rows
        else None
    )
    retrieval_success_rate = (
        sum(1 for r in retrieval_rows if r["retrieval_hit"] is True) / len(retrieval_rows)
        if retrieval_rows
        else None
    )
    avg_concept_score = sum(r["concept_score"] for r in rows) / total if total else 0.0
    avg_response_time = sum(r["response_time_sec"] for r in rows) / total if total else 0.0
    unsupported_correct_refusal_rate = (
        sum(1 for r in oos_rows if r["correctly_refused_oos"] is True) / len(oos_rows)
        if oos_rows
        else None
    )

    return {
        "experiment": experiment_name,
        "total_questions": total,
        "routing_accuracy": round(routing_accuracy, 3) if routing_accuracy is not None else "N/A",
        "retrieval_success_rate": round(retrieval_success_rate, 3) if retrieval_success_rate is not None else "N/A",
        "answer_accuracy_concept_overlap": round(avg_concept_score, 3),
        "average_response_time_sec": round(avg_response_time, 3),
        "out_of_domain_correct_refusal_rate": round(unsupported_correct_refusal_rate, 3)
        if unsupported_correct_refusal_rate is not None
        else "N/A",
    }


EXPERIMENTS = {
    "a": ("Experiment A - Plain LLM (no RAG)", run_experiment_a, "experiment_a_plain_llm.csv"),
    "b": ("Experiment B - LLM + RAG (no routing)", run_experiment_b, "experiment_b_rag_no_routing.csv"),
    "c": ("Experiment C - Multi-Agent + RAG (full system)", run_experiment_c, "evaluation_results.csv"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the AI Campus Helpdesk.")
    parser.add_argument(
        "--experiment", choices=["a", "b", "c", "all"], default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N questions (useful for a quick run on slow, CPU-only "
        "local models). Default: run all questions.",
    )
    args = parser.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    logger.info("Loaded %d evaluation questions (limit=%s).", len(questions), args.limit)

    keys = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]
    summaries = []

    for key in keys:
        label, run_fn, filename = EXPERIMENTS[key]
        logger.info("Running %s ...", label)
        rows = run_fn(questions)
        out_path = RESULTS_DIR / filename
        write_csv(rows, out_path)
        logger.info("Wrote results to %s", out_path)

        summary = summarize(rows, label)
        summaries.append(summary)
        logger.info("Summary for %s: %s", label, summary)

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Wrote overall summary to %s", summary_path)

    print("\n=== EVALUATION SUMMARY ===")
    for s in summaries:
        print(f"\n{s['experiment']}")
        for k, v in s.items():
            if k != "experiment":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
