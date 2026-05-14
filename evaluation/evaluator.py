# evaluation/evaluator.py
# Runs all test questions through the RAG pipeline and produces a report.
#
# Usage:
#   python evaluation/evaluator.py
#   python evaluation/evaluator.py --mode hybrid
#   python evaluation/evaluator.py --mode semantic --no-rerank
#   python evaluation/evaluator.py --compare   ← runs all modes side by side

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import (
    EvalResult, compute_retrieval_metrics,
    compute_keyword_coverage, has_file_citation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluator
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    mode:        str  = "hybrid",
    use_reranker: bool = True,
    backend:     str  = "chromadb",
    questions_path: str = None,
    verbose:     bool = False,
) -> List[EvalResult]:
    """
    Run all test questions through the RAG pipeline and return EvalResult list.
    """
    from generation.qa_chain import CodebaseQAChain

    # Load test questions
    q_path = questions_path or str(Path(__file__).parent / "test_questions.json")
    questions = json.loads(Path(q_path).read_text())

    print(f"\n{'='*60}")
    print(f"Evaluating: mode={mode}  reranker={use_reranker}  backend={backend}")
    print(f"Questions : {len(questions)}")
    print(f"{'='*60}\n")

    # Init chain
    chain = CodebaseQAChain(backend=backend, search_mode=mode, use_reranker=use_reranker)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i:>2}/{len(questions)}] {q['question'][:60]}...")

        t0 = time.time()
        try:
            result = chain.ask(
                question  = q["question"],
                min_score = 0.0,   # don't filter during eval — we want to measure all
            )
            latency = time.time() - t0

            retrieved_files = [s.relative_path for s in result.sources]

            precision, recall = compute_retrieval_metrics(
                retrieved_files, q.get("expected_files", [])
            )
            kw_coverage = compute_keyword_coverage(
                result.answer, q.get("expected_keywords", [])
            )
            citation = has_file_citation(result.answer)

            er = EvalResult(
                question_id         = q["id"],
                question            = q["question"],
                answer              = result.answer,
                latency_sec         = latency,
                retrieved_files     = retrieved_files,
                expected_files      = q.get("expected_files", []),
                retrieval_precision = precision,
                retrieval_recall    = recall,
                keyword_coverage    = kw_coverage,
                has_file_citation   = citation,
                answer_length       = len(result.answer.split()),
                num_sources         = len(result.sources),
                search_mode         = mode,
            )

            status = "✅" if precision > 0.3 and kw_coverage > 0.4 else "⚠️ "
            print(f"       {status}  prec={precision:.2f}  recall={recall:.2f}  kw={kw_coverage:.2f}  latency={latency:.1f}s")

            if verbose:
                print(f"       Answer: {result.answer[:150]}...")
                print(f"       Files : {retrieved_files[:3]}")

        except Exception as e:
            print(f"       ❌ ERROR: {e}")
            er = EvalResult(
                question_id = q["id"],
                question    = q["question"],
                answer      = f"ERROR: {e}",
                latency_sec = time.time() - t0,
                search_mode = mode,
            )

        results.append(er)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report generator
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: List[EvalResult], label: str = "") -> Dict:
    """Print a summary report and return aggregate metrics."""
    n = len(results)
    if n == 0:
        return {}

    avg_precision  = sum(r.retrieval_precision for r in results) / n
    avg_recall     = sum(r.retrieval_recall    for r in results) / n
    avg_f1         = sum(r.f1_retrieval        for r in results) / n
    avg_kw         = sum(r.keyword_coverage    for r in results) / n
    avg_latency    = sum(r.latency_sec         for r in results) / n
    citation_rate  = sum(1 for r in results if r.has_file_citation) / n
    avg_sources    = sum(r.num_sources         for r in results) / n

    title = f"Evaluation Report — {label}" if label else "Evaluation Report"
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Questions evaluated : {n}")
    print(f"  Search mode         : {results[0].search_mode}")
    print(f"{'─'*60}")
    print(f"  RETRIEVAL")
    print(f"    Avg Precision     : {avg_precision:.3f}  (relevant chunks / retrieved)")
    print(f"    Avg Recall        : {avg_recall:.3f}  (expected files found)")
    print(f"    Avg F1            : {avg_f1:.3f}")
    print(f"{'─'*60}")
    print(f"  ANSWER QUALITY")
    print(f"    Keyword Coverage  : {avg_kw:.3f}  (expected terms in answer)")
    print(f"    File Citation Rate: {citation_rate:.3f}  (answers mentioning files)")
    print(f"    Avg Answer Length : {sum(r.answer_length for r in results)/n:.0f} words")
    print(f"    Avg Sources Used  : {avg_sources:.1f} chunks")
    print(f"{'─'*60}")
    print(f"  PERFORMANCE")
    print(f"    Avg Latency       : {avg_latency:.2f}s per query")
    print(f"    Total Time        : {sum(r.latency_sec for r in results):.1f}s")
    print(f"{'='*60}\n")

    # Per-question breakdown
    print(f"  Per-question results:")
    print(f"  {'ID':<6} {'P':>5} {'R':>5} {'F1':>5} {'KW':>5} {'Cite':>5} {'Lat':>6}")
    print(f"  {'─'*45}")
    for r in results:
        cite = "✓" if r.has_file_citation else "✗"
        print(f"  {r.question_id:<6} {r.retrieval_precision:>5.2f} {r.retrieval_recall:>5.2f} "
              f"{r.f1_retrieval:>5.2f} {r.keyword_coverage:>5.2f} {cite:>5} {r.latency_sec:>5.1f}s")

    return {
        "label":            label,
        "n":                n,
        "avg_precision":    round(avg_precision,  3),
        "avg_recall":       round(avg_recall,     3),
        "avg_f1":           round(avg_f1,         3),
        "avg_kw_coverage":  round(avg_kw,         3),
        "citation_rate":    round(citation_rate,  3),
        "avg_latency_sec":  round(avg_latency,    2),
    }


def save_results(results: List[EvalResult], path: str) -> None:
    """Save detailed results to JSON."""
    data = [r.to_dict() for r in results]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[Eval] Results saved to {path}")


def compare_modes(backend: str = "chromadb", use_reranker: bool = True) -> None:
    """Run evaluation on all three search modes and compare."""
    summaries = []
    all_results = {}

    for mode in ["hybrid", "semantic", "bm25"]:
        results = run_evaluation(mode=mode, use_reranker=use_reranker, backend=backend)
        summary = print_report(results, label=mode.upper())
        summaries.append(summary)
        all_results[mode] = results
        save_results(results, f"evaluation/results_{mode}.json")

    # Side-by-side comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON: hybrid vs semantic vs bm25")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'Hybrid':>8} {'Semantic':>8} {'BM25':>8}")
    print(f"  {'─'*50}")
    metrics = [
        ("Avg Precision",    "avg_precision"),
        ("Avg Recall",       "avg_recall"),
        ("Avg F1",           "avg_f1"),
        ("Keyword Coverage", "avg_kw_coverage"),
        ("Citation Rate",    "citation_rate"),
        ("Avg Latency (s)",  "avg_latency_sec"),
    ]
    for label, key in metrics:
        vals = [s.get(key, 0) for s in summaries]
        best = max(range(3), key=lambda i: vals[i] if key != "avg_latency_sec" else -vals[i])
        row  = f"  {label:<22}"
        for i, v in enumerate(vals):
            marker = " ←" if i == best else "  "
            row += f" {v:>7.3f}{marker}"[:-1] if key != "avg_latency_sec" else f" {v:>7.2f}"
        print(row)
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--mode",      default="hybrid", choices=["hybrid", "semantic", "bm25"])
    parser.add_argument("--backend",   default="chromadb", choices=["chromadb", "faiss"])
    parser.add_argument("--no-rerank", action="store_true", help="Disable re-ranker")
    parser.add_argument("--compare",   action="store_true", help="Compare all 3 modes")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    use_reranker = not args.no_rerank

    if args.compare:
        compare_modes(backend=args.backend, use_reranker=use_reranker)
    else:
        results = run_evaluation(
            mode         = args.mode,
            use_reranker = use_reranker,
            backend      = args.backend,
            verbose      = args.verbose,
        )
        print_report(results, label=args.mode.upper())
        save_results(results, f"evaluation/results_{args.mode}.json")