# evaluation/full_eval.py
# Unified evaluation — runs BOTH our custom metrics AND RAGAS in one shot.
# Produces a single combined report.
#
# Usage:
#   python evaluation/full_eval.py
#   python evaluation/full_eval.py --questions 5   ← quick 5-question test
#   python evaluation/full_eval.py --mode hybrid --no-rerank

import argparse
import json
import time
from pathlib import Path
from typing import List

import sys
sys.path.append(str(Path(__file__).parent.parent))


def run_full_evaluation(
    backend:      str  = "chromadb",
    use_reranker: bool = True,
    search_mode:  str  = "hybrid",
    n_questions:  int  = None,
) -> None:

    from evaluation.evaluator  import run_evaluation, print_report, save_results
    from evaluation.ragas_eval import run_ragas_evaluation, print_ragas_report, save_ragas_results

    q_path    = str(Path(__file__).parent / "test_questions.json")
    questions = json.loads(Path(q_path).read_text())
    if n_questions:
        questions = questions[:n_questions]

    label = f"{search_mode.upper()} {'+ rerank' if use_reranker else '(no rerank)'}"

    print(f"\n{'█'*62}")
    print(f"  FULL EVALUATION  —  {label}")
    print(f"  {len(questions)} questions  |  backend={backend}")
    print(f"{'█'*62}")

    # ── Part 1: Custom metrics (fast, no extra LLM calls) ─────────────────────
    print(f"\n{'─'*62}")
    print(f"  PART 1: Custom Metrics (Precision / Recall / Keyword Coverage)")
    print(f"{'─'*62}")
    custom_results = run_evaluation(
        mode         = search_mode,
        use_reranker = use_reranker,
        backend      = backend,
        questions_path = q_path,
    )
    custom_summary = print_report(custom_results, label=label)
    save_results(custom_results, f"evaluation/results_{search_mode}.json")

    # ── Part 2: RAGAS (LLM-as-judge, richer but slower) ───────────────────────
    print(f"\n{'─'*62}")
    print(f"  PART 2: RAGAS Metrics (LLM-as-Judge)")
    print(f"{'─'*62}")
    try:
        ragas_results, elapsed, raw = run_ragas_evaluation(
            backend        = backend,
            use_reranker   = use_reranker,
            search_mode    = search_mode,
            n_questions    = n_questions,
            questions_path = q_path,
        )
        print_ragas_report(ragas_results, elapsed, raw, label=label)
        save_ragas_results(ragas_results, f"evaluation/ragas_results_{search_mode}.json")

        # ── Combined summary ───────────────────────────────────────────────────
        df = ragas_results.to_pandas()
        print(f"\n{'='*62}")
        print(f"  COMBINED SUMMARY  —  {label}")
        print(f"{'='*62}")
        print(f"  Custom Metrics:")
        print(f"    Retrieval F1        : {custom_summary.get('avg_f1', 0):.3f}")
        print(f"    Keyword Coverage    : {custom_summary.get('avg_kw_coverage', 0):.3f}")
        print(f"    File Citation Rate  : {custom_summary.get('citation_rate', 0):.3f}")
        print(f"    Avg Latency         : {custom_summary.get('avg_latency_sec', 0):.2f}s")
        print(f"\n  RAGAS Metrics (LLM-as-Judge):")
        for col in ["faithfulness","answer_relevancy","context_precision","context_recall"]:
            if col in df.columns:
                print(f"    {col:<22}: {df[col].mean():.3f}")
        print(f"{'='*62}\n")

    except ImportError:
        print("\n⚠️  RAGAS not installed. Run:  pip install ragas")
        print("   Skipping RAGAS evaluation.\n")
    except Exception as e:
        print(f"\n❌ RAGAS evaluation failed: {e}")
        print("   Custom metrics above are still valid.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full evaluation: custom + RAGAS")
    parser.add_argument("--backend",    default="chromadb", choices=["chromadb","faiss"])
    parser.add_argument("--mode",       default="hybrid",   choices=["hybrid","semantic","bm25"])
    parser.add_argument("--no-rerank",  action="store_true")
    parser.add_argument("--questions",  type=int, default=None,
                        help="Only evaluate first N questions (faster for testing)")
    args = parser.parse_args()

    run_full_evaluation(
        backend      = args.backend,
        use_reranker = not args.no_rerank,
        search_mode  = args.mode,
        n_questions  = args.questions,
    )