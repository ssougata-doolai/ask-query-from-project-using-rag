# evaluation/ragas_eval.py
# RAGAS evaluation — uses LLM-as-judge to measure:
#   1. Faithfulness       — did the answer hallucinate?
#   2. Answer Relevancy   — does the answer address the question?
#   3. Context Precision  — were retrieved chunks actually useful?
#   4. Context Recall     — did retrieved chunks cover the answer?
#
# Judge LLM: Groq (Llama 3.3 70B) — free, same API key you already have
#
# Usage:
#   python evaluation/ragas_eval.py
#   python evaluation/ragas_eval.py --questions 5   ← run on first 5 only (faster)
#   python evaluation/ragas_eval.py --no-rerank

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict

import sys
sys.path.append(str(Path(__file__).parent.parent))

os.environ["RAGAS_DO_NOT_TRACK"] = "true"   # disable RAGAS telemetry


# ─────────────────────────────────────────────────────────────────────────────
# Setup RAGAS with Groq as judge
# ─────────────────────────────────────────────────────────────────────────────

def build_ragas_llm():
    """
    Configure RAGAS to use Groq (Llama 3.3 70B) as the judge LLM.
    Uses the same GROQ_API_KEY from your .env file.
    """
    from groq import Groq
    from ragas.llms import llm_factory
    from config import GROQ_API_KEY, LLM_MODEL

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    client = Groq(api_key=GROQ_API_KEY)
    llm    = llm_factory(LLM_MODEL, provider="groq", client=client)
    print(f"[RAGAS] ✅ Judge LLM: groq/{LLM_MODEL}")
    return llm


def build_ragas_embeddings():
    """
    Use our local sentence-transformer model for RAGAS embeddings.
    (Needed for Answer Relevancy metric — embeds question + answer to compare)
    """
    from ragas.embeddings import BaseRagasEmbeddings
    from sentence_transformers import SentenceTransformer
    from config import EMBEDDING_MODEL
    import numpy as np

    class LocalEmbeddings(BaseRagasEmbeddings):
        def __init__(self):
            self.model = SentenceTransformer(EMBEDDING_MODEL)

        def embed_text(self, text: str) -> List[float]:
            return self.model.encode(text, normalize_embeddings=True).tolist()

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self.model.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text: str) -> List[float]:
            return self.embed_text(text)

    embeddings = LocalEmbeddings()
    print(f"[RAGAS] ✅ Embeddings: local/{EMBEDDING_MODEL}")
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Build RAGAS dataset from our QA pipeline output
# ─────────────────────────────────────────────────────────────────────────────

def collect_ragas_data(
    questions:    List[dict],
    backend:      str  = "chromadb",
    use_reranker: bool = True,
    search_mode:  str  = "hybrid",
) -> tuple:
    """
    Run all questions through the RAG pipeline and collect:
        user_input   : the question
        response     : the LLM answer
        retrieved_contexts : list of retrieved chunk texts
        reference    : ground truth (we use expected keywords joined as sentence)

    Returns (ragas_dataset, raw_results) where raw_results has timing info.
    """
    from ragas import EvaluationDataset, SingleTurnSample
    from generation.qa_chain import CodebaseQAChain

    chain      = CodebaseQAChain(
        backend      = backend,
        search_mode  = search_mode,
        use_reranker = use_reranker,
    )
    samples    = []
    raw        = []

    print(f"\n[RAGAS] Collecting answers for {len(questions)} questions...")

    for i, q in enumerate(questions, 1):
        print(f"  [{i:>2}/{len(questions)}] {q['question'][:60]}...")
        t0 = time.time()

        try:
            result = chain.ask(q["question"], min_score=0.0)

            # Build a simple reference from expected keywords
            # (In production you'd have real gold answers here)
            reference = (
                q.get("reference_answer") or
                f"httpx handles this using: {', '.join(q.get('expected_keywords', []))}"
            )

            sample = SingleTurnSample(
                user_input         = q["question"],
                response           = result.answer,
                retrieved_contexts = [s.text for s in result.sources],
                reference          = reference,
            )
            samples.append(sample)
            raw.append({
                "id":      q["id"],
                "latency": round(time.time() - t0, 2),
                "sources": len(result.sources),
            })
            print(f"         ✅  {len(result.sources)} sources  |  {time.time()-t0:.1f}s")

        except Exception as e:
            print(f"         ❌  ERROR: {e}")
            # Add a placeholder so indices stay aligned
            samples.append(SingleTurnSample(
                user_input         = q["question"],
                response           = f"ERROR: {e}",
                retrieved_contexts = [],
                reference          = "",
            ))
            raw.append({"id": q["id"], "latency": 0, "sources": 0, "error": str(e)})

    dataset = EvaluationDataset(samples=samples)
    return dataset, raw


# ─────────────────────────────────────────────────────────────────────────────
# Run RAGAS evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas_evaluation(
    backend:      str   = "chromadb",
    use_reranker: bool  = True,
    search_mode:  str   = "hybrid",
    n_questions:  int   = None,
    questions_path: str = None,
) -> dict:
    """
    Full RAGAS evaluation pipeline.
    Returns dict of metric scores.
    """
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )

    # Load questions
    q_path    = questions_path or str(Path(__file__).parent / "test_questions.json")
    questions = json.loads(Path(q_path).read_text())
    if n_questions:
        questions = questions[:n_questions]

    # Setup judge LLM + embeddings
    judge_llm  = build_ragas_llm()
    embeddings = build_ragas_embeddings()

    # Define metrics
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    # Collect pipeline outputs
    dataset, raw = collect_ragas_data(
        questions    = questions,
        backend      = backend,
        use_reranker = use_reranker,
        search_mode  = search_mode,
    )

    # Run RAGAS evaluation
    print(f"\n[RAGAS] 🔍 Running LLM-as-judge evaluation ({len(questions)} questions)...")
    print(f"[RAGAS]    This makes {len(questions) * len(metrics)} LLM judge calls.")
    print(f"[RAGAS]    Estimated time: {len(questions) * 8:.0f}–{len(questions) * 15:.0f}s\n")

    t0      = time.time()
    results = evaluate(dataset=dataset, metrics=metrics)
    elapsed = time.time() - t0

    return results, elapsed, raw


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_ragas_report(results, elapsed: float, raw: list, label: str = "") -> None:
    """Pretty-print RAGAS results."""
    title = f"RAGAS Evaluation Report — {label}" if label else "RAGAS Evaluation Report"

    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")
    print(f"  Evaluation time : {elapsed:.1f}s")
    print(f"  Judge LLM       : groq/llama-3.3-70b-versatile")
    print(f"{'─'*62}")

    # Core metrics
    score_map = {
        "faithfulness":     ("Faithfulness",       "Was answer grounded in context? (no hallucination)"),
        "answer_relevancy": ("Answer Relevancy",    "Did answer actually address the question?"),
        "context_precision":("Context Precision",   "Were retrieved chunks relevant?"),
        "context_recall":   ("Context Recall",      "Did chunks cover everything needed to answer?"),
    }

    df = results.to_pandas()

    print(f"  {'Metric':<22} {'Score':>7}  Description")
    print(f"  {'─'*58}")
    for key, (name, desc) in score_map.items():
        if key in df.columns:
            score = df[key].mean()
            bar   = "█" * int(score * 20)
            grade = "✅" if score >= 0.7 else ("⚠️ " if score >= 0.4 else "❌")
            print(f"  {grade} {name:<20} {score:>6.3f}  {bar}")
    print(f"{'─'*62}")

    # Per-question breakdown
    print(f"\n  Per-question scores:")
    cols = [c for c in ["faithfulness","answer_relevancy","context_precision","context_recall"] if c in df.columns]
    header = f"  {'Q':<6} " + " ".join(f"{c[:8]:>8}" for c in cols)
    print(header)
    print(f"  {'─'*50}")
    for i, row in df.iterrows():
        q_id  = f"q{i+1:03d}"
        scores = " ".join(f"{row[c]:>8.3f}" if c in row else f"{'N/A':>8}" for c in cols)
        print(f"  {q_id:<6} {scores}")

    print(f"\n{'='*62}")

    # Interpretation guide
    print(f"""
  INTERPRETATION GUIDE
  ─────────────────────────────────────────────────────────
  Faithfulness (0–1):
    1.0 = every claim in answer is supported by sources
    0.0 = answer is completely hallucinated
    → Your target: > 0.80

  Answer Relevancy (0–1):
    1.0 = answer directly addresses the question
    0.0 = answer is off-topic
    → Your target: > 0.75

  Context Precision (0–1):
    1.0 = all retrieved chunks were useful
    0.0 = all retrieved chunks were noise
    → Your target: > 0.60

  Context Recall (0–1):
    1.0 = retrieved chunks contained all info needed
    0.0 = retrieved chunks missed critical information
    → Your target: > 0.65
  {'─'*57}
  Score ≥ 0.70  ✅ Good    |  0.40–0.70  ⚠️  Needs work  |  < 0.40  ❌ Poor
    """)


def save_ragas_results(results, output_path: str) -> None:
    """Save RAGAS results to CSV and JSON."""
    df = results.to_pandas()
    csv_path = output_path.replace(".json", ".csv")
    df.to_csv(csv_path, index=False)
    df.to_json(output_path, orient="records", indent=2)
    print(f"[RAGAS] 💾 Results saved to {output_path} and {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation of the RAG pipeline")
    parser.add_argument("--backend",    default="chromadb", choices=["chromadb","faiss"])
    parser.add_argument("--mode",       default="hybrid",   choices=["hybrid","semantic","bm25"])
    parser.add_argument("--no-rerank",  action="store_true")
    parser.add_argument("--questions",  type=int, default=None,
                        help="Only evaluate first N questions (default: all)")
    args = parser.parse_args()

    use_reranker = not args.no_rerank

    try:
        import ragas
    except ImportError:
        print("RAGAS not installed. Run:  pip install ragas")
        sys.exit(1)

    results, elapsed, raw = run_ragas_evaluation(
        backend      = args.backend,
        use_reranker = use_reranker,
        search_mode  = args.mode,
        n_questions  = args.questions,
    )

    label = f"{args.mode.upper()} {'+ rerank' if use_reranker else '(no rerank)'}"
    print_ragas_report(results, elapsed, raw, label=label)

    out_path = f"evaluation/ragas_results_{args.mode}.json"
    save_ragas_results(results, out_path)