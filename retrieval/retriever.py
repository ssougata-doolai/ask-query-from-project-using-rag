# retrieval/retriever.py
# Semantic + BM25 hybrid search with Reciprocal Rank Fusion (RRF).
#
# Search modes:
#   "semantic" → dense vector search only (original)
#   "bm25"     → keyword search only
#   "hybrid"   → both combined via RRF (default, best quality)

import os
os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"]               = "0"
os.environ["USE_TORCH"]            = "1"

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import TOP_K, VECTOR_BACKEND
from ingestion.embedder import BaseVectorStore, get_vector_store, load_embedding_model
from retrieval.bm25 import BM25Index


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text:          str
    score:         float          # final hybrid / semantic / bm25 score
    relative_path: str
    node_name:     str
    node_type:     str
    language:      str
    start_line:    int
    end_line:      int
    semantic_score: float = 0.0   # raw cosine similarity
    bm25_score:     float = 0.0   # raw BM25 score
    search_mode:    str   = ""    # "semantic" | "bm25" | "hybrid"

    def format_source(self) -> str:
        loc = f"lines {self.start_line}–{self.end_line}" if self.start_line else ""
        if self.node_name:
            return f"{self.relative_path}  →  {self.node_name}  ({loc})"
        return f"{self.relative_path}  ({loc})"


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    semantic_results: List[dict],
    bm25_results:     List[dict],
    k:                int = 60,
    semantic_weight:  float = 0.7,
    bm25_weight:      float = 0.3,
) -> List[dict]:
    """
    Combine semantic and BM25 results using Reciprocal Rank Fusion.

    RRF score = semantic_weight * (1 / (k + rank_s))
              + bm25_weight    * (1 / (k + rank_b))

    k=60 is the standard RRF constant (dampens the impact of rank differences).
    Weights: semantic 0.7, BM25 0.3 — semantic usually better for code Q&A.

    Uses (relative_path + node_name) as the deduplication key so the same
    chunk doesn't appear twice from both result lists.
    """
    def chunk_key(r: dict) -> str:
        return f"{r.get('relative_path','')}::{r.get('node_name','')}::{r.get('start_line','')}"

    # Build rank maps
    sem_rank  = {chunk_key(r): (i + 1, r) for i, r in enumerate(semantic_results)}
    bm25_rank = {chunk_key(r): (i + 1, r) for i, r in enumerate(bm25_results)}

    # Union of all keys
    all_keys = set(sem_rank.keys()) | set(bm25_rank.keys())

    fused = []
    for key in all_keys:
        sem_r,  sem_result  = sem_rank.get(key,  (len(semantic_results) + k, None))
        bm25_r, bm25_result = bm25_rank.get(key, (len(bm25_results)    + k, None))

        rrf_score = (semantic_weight / (k + sem_r)) + (bm25_weight / (k + bm25_r))

        # Use whichever result dict is available
        result = sem_result or bm25_result
        fused.append({
            **result,
            "hybrid_score":   round(rrf_score, 6),
            "semantic_score": round(sem_result["score"]       if sem_result  else 0.0, 4),
            "bm25_score":     round(bm25_result["bm25_score"] if bm25_result else 0.0, 4),
        })

    fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────

SearchMode = Literal["semantic", "bm25", "hybrid"]

class CodebaseRetriever:
    """
    Hybrid retriever: semantic + BM25 combined via Reciprocal Rank Fusion.

    Usage:
        retriever = CodebaseRetriever()

        # Hybrid (default — best quality)
        results = retriever.search("how does httpx handle redirects?")

        # Semantic only
        results = retriever.search("timeout config", mode="semantic")

        # BM25 only (fast, keyword-exact)
        results = retriever.search("AsyncClient.send", mode="bm25")
    """

    def __init__(
        self,
        backend:          str = VECTOR_BACKEND,
        semantic_weight:  float = 0.7,
        bm25_weight:      float = 0.3,
    ):
        print(f"[Retriever] 🔄 Initializing  (backend={backend})")
        self.semantic_weight = semantic_weight
        self.bm25_weight     = bm25_weight

        # Dense semantic store
        self.model   = load_embedding_model()
        self.store   = get_vector_store(backend)
        self.store.load()

        # Sparse BM25 index
        self.bm25_idx = BM25Index()
        try:
            self.bm25_idx.load()
            self._bm25_ready = True
        except FileNotFoundError:
            print("[Retriever] ⚠️  BM25 index not found — falling back to semantic only.")
            print("            Re-run: python main.py index  to build the BM25 index.")
            self._bm25_ready = False

        print(f"[Retriever] ✅ Ready  ({self.store.count} chunks  |  BM25={'✓' if self._bm25_ready else '✗'})")


    def search(
        self,
        query:            str,
        top_k:            int = TOP_K,
        mode:             SearchMode = "hybrid",
        language:         Optional[str] = None,
        path_prefix:      Optional[str] = None,
        min_score:        float = 0.0,
        semantic_weight:  Optional[float] = None,
        bm25_weight:      Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Search codebase using the specified mode.

        Args:
            query:           Natural language or keyword question.
            top_k:           Max results to return.
            mode:            "hybrid" | "semantic" | "bm25"
            language:        Filter to a specific language.
            path_prefix:     Filter to files whose path contains this string.
            min_score:       Min similarity threshold (semantic/hybrid only).
            semantic_weight: Override default semantic weight in fusion (0–1).
            bm25_weight:     Override default BM25 weight in fusion (0–1).
        """
        sw = semantic_weight or self.semantic_weight
        bw = bm25_weight     or self.bm25_weight

        # Fall back to semantic if BM25 not ready
        if mode in ("hybrid", "bm25") and not self._bm25_ready:
            mode = "semantic"

        fetch_k = top_k * 3   # fetch more than needed before merging/filtering

        # ── Semantic search ────────────────────────────────────────────────────
        sem_results = []
        if mode in ("semantic", "hybrid"):
            q_vec = self.model.encode(query, normalize_embeddings=True).astype("float32")
            sem_results = self.store.search(
                query_vector = q_vec,
                top_k        = fetch_k,
                language     = language,
                path_prefix  = path_prefix,
                min_score    = min_score,
            )

        # ── BM25 search ────────────────────────────────────────────────────────
        bm25_results = []
        if mode in ("bm25", "hybrid"):
            bm25_results = self.bm25_idx.search(
                query       = query,
                top_k       = fetch_k,
                language    = language,
                path_prefix = path_prefix,
            )

        # ── Combine / select ───────────────────────────────────────────────────
        if mode == "hybrid":
            raw = reciprocal_rank_fusion(sem_results, bm25_results,
                                         semantic_weight=sw, bm25_weight=bw)
            score_key = "hybrid_score"
        elif mode == "semantic":
            raw = [{**r, "hybrid_score": r["score"], "semantic_score": r["score"], "bm25_score": 0.0}
                   for r in sem_results]
            score_key = "hybrid_score"
        else:  # bm25
            raw = [{**r, "hybrid_score": r["bm25_score"], "semantic_score": 0.0}
                   for r in bm25_results]
            score_key = "hybrid_score"

        # Take top_k and convert to RetrievedChunk
        return [
            RetrievedChunk(
                text           = r["text"],
                score          = r[score_key],
                relative_path  = r.get("relative_path", ""),
                node_name      = r.get("node_name", ""),
                node_type      = r.get("node_type", ""),
                language       = r.get("language", ""),
                start_line     = r.get("start_line", 0),
                end_line       = r.get("end_line", 0),
                semantic_score = r.get("semantic_score", 0.0),
                bm25_score     = r.get("bm25_score", 0.0),
                search_mode    = mode,
            )
            for r in raw[:top_k]
        ]


    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format chunks into a single LLM-ready context block."""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}] {c.format_source()}\n"
                f"```{c.language}\n{c.text}\n```"
            )
        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = CodebaseRetriever()

    test_cases = [
        ("how does httpx handle timeouts?",        "hybrid"),
        ("AsyncClient send method",                "hybrid"),
        ("authentication bearer token",            "hybrid"),
        ("AsyncClient",                            "bm25"),     # exact name → BM25 wins
        ("connection retry mechanism",             "semantic"), # concept → semantic wins
    ]

    for query, mode in test_cases:
        print(f"\n{'='*60}")
        print(f"Query : {query}")
        print(f"Mode  : {mode}")
        print("=" * 60)
        results = retriever.search(query, top_k=3, mode=mode)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r.score:.4f}  sem={r.semantic_score:.3f}  bm25={r.bm25_score:.3f}")
            print(f"       {r.format_source()}")