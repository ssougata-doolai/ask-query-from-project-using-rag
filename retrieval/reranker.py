# retrieval/reranker.py
# Cross-encoder re-ranking — the quality improvement layer after retrieval.
#
# How it works:
#   Retrieval (fast)  : bi-encoder embeds query + docs SEPARATELY → cosine sim
#   Re-ranking (slow) : cross-encoder reads query + doc TOGETHER → deeper score
#
# The cross-encoder sees both the query and the chunk at the same time,
# so it can model interaction between them (e.g. "timeout" in query matching
# "connect_timeout" in code). This gives much better relevance scores.
#
# Trade-off: slower (runs inference per chunk) — only on top-K from retrieval,
# not on the full corpus. Typical flow:
#   Retrieve top-20 → Re-rank → Return top-5

import os
os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"]               = "0"
os.environ["USE_TORCH"]            = "1"

from pathlib import Path
from typing import List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Model options (all free, local)
# ─────────────────────────────────────────────────────────────────────────────
#
# RERANKER_MODEL options (trade-off: quality vs speed):
#   "cross-encoder/ms-marco-MiniLM-L-6-v2"   ← fast, good quality  (~70MB)
#   "cross-encoder/ms-marco-MiniLM-L-12-v2"  ← slower, better      (~130MB)
#   "BAAI/bge-reranker-base"                  ← excellent for code  (~280MB)
#
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Re-ranks a list of retrieved chunks using a cross-encoder model.

    The cross-encoder scores (query, chunk) pairs jointly, giving a
    much more accurate relevance score than bi-encoder cosine similarity.

    Usage:
        reranker = CrossEncoderReranker()
        chunks   = retriever.search(query, top_k=20)   # fetch more
        chunks   = reranker.rerank(query, chunks, top_n=5)  # keep best 5
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        print(f"[Reranker] 🔄 Loading cross-encoder: {model_name}")
        try:
            from sentence_transformers import CrossEncoder
            self.model      = CrossEncoder(model_name)
            self.model_name = model_name
            self._ready     = True
            print(f"[Reranker] ✅ Ready")
        except Exception as e:
            print(f"[Reranker] ⚠️  Failed to load: {e}")
            print(f"[Reranker]    Falling back to original scores.")
            self.model  = None
            self._ready = False

    def rerank(
        self,
        query:  str,
        chunks: list,    # List[RetrievedChunk]
        top_n:  int = 5,
    ) -> list:
        """
        Re-rank chunks by cross-encoder score and return top_n.

        Args:
            query:  The original query string.
            chunks: List of RetrievedChunk from retriever.search().
            top_n:  How many to return after re-ranking.

        Returns:
            List of RetrievedChunk sorted by cross-encoder score (best first).
            The .score field is updated to the cross-encoder score.
            Original retrieval score preserved in .semantic_score.
        """
        if not self._ready or not chunks:
            return chunks[:top_n]

        # Build (query, chunk_text) pairs
        pairs  = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)

        # Attach reranker scores and sort
        scored = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for chunk, ce_score in scored[:top_n]:
            chunk.score = round(float(ce_score), 4)
            result.append(chunk)

        return result

    @property
    def ready(self) -> bool:
        return self._ready


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from retrieval.retriever import CodebaseRetriever

    retriever = CodebaseRetriever()
    reranker  = CrossEncoderReranker()

    queries = [
        "how does httpx handle timeouts?",
        "AsyncClient authentication bearer token",
        "connection pool keep alive",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        # Fetch more candidates than needed
        candidates = retriever.search(query, top_k=15, mode="hybrid")
        reranked   = reranker.rerank(query, candidates, top_n=5)

        print(f"\nBefore rerank (top 5 by hybrid score):")
        for i, c in enumerate(candidates[:5], 1):
            print(f"  [{i}] {c.score:.4f}  {c.format_source()}")

        print(f"\nAfter rerank (top 5 by cross-encoder score):")
        for i, c in enumerate(reranked, 1):
            print(f"  [{i}] {c.score:.4f}  {c.format_source()}")