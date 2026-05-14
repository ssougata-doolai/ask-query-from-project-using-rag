# retrieval/bm25.py
# BM25 keyword search over indexed chunks.
# BM25 is the industry standard for keyword/term-frequency search —
# it's what Elasticsearch uses under the hood.
#
# Why BM25 + semantic = hybrid search?
#   Semantic: "how does httpx manage delays" → finds timeout-related chunks
#   BM25:     "AsyncClient.send"             → finds exact method name
#   Hybrid:   both queries benefit from both approaches

import json
import math
import re
import string
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import VECTOR_DB_PATH, CHROMA_DB_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Text tokenizer
# ─────────────────────────────────────────────────────────────────────────────

# Common English stop words (skip these in keyword matching)
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "do","does","did","will","would","could","should","may","might","can",
    "this","that","these","those","i","we","you","he","she","it","they",
    "what","which","who","how","when","where","why","not","no","if","as",
}

def tokenize(text: str) -> List[str]:
    """
    Lowercase, remove punctuation, split into tokens, remove stop words.
    Keeps camelCase and snake_case as-is (important for code).
    Also splits camelCase into sub-tokens: AsyncClient → [asyncclient, async, client]
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation except underscores (keep snake_case)
    text = re.sub(r"[^\w\s_]", " ", text)

    # Split on whitespace
    tokens = text.split()

    # Remove stop words and very short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    # Also split snake_case: async_client → [async_client, async, client]
    expanded = []
    for t in tokens:
        expanded.append(t)
        if "_" in t:
            parts = [p for p in t.split("_") if len(p) > 1 and p not in STOP_WORDS]
            expanded.extend(parts)

    return expanded


# ─────────────────────────────────────────────────────────────────────────────
# BM25 implementation
# ─────────────────────────────────────────────────────────────────────────────

class BM25:
    """
    BM25 Okapi — the gold standard for sparse keyword retrieval.

    Params:
        k1 = 1.5   term frequency saturation (higher = more weight to repeated terms)
        b  = 0.75  document length normalization (1.0 = full normalization)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1     = k1
        self.b      = b
        self.corpus: List[List[str]] = []    # tokenized documents
        self.docs:   List[str]       = []    # raw documents
        self.metas:  List[dict]      = []    # metadata per doc
        self.df:     Dict[str, int]  = {}    # document frequency per term
        self.idf:    Dict[str, float]= {}    # inverse document frequency
        self.avgdl:  float           = 0.0   # average document length
        self.N:      int             = 0     # total number of documents

    def fit(self, docs: List[str], metas: List[dict]) -> None:
        """Index all documents."""
        self.docs  = docs
        self.metas = metas
        self.N     = len(docs)

        # Tokenize all docs
        self.corpus = [tokenize(d) for d in docs]

        # Average document length
        lengths    = [len(tokens) for tokens in self.corpus]
        self.avgdl = sum(lengths) / self.N if self.N > 0 else 1.0

        # Document frequency
        self.df = {}
        for tokens in self.corpus:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {
            term: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in self.df.items()
        }

        print(f"[BM25] Indexed {self.N} documents  |  vocab size: {len(self.df)}")

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """BM25 score for one document."""
        doc_tokens = self.corpus[doc_idx]
        dl         = len(doc_tokens)
        score      = 0.0

        # Term frequency map for this doc
        tf_map: Dict[str, int] = {}
        for t in doc_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        for term in query_tokens:
            if term not in self.idf:
                continue
            tf    = tf_map.get(term, 0)
            idf   = self.idf[term]
            numer = tf * (self.k1 + 1)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (numer / denom)

        return score

    def search(
        self,
        query:       str,
        top_k:       int = 10,
        language:    Optional[str] = None,
        path_prefix: Optional[str] = None,
    ) -> List[dict]:
        """
        Return top-K documents scored by BM25.
        Returns list of dicts with keys: text, bm25_score, **metadata
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i in range(self.N):
            meta = self.metas[i]
            # Apply filters
            if language    and meta.get("language")           != language:    continue
            if path_prefix and path_prefix not in meta.get("relative_path", ""): continue
            s = self._score(query_tokens, i)
            if s > 0:
                scores.append((i, s))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append({
                "text":       self.docs[idx],
                "bm25_score": round(score, 4),
                **self.metas[idx],
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# BM25 index — save/load from disk
# ─────────────────────────────────────────────────────────────────────────────

class BM25Index:
    """
    Wrapper that loads docs/metadata from the FAISS or ChromaDB store
    and builds a BM25 index on top. Saved separately as bm25_corpus.json.
    """

    def __init__(self, index_dir: Optional[str] = None):
        # Use whichever vector store exists
        from config import VECTOR_BACKEND
        if index_dir:
            self.corpus_path = Path(index_dir) / "bm25_corpus.json"
        elif VECTOR_BACKEND == "faiss":
            self.corpus_path = Path(VECTOR_DB_PATH) / "bm25_corpus.json"
        else:
            self.corpus_path = Path(CHROMA_DB_PATH) / "bm25_corpus.json"

        self.bm25 = BM25()

    def build_from_store(self, docs: List[str], metas: List[dict]) -> None:
        """Build BM25 index from already-loaded docs and metadata."""
        self.bm25.fit(docs, metas)
        # Save corpus for fast reload
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"docs": docs, "metas": metas}
        self.corpus_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[BM25] 💾 Saved corpus to {self.corpus_path}")

    def load(self) -> None:
        """Load BM25 index from saved corpus file."""
        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"BM25 corpus not found at {self.corpus_path}\n"
                "Run: python main.py index  to rebuild."
            )
        payload = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        self.bm25.fit(payload["docs"], payload["metas"])
        print(f"[BM25] 📂 Loaded corpus from {self.corpus_path}")

    def search(self, query: str, top_k: int = 10,
               language: Optional[str] = None,
               path_prefix: Optional[str] = None) -> List[dict]:
        return self.bm25.search(query, top_k=top_k,
                                language=language, path_prefix=path_prefix)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    idx = BM25Index()
    idx.load()

    queries = ["AsyncClient timeout", "authentication bearer token", "redirect follow"]
    for q in queries:
        print(f"\nQuery: {q}")
        results = idx.search(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] bm25={r['bm25_score']:.3f}  {r['relative_path']} → {r.get('node_name','')}")