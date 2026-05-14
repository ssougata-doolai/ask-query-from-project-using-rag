# generation/qa_chain.py
# RAG chain: retrieve context → build prompt → call Groq (Llama 3) → return answer + citations.

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    LLM_MODEL, GROQ_API_KEY, MAX_TOKENS, TEMPERATURE,
    TOP_K, RETRIEVAL_K, LLM_BACKEND, OLLAMA_BASE_URL, OLLAMA_MODEL,
    VECTOR_BACKEND, SEARCH_MODE, SEMANTIC_WEIGHT, BM25_WEIGHT,
    USE_RERANKER, RERANKER_MODEL,
)
from retrieval.retriever import CodebaseRetriever, RetrievedChunk


# ─────────────────────────────────────────────────────────────────────────────
# Answer dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QAResult:
    question:   str
    answer:     str
    sources:    List[RetrievedChunk]
    model_used: str

    def print(self) -> None:
        print(f"\n{'='*60}")
        print(f"❓ {self.question}")
        print(f"{'='*60}")
        print(f"\n{self.answer}")
        if self.sources:
            print(f"\n── Sources ──────────────────────────────────────────────")
            for i, s in enumerate(self.sources, 1):
                print(f"  [{i}] {s.format_source()}  (score: {s.score:.3f})")
        print(f"\n[{self.model_used}  |  vector_backend={VECTOR_BACKEND}]")


# ─────────────────────────────────────────────────────────────────────────────
# System prompt + prompt builder
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert code assistant helping developers understand a Python codebase.

You are given relevant code excerpts retrieved from the codebase, followed by a developer's question.
Each excerpt is labelled with [Source N] showing the file path and function name.

Rules:
1. Answer using ONLY the provided code context — do not hallucinate code or behaviour.
2. ALWAYS mention the specific file and function/class where relevant code lives.
   Example: "In `httpx/_client.py`, the `AsyncClient.send()` method handles..."
3. Quote or reference exact code parts when it helps clarity.
4. If the context doesn't contain enough information, say so clearly.
5. Be concise but thorough. Use inline backticks for code identifiers."""


def _build_prompt(question: str, context: str) -> str:
    return f"""Here are relevant code excerpts from the codebase:

{context}

---

Developer question: {question}

Answer:"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM backends
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(messages: list) -> str:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set.\n"
            "1. Sign up at https://console.groq.com\n"
            "2. Create a free API key\n"
            "3. Add GROQ_API_KEY=gsk_... to your .env file"
        )
    from groq import Groq
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model       = LLM_MODEL,
        messages    = messages,
        max_tokens  = MAX_TOKENS,
        temperature = TEMPERATURE,
    )
    return response.choices[0].message.content


def _call_ollama(messages: list) -> str:
    import requests as req
    response = req.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
              "options": {"temperature": TEMPERATURE}},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _call_llm(messages: list) -> str:
    if LLM_BACKEND == "groq":
        return _call_groq(messages)
    elif LLM_BACKEND == "ollama":
        return _call_ollama(messages)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND!r}. Use 'groq' or 'ollama'.")


# ─────────────────────────────────────────────────────────────────────────────
# QA Chain
# ─────────────────────────────────────────────────────────────────────────────

class CodebaseQAChain:
    """
    Full RAG pipeline: retrieve → prompt → generate → return QAResult.

    Usage:
        chain  = CodebaseQAChain()
        result = chain.ask("How does httpx handle redirects?")
        result.print()
    """

    def __init__(self, top_k: int = TOP_K, backend: str = VECTOR_BACKEND,
                 search_mode: str = SEARCH_MODE, use_reranker: bool = USE_RERANKER):
        self.retriever   = CodebaseRetriever(
            backend         = backend,
            semantic_weight = SEMANTIC_WEIGHT,
            bm25_weight     = BM25_WEIGHT,
        )
        self.top_k        = top_k
        self.search_mode  = search_mode
        self.use_reranker = use_reranker

        # Load reranker (lazy — only if enabled)
        self.reranker = None
        if use_reranker:
            from retrieval.reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(model_name=RERANKER_MODEL)
            if not self.reranker.ready:
                self.reranker = None

        llm_label = f"groq/{LLM_MODEL}" if LLM_BACKEND == "groq" else f"ollama/{OLLAMA_MODEL}"
        rerank_label = f"✓ {RERANKER_MODEL.split('/')[-1]}" if (use_reranker and self.reranker) else "✗"
        print(f"[QAChain]  🤖 LLM: {llm_label}  |  search={search_mode}  |  rerank={rerank_label}")

    def ask(
        self,
        question:    str,
        top_k:       Optional[int] = None,
        mode:        Optional[str] = None,
        language:    Optional[str] = None,
        path_prefix: Optional[str] = None,
        min_score:   float = 0.15,
        verbose:     bool = False,
    ) -> QAResult:
        """
        Ask a question about the indexed codebase.

        Args:
            question:    Natural language question.
            top_k:       How many chunks to retrieve (overrides config default).
            mode:        "hybrid" | "semantic" | "bm25" — overrides config default.
            language:    Only retrieve from this language ("python", "markdown" …).
            path_prefix: Only retrieve from files whose path contains this string.
            min_score:   Minimum similarity score — filters out weak matches.
            verbose:     Print retrieved chunks before generating the answer.
        """
        k           = top_k or self.top_k
        search_mode = mode  or self.search_mode

        # ── 1. Retrieve (fetch more if reranking) ──────────────────────────────
        fetch_k = RETRIEVAL_K if self.reranker else k
        chunks  = self.retriever.search(
            query       = question,
            top_k       = fetch_k,
            mode        = search_mode,
            language    = language,
            path_prefix = path_prefix,
            min_score   = min_score,
        )

        if not chunks:
            return QAResult(
                question   = question,
                answer     = "⚠️  No relevant code found. Try rephrasing or lowering min_score.",
                sources    = [],
                model_used = f"{LLM_BACKEND}/{LLM_MODEL}",
            )

        # ── 2. Re-rank ─────────────────────────────────────────────────────────
        if self.reranker and len(chunks) > 1:
            if verbose:
                print(f"\n[QAChain] Re-ranking {len(chunks)} candidates → top {k}...")
            chunks = self.reranker.rerank(question, chunks, top_n=k)

        if verbose:
            print(f"\n[QAChain] Final {len(chunks)} chunks:")
            for i, c in enumerate(chunks, 1):
                print(f"  [{i}] score={c.score:.4f}  {c.format_source()}")

        # ── 2. Build prompt ────────────────────────────────────────────────────
        context      = self.retriever.format_context(chunks)
        user_message = _build_prompt(question, context)
        messages     = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        # ── 3. Generate ────────────────────────────────────────────────────────
        answer = _call_llm(messages)

        return QAResult(
            question   = question,
            answer     = answer,
            sources    = chunks,
            model_used = f"{LLM_BACKEND}/{LLM_MODEL}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chain = CodebaseQAChain()
    questions = [
        "How does httpx handle timeouts?",
        "What authentication methods does httpx support?",
        "How does AsyncClient differ from the regular Client?",
    ]
    for q in questions:
        chain.ask(q, verbose=True).print()
        print()