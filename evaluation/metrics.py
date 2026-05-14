# evaluation/metrics.py
# RAG evaluation metrics — no external dependencies, pure Python.
#
# Metrics computed:
#   1. Retrieval Precision    — did we retrieve chunks from the expected files?
#   2. Keyword Coverage       — does the answer contain expected keywords?
#   3. Answer Faithfulness    — does the answer stay grounded in sources?
#   4. Source Citation Rate   — does the answer mention file/function names?
#   5. Latency                — end-to-end response time

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


@dataclass
class EvalResult:
    """Evaluation result for a single question."""
    question_id:        str
    question:           str
    answer:             str
    latency_sec:        float

    # Retrieval metrics
    retrieved_files:    List[str]  = field(default_factory=list)
    expected_files:     List[str]  = field(default_factory=list)
    retrieval_precision: float     = 0.0   # fraction of retrieved that are relevant
    retrieval_recall:    float     = 0.0   # fraction of expected that were retrieved

    # Answer quality metrics
    keyword_coverage:   float      = 0.0   # fraction of expected keywords found in answer
    has_file_citation:  bool       = False  # does answer mention a file/function?
    answer_length:      int        = 0      # word count
    num_sources:        int        = 0      # chunks used

    # Search mode used
    search_mode:        str        = "hybrid"

    @property
    def f1_retrieval(self) -> float:
        """Harmonic mean of precision and recall."""
        if self.retrieval_precision + self.retrieval_recall == 0:
            return 0.0
        return 2 * (self.retrieval_precision * self.retrieval_recall) / \
               (self.retrieval_precision + self.retrieval_recall)

    def to_dict(self) -> dict:
        return {
            "id":                   self.question_id,
            "question":             self.question[:80],
            "latency_sec":          round(self.latency_sec, 2),
            "retrieval_precision":  round(self.retrieval_precision, 3),
            "retrieval_recall":     round(self.retrieval_recall, 3),
            "f1_retrieval":         round(self.f1_retrieval, 3),
            "keyword_coverage":     round(self.keyword_coverage, 3),
            "has_file_citation":    self.has_file_citation,
            "answer_length_words":  self.answer_length,
            "num_sources":          self.num_sources,
            "search_mode":          self.search_mode,
        }


def compute_retrieval_metrics(
    retrieved_paths: List[str],
    expected_files:  List[str],
) -> tuple:
    """
    Compute precision and recall for retrieval.

    Precision = relevant retrieved / total retrieved
    Recall    = relevant retrieved / total expected

    A retrieved chunk is "relevant" if its path contains any expected file substring.
    """
    if not retrieved_paths or not expected_files:
        return 0.0, 0.0

    def is_relevant(path: str) -> bool:
        return any(exp.lower() in path.lower() for exp in expected_files)

    relevant_retrieved = sum(1 for p in retrieved_paths if is_relevant(p))
    precision = relevant_retrieved / len(retrieved_paths)
    recall    = min(relevant_retrieved / len(expected_files), 1.0)

    return precision, recall


def compute_keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def has_file_citation(answer: str) -> bool:
    """
    Check if the answer mentions a file path or function name.
    Looks for patterns like `file.py`, `module/file.py`, or backtick-wrapped identifiers.
    """
    patterns = [
        r"`[a-zA-Z_][a-zA-Z0-9_]*\.py`",        # `_client.py`
        r"\b[a-zA-Z_][a-zA-Z0-9_]*/[a-zA-Z_]",  # httpx/_client
        r"`[a-zA-Z_][a-zA-Z0-9_]+\.[a-zA-Z_]",  # `AsyncClient.send`
        r"In `[a-zA-Z_]",                         # In `httpx/...`
        r"in `[a-zA-Z_]",
        r"file [a-zA-Z_][a-zA-Z0-9_/]*\.py",
    ]
    return any(re.search(p, answer) for p in patterns)