# evaluation/inspect_index.py
# Run this ONCE after indexing to see what files are actually indexed.
# Use the output to validate / fix expected_files in test_questions.json
#
# Usage:
#   python evaluation/inspect_index.py
#   python evaluation/inspect_index.py --search "_auth"
#   python evaluation/inspect_index.py --validate

import argparse
import json
from pathlib import Path
from collections import Counter

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHROMA_DB_PATH, VECTOR_DB_PATH, VECTOR_BACKEND


def load_corpus() -> list:
    """Load metadata from whichever backend is active."""
    # Try ChromaDB path first
    chroma_corpus = Path(CHROMA_DB_PATH) / "bm25_corpus.json"
    faiss_corpus  = Path(VECTOR_DB_PATH) / "bm25_corpus.json"

    for path in [chroma_corpus, faiss_corpus]:
        if path.exists():
            print(f"[Inspect] Loading from {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload["metas"]

    raise FileNotFoundError(
        "No bm25_corpus.json found. Run: python main.py index"
    )


def list_files(metas: list, language: str = "python") -> list:
    """Return sorted unique file paths for a given language."""
    paths = sorted(set(
        m["relative_path"] for m in metas
        if not language or m.get("language") == language
    ))
    return paths


def search_files(metas: list, keyword: str) -> list:
    """Find files whose path contains the keyword."""
    return sorted(set(
        m["relative_path"] for m in metas
        if keyword.lower() in m["relative_path"].lower()
    ))


def validate_questions(metas: list, questions_path: str) -> None:
    """Check each test question's expected_files against the actual index."""
    questions = json.loads(Path(questions_path).read_text())
    all_paths = set(m["relative_path"] for m in metas)

    print(f"\n── Validating {len(questions)} test questions ───────────────────")
    print(f"{'ID':<6} {'Status':<8} {'Expected files'}")
    print("─" * 60)

    issues = []
    for q in questions:
        expected = q.get("expected_files", [])
        found    = []
        missing  = []

        for exp in expected:
            matched = [p for p in all_paths if exp.lower() in p.lower()]
            if matched:
                found.append(exp)
            else:
                missing.append(exp)

        if missing:
            status = "⚠️  WARN"
            issues.append((q["id"], q["question"][:50], missing))
        else:
            status = "✅ OK  "

        print(f"{q['id']:<6} {status}  {', '.join(expected)}")
        if missing:
            print(f"             ❌ Not found in index: {missing}")

    if issues:
        print(f"\n⚠️  {len(issues)} questions have unmatched expected_files.")
        print("   Suggestion: run  python evaluation/inspect_index.py --search <term>")
        print("   to find the correct filename, then update test_questions.json")
    else:
        print(f"\n✅ All expected files matched in index.")


def main():
    parser = argparse.ArgumentParser(description="Inspect indexed files")
    parser.add_argument("--search",   default=None, help="Search for files containing this string")
    parser.add_argument("--lang",     default="python", help="Filter by language (default: python)")
    parser.add_argument("--all",      action="store_true", help="List ALL indexed files")
    parser.add_argument("--validate", action="store_true", help="Validate test_questions.json")
    parser.add_argument("--stats",    action="store_true", help="Show index statistics")
    args = parser.parse_args()

    metas = load_corpus()

    if args.stats or not any([args.search, args.all, args.validate]):
        # Default: show stats
        print(f"\n── Index Statistics ──────────────────────────────────────")
        print(f"  Total chunks   : {len(metas)}")
        lang_counts = Counter(m.get("language","?") for m in metas)
        for lang, count in lang_counts.most_common():
            print(f"  {lang:<14} : {count} chunks")
        node_counts = Counter(m.get("node_type","?") for m in metas)
        print(f"\n  Chunk types:")
        for ntype, count in node_counts.most_common():
            print(f"  {ntype:<20} : {count}")

        py_files = list_files(metas, language="python")
        print(f"\n  Python files   : {len(py_files)}")
        print(f"\n  Sample Python files:")
        for p in py_files[:20]:
            print(f"    {p}")
        if len(py_files) > 20:
            print(f"    ... and {len(py_files)-20} more")

    if args.search:
        results = search_files(metas, args.search)
        print(f"\n── Files matching '{args.search}' ──────────────────────────")
        if results:
            for p in results:
                print(f"  {p}")
        else:
            print(f"  No files found matching '{args.search}'")

    if args.all:
        files = list_files(metas, language=args.lang if args.lang != "all" else None)
        print(f"\n── All {args.lang} files ({len(files)}) ──────────────────────────")
        for p in files:
            print(f"  {p}")

    if args.validate:
        q_path = str(Path(__file__).parent / "test_questions.json")
        validate_questions(metas, q_path)


if __name__ == "__main__":
    main()