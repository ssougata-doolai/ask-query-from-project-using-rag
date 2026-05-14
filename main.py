# main.py — CLI entrypoint for the Codebase Q&A RAG system
#
# Commands:
#   python main.py index                          ← index with default backend (config.py)
#   python main.py index --backend faiss          ← index with FAISS
#   python main.py index --backend chromadb       ← index with ChromaDB
#   python main.py index --reset                  ← wipe and re-index
#   python main.py ask                            ← interactive Q&A (default backend)
#   python main.py ask "question"                 ← single question
#   python main.py ask "question" --backend faiss ← use specific backend
#   python main.py compare "question"             ← run same question on BOTH backends

import argparse
import sys
from config import VECTOR_BACKEND


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_index(reset: bool = False, backend: str = VECTOR_BACKEND) -> None:
    from ingestion.embedder import run_ingestion_pipeline
    print(f"\n🔍 Indexing  [backend={backend}  reset={reset}]\n")
    run_ingestion_pipeline(reset_db=reset, backend=backend)
    print(f"\n✅ Done! Run  python main.py ask  to start Q&A.\n")


def cmd_ask(question: str = None, backend: str = VECTOR_BACKEND, mode: str = "hybrid") -> None:
    from generation.qa_chain import CodebaseQAChain
    chain = CodebaseQAChain(backend=backend)

    if question:
        chain.ask(question, mode=mode).print()
        return

    # Interactive mode
    print(f"\n💬 Codebase Q&A  [backend={backend}]")
    print("   Type your question or 'quit' to exit.\n")
    print("   Tips:")
    print("   • 'How does AsyncClient.get() handle redirects?'")
    print("   • 'What does _auth.py do?'")
    print("   • 'How are timeouts configured?'\n")

    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nBye! 👋")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("\nBye! 👋")
            break
        chain.ask(q).print()
        print()


def cmd_compare(question: str) -> None:
    """
    Run the same question on BOTH backends side by side.
    Great for seeing the difference in retrieved chunks and answers.
    """
    from generation.qa_chain import CodebaseQAChain

    print(f"\n🔀 Comparing backends for: \"{question}\"\n")

    for backend in ["faiss", "chromadb"]:
        print(f"\n{'▓'*60}")
        print(f"  BACKEND: {backend.upper()}")
        print(f"{'▓'*60}")
        try:
            chain  = CodebaseQAChain(backend=backend)
            result = chain.ask(question, verbose=True)
            result.print()
        except FileNotFoundError as e:
            print(f"  ⚠️  Index not found for {backend}.")
            print(f"     Run:  python main.py index --backend {backend}")


def cmd_status() -> None:
    """Show what's indexed in each backend."""
    from pathlib import Path
    from config import VECTOR_DB_PATH, CHROMA_DB_PATH

    print("\n── Index Status ─────────────────────────────────────")

    # FAISS
    faiss_index = Path(VECTOR_DB_PATH) / "index.faiss"
    faiss_meta  = Path(VECTOR_DB_PATH) / "metadata.json"
    if faiss_index.exists():
        import json
        metas  = json.loads(faiss_meta.read_text())
        size   = faiss_index.stat().st_size // 1024
        print(f"  FAISS     ✅  {len(metas):>5} chunks  |  {size} KB  →  {VECTOR_DB_PATH}/")
    else:
        print(f"  FAISS     ❌  not indexed  (run: python main.py index --backend faiss)")

    # ChromaDB
    chroma_dir = Path(CHROMA_DB_PATH)
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        print(f"  ChromaDB  ✅  indexed      →  {CHROMA_DB_PATH}/")
    else:
        print(f"  ChromaDB  ❌  not indexed  (run: python main.py index --backend chromadb)")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Codebase Q&A — RAG over a Python repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py index                              # index with default backend
  python main.py index --backend faiss             # index with FAISS
  python main.py index --backend chromadb          # index with ChromaDB
  python main.py index --backend faiss --reset     # wipe and re-index
  python main.py ask                               # interactive Q&A
  python main.py ask "How does httpx handle timeouts?"
  python main.py ask "question" --backend faiss
  python main.py compare "How does httpx handle timeouts?"
  python main.py status                            # show index status
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── index ──────────────────────────────────────────────────────────────────
    p_index = subparsers.add_parser("index", help="Index the repo into the vector store")
    p_index.add_argument("--backend", default=VECTOR_BACKEND, choices=["faiss", "chromadb"],
                         help=f"Vector backend to use (default: {VECTOR_BACKEND})")
    p_index.add_argument("--reset", action="store_true",
                         help="Wipe existing index and rebuild from scratch")

    # ── ask ────────────────────────────────────────────────────────────────────
    p_ask = subparsers.add_parser("ask", help="Ask a question about the codebase")
    p_ask.add_argument("question", nargs="?", default=None,
                       help="Question to ask (omit for interactive mode)")
    p_ask.add_argument("--backend", default=VECTOR_BACKEND, choices=["faiss", "chromadb"],
                       help=f"Vector backend to use (default: {VECTOR_BACKEND})")
    p_ask.add_argument("--mode", default="hybrid", choices=["hybrid", "semantic", "bm25"],
                       help="Search mode: hybrid (default), semantic, or bm25")

    # ── compare ────────────────────────────────────────────────────────────────
    p_cmp = subparsers.add_parser("compare", help="Run same question on both backends")
    p_cmp.add_argument("question", help="Question to compare")

    # ── status ─────────────────────────────────────────────────────────────────
    subparsers.add_parser("status", help="Show what's indexed in each backend")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(reset=args.reset, backend=args.backend)
    elif args.command == "ask":
        cmd_ask(question=args.question, backend=args.backend, mode=args.mode)
    elif args.command == "compare":
        cmd_compare(question=args.question)
    elif args.command == "status":
        cmd_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()