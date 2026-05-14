# config.py — Central config for the entire project

import os
from dotenv import load_dotenv
load_dotenv()   # loads .env file automatically if present

# ── Repo ──────────────────────────────────────────────────────────────────────
REPO_PATH = "./repo"          # path to the cloned repo
REPO_NAME = "httpx"

# ── File types to index ───────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".py":   "python",
    ".md":   "markdown",
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".txt":  "text",
    ".rst":  "text",
}

# Folders to skip entirely
EXCLUDED_DIRS = {
    ".git", "__pycache__", ".pytest_cache", "node_modules",
    "dist", "build", ".eggs", "*.egg-info", ".venv", "venv", "env",
}

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 1000   # characters (fallback splitter)
CHUNK_OVERLAP     = 150    # characters overlap between chunks

# ── Embedding ─────────────────────────────────────────────────────────────────
# Free local embeddings — no API key needed
# Other option: "all-mpnet-base-v2" (slower but higher quality)
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
EMBEDDING_BACKEND = "local"               # "local" only for now

# ── Vector Store ──────────────────────────────────────────────────────────────
# Switch between backends by changing VECTOR_BACKEND:
#   "faiss"    → FAISS (Meta) — pure vector index, works on any Python version
#   "chromadb" → ChromaDB     — full-featured vector DB, requires Python 3.9+
#
VECTOR_BACKEND    = "chromadb"          # ← change to "faiss" to use FAISS

# Shared storage path (used by both backends)
VECTOR_DB_PATH    = "./vector_db"       # FAISS saves files here
CHROMA_DB_PATH    = "./chroma_db"       # ChromaDB saves its DB here
COLLECTION_NAME   = "codebase_qa"      # ChromaDB collection name

# ── LLM — Groq (free tier, fast) ─────────────────────────────────────────────
# Sign up at https://console.groq.com → copy your free API key → paste below
# OR set env variable:  export GROQ_API_KEY="gsk_..."
#
# Available free models on Groq (May 2026):
#   "llama-3.3-70b-versatile"   ← best quality, recommended ✅
#   "llama-3.1-8b-instant"      ← fastest, lowest quality
#   "llama-4-scout-17b-16e-instruct" ← Llama 4, good balance
#   "qwen-qwq-32b"              ← strong reasoning model
#
LLM_BACKEND       = "groq"                    # "groq" | "ollama" | "anthropic"
LLM_MODEL         = "llama-3.3-70b-versatile"  # Llama 3.3 70B via Groq (free, replaces llama3-70b-8192)
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")   # set via env var

# ── Ollama (alternative — fully local, zero API calls) ────────────────────────
# If you prefer Ollama: set LLM_BACKEND = "ollama" and run:
#   ollama pull llama3
# then make sure Ollama is running: ollama serve
OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_MODEL      = "llama3"

# ── Generation ────────────────────────────────────────────────────────────────
MAX_TOKENS        = 1024
TEMPERATURE       = 0.1    # low = more factual, deterministic answers

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K             = 5      # final number of chunks returned to LLM
RETRIEVAL_K       = 20     # how many to fetch before reranking (should be > TOP_K)
SEARCH_MODE       = "hybrid"   # "hybrid" | "semantic" | "bm25"
SEMANTIC_WEIGHT   = 0.7        # weight for semantic scores in RRF fusion
BM25_WEIGHT       = 0.3        # weight for BM25 scores in RRF fusion

# ── Re-ranking ────────────────────────────────────────────────────────────────
USE_RERANKER      = True       # set False to skip reranking (faster but lower quality)
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~70MB, fast