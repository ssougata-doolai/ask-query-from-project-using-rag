# 🔍 Codebase Q&A — RAG over a Python Repository

A production-quality **Retrieval-Augmented Generation (RAG)** system that lets you ask natural language questions about any Python codebase and get accurate, cited answers.

Built with **hybrid search** (semantic + BM25), **cross-encoder re-ranking**, a **Streamlit UI**, and a full **evaluation framework** using both custom metrics and RAGAS.

---

## ✨ Features

- **AST-aware chunking** — Python files split by function/class boundaries, not arbitrary token counts
- **Dual vector backend** — ChromaDB (feature-rich) or FAISS (lightweight), switchable via config
- **Hybrid search** — semantic + BM25 keyword search fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder re-ranking** — re-scores top-20 candidates for higher precision
- **Free LLM** — Groq (Llama 3.3 70B), no paid API needed
- **Streamlit UI** — dark-themed chat interface with source citations
- **Full evaluation** — custom metrics (Precision/Recall/F1) + RAGAS (Faithfulness/Relevancy)
- **CLI** — index, ask, compare backends, check status

---

## 📊 Evaluation Results (httpx codebase, 1467 chunks)

| Mode | Avg F1 | Keyword Coverage | Avg Latency |
|---|---|---|---|
| **Hybrid + Rerank** | **0.392** | **0.772** | 2.55s |
| Hybrid (no rerank) | 0.289 | 0.643 | 1.43s |
| Semantic only | 0.170 | 0.747 | 6.46s |
| BM25 only | 0.122 | 0.722 | 7.04s |

> Hybrid + Reranker is **2.3x better F1** than semantic alone, at only +1.1s latency cost.

---

## 🏗️ Architecture

```
GitHub Repo
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  INGESTION PIPELINE                                  │
│                                                      │
│  loader.py  →  chunker.py  →  embedder.py           │
│  (walk repo)   (AST split)    (embed + store)        │
│                               ├── ChromaDB           │
│                               ├── FAISS              │
│                               └── BM25 index         │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  RETRIEVAL PIPELINE                                  │
│                                                      │
│  Query → [Semantic Search]  ─┐                       │
│          [BM25 Search]      ─┤→ RRF Fusion           │
│                               └→ Cross-Encoder       │
│                                  Re-ranking          │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  GENERATION                                          │
│                                                      │
│  Context + Question → Groq (Llama 3.3 70B)          │
│                     → Answer + File Citations        │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
codebase-qa/
├── app.py                        # Streamlit web UI
├── main.py                       # CLI entrypoint
├── config.py                     # All settings (models, paths, weights)
├── requirements.txt
├── .env.example                  # API key template
├── .gitignore
│
├── ingestion/
│   ├── loader.py                 # Walk repo, load files with metadata
│   ├── chunker.py                # AST-aware chunker (Python) + text fallback
│   └── embedder.py               # Dual-backend embedder (FAISS + ChromaDB)
│
├── retrieval/
│   ├── retriever.py              # Hybrid search with RRF fusion
│   ├── bm25.py                   # Pure Python BM25 implementation
│   └── reranker.py               # Cross-encoder re-ranking
│
├── generation/
│   └── qa_chain.py               # RAG chain: retrieve → rerank → generate
│
├── evaluation/
│   ├── test_questions.json       # 12 ground-truth Q&A pairs
│   ├── metrics.py                # Custom metrics (Precision/Recall/F1/Coverage)
│   ├── evaluator.py              # Custom evaluation runner
│   ├── ragas_eval.py             # RAGAS evaluation (LLM-as-judge)
│   ├── full_eval.py              # Combined evaluation (custom + RAGAS)
│   └── inspect_index.py          # Index inspector + question validator
│
└── notebooks/
    ├── step1_load_and_chunk.ipynb
    └── step2_embed_store_ask.ipynb
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/codebase-qa.git
cd codebase-qa

# Create a Python 3.10 environment (required — 3.8 has compatibility issues)
conda create -n codebase_qa python=3.10 -y
conda activate codebase_qa

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your free Groq API key:
# GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Clone a repo to index

```bash
# We use httpx as the example — replace with any Python repo
git clone https://github.com/encode/httpx.git
```

### 4. Index the codebase

```bash
# Index with ChromaDB (recommended)
python main.py index --backend chromadb

# Or with FAISS
python main.py index --backend faiss
```

This runs the full pipeline:
- Loads and filters all supported files (`.py`, `.md`, `.yaml`, `.txt`)
- Splits Python files by AST (function/class boundaries)
- Embeds all chunks with `all-MiniLM-L6-v2` (local, free, ~90MB download)
- Stores in ChromaDB/FAISS + builds BM25 keyword index

### 5. Ask questions

```bash
# Interactive mode
python main.py ask

# Single question
python main.py ask "How does httpx handle timeouts?"

# Specify search mode
python main.py ask "AsyncClient.send method" --mode bm25

# Compare both backends on same question
python main.py compare "How does httpx handle redirects?"

# Check index status
python main.py status
```

### 6. Launch the UI

```bash
streamlit run app.py

# On a remote server (SSH tunnel recommended)
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then open `http://localhost:8501` in your browser.

---

## ⚙️ Configuration

All settings are in `config.py`:

```python
# Switch vector backend
VECTOR_BACKEND  = "chromadb"   # "chromadb" | "faiss"

# Switch LLM
LLM_BACKEND     = "groq"       # "groq" | "ollama"
LLM_MODEL       = "llama-3.3-70b-versatile"

# Search settings
SEARCH_MODE     = "hybrid"     # "hybrid" | "semantic" | "bm25"
SEMANTIC_WEIGHT = 0.7          # weight in RRF fusion
BM25_WEIGHT     = 0.3

# Re-ranking
USE_RERANKER    = True
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval
TOP_K           = 5            # chunks returned to LLM
RETRIEVAL_K     = 20           # candidates before reranking
```

### Using Ollama (fully local, no API key)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
ollama serve

# Update config.py
LLM_BACKEND = "ollama"
OLLAMA_MODEL = "llama3"
```

---

## 🔍 Search Modes Explained

| Mode | How it works | Best for |
|---|---|---|
| `hybrid` | Semantic + BM25 fused via RRF | General use — best quality |
| `semantic` | Dense vector cosine similarity | Conceptual questions |
| `bm25` | Keyword term-frequency ranking | Exact function/class names |

```bash
# hybrid is the default and recommended
python main.py ask "how does authentication work?"

# bm25 is great for exact names
python main.py ask "AsyncClient.send" --mode bm25

# semantic for conceptual queries
python main.py ask "connection retry mechanism" --mode semantic
```

---

## 📊 Evaluation

### Run custom metrics (fast)

```bash
# Evaluate hybrid mode
python evaluation/evaluator.py --mode hybrid

# Compare all three modes side by side
python evaluation/evaluator.py --compare

# Compare with/without reranker
python evaluation/evaluator.py --mode hybrid --no-rerank
```

**Custom metrics:**
- **Retrieval Precision** — relevant chunks / total retrieved
- **Retrieval Recall** — expected files found / total expected
- **Retrieval F1** — harmonic mean of precision and recall
- **Keyword Coverage** — expected terms found in answer
- **File Citation Rate** — answers that mention specific files
- **Latency** — end-to-end response time

### Run RAGAS evaluation (LLM-as-judge)

```bash
# Quick test on 3 questions (~30s)
python evaluation/ragas_eval.py --questions 3

# Full RAGAS evaluation
python evaluation/ragas_eval.py
```

**RAGAS metrics:**
- **Faithfulness** — are all claims in the answer grounded in retrieved context? (detects hallucination)
- **Answer Relevancy** — does the answer actually address the question?
- **Context Precision** — were the retrieved chunks useful?
- **Context Recall** — did retrieved chunks cover all necessary information?

### Run full combined evaluation

```bash
python evaluation/full_eval.py
python evaluation/full_eval.py --questions 5   # quick test
```

### Inspect the index

```bash
# Show index statistics
python evaluation/inspect_index.py

# Search for specific files
python evaluation/inspect_index.py --search "_auth"
python evaluation/inspect_index.py --search "transport"

# Validate test questions against actual index
python evaluation/inspect_index.py --validate
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| **Vector DB** | ChromaDB or FAISS |
| **Keyword Search** | Custom BM25 (pure Python) |
| **Fusion** | Reciprocal Rank Fusion (RRF) |
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | Llama 3.3 70B via Groq (free tier) |
| **UI** | Streamlit |
| **Evaluation** | Custom metrics + RAGAS |

---

## 🔧 CLI Reference

```bash
# Indexing
python main.py index                          # index with default backend
python main.py index --backend faiss          # index with FAISS
python main.py index --backend chromadb       # index with ChromaDB
python main.py index --reset                  # wipe and re-index

# Asking
python main.py ask                            # interactive Q&A
python main.py ask "your question"            # single question
python main.py ask "question" --mode hybrid   # specify search mode
python main.py ask "question" --backend faiss # specify backend

# Utilities
python main.py compare "question"             # compare both backends
python main.py status                         # show what's indexed
```

---

## 📝 Extending to Other Repos

To index a different Python repository:

1. Clone the repo into the project folder
2. Update `config.py`:
   ```python
   REPO_PATH = "./your-repo-name"
   REPO_NAME = "your-repo-name"
   ```
3. Re-index:
   ```bash
   python main.py index --reset
   ```

Works with any Python codebase — Django, FastAPI, NumPy, your own projects, etc.

---

## 🐛 Troubleshooting

**SQLite error with ChromaDB on Python 3.8:**
```bash
pip install pysqlite3-binary
# or upgrade to Python 3.10 (recommended)
```

**`numpy.typeDict` error:**
```bash
pip install "numpy==1.26.4" "transformers==4.41.0" "sentence-transformers==3.0.0"
```

**Groq model decommissioned error:**
Update `LLM_MODEL` in `config.py`. Check current models at [console.groq.com/docs](https://console.groq.com/docs/deprecations).

**BM25 index not found:**
```bash
# Re-run indexing — BM25 is built alongside the vector index
python main.py index --backend chromadb
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [httpx](https://github.com/encode/httpx) — the example codebase used for testing
- [RAGAS](https://github.com/explodinggradients/ragas) — RAG evaluation framework
- [Groq](https://groq.com) — free LLM inference API
- [ChromaDB](https://trychroma.com) — vector database
- [FAISS](https://github.com/facebookresearch/faiss) — Meta's vector similarity library
- [sentence-transformers](https://www.sbert.net) — local embedding models
