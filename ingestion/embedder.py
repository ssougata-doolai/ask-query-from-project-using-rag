# ingestion/embedder.py
import os
os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"]               = "0"
os.environ["USE_TORCH"]            = "1"

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    EMBEDDING_MODEL, VECTOR_BACKEND,
    VECTOR_DB_PATH, CHROMA_DB_PATH, COLLECTION_NAME,
)
from ingestion.chunker import Chunk


# ─────────────────────────────────────────────────────────────────────────────
# Common interface — both backends implement these methods
# ─────────────────────────────────────────────────────────────────────────────

class BaseVectorStore:
    def save(self): raise NotImplementedError
    def load(self): raise NotImplementedError
    def search(self, query_vector, top_k, language, path_prefix, min_score) -> List[dict]: raise NotImplementedError
    @property
    def count(self) -> int: raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Backend A — FAISS
# ─────────────────────────────────────────────────────────────────────────────

class FAISSVectorStore(BaseVectorStore):
    """
    FAISS flat inner-product index.
    Saves: index.faiss + metadata.json + documents.json
    Works on Python 3.8+, no extra dependencies beyond faiss-cpu.
    """

    def __init__(self, index_dir: str = VECTOR_DB_PATH):
        try:
            import faiss as _faiss
            self._faiss = _faiss
        except ImportError:
            raise ImportError("Run:  pip install faiss-cpu")

        self.index_dir  = Path(index_dir)
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path  = self.index_dir / "metadata.json"
        self.docs_path  = self.index_dir / "documents.json"
        self._index     = None
        self._metas:    List[dict] = []
        self._docs:     List[str]  = []

    def build(self, embeddings: np.ndarray, metas: List[dict], docs: List[str]):
        dim          = embeddings.shape[1]
        self._index  = self._faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._metas  = metas
        self._docs   = docs
        print(f"[FAISS] Built index: {self._index.ntotal} vectors (dim={dim})")

    def save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self._metas, ensure_ascii=False), encoding="utf-8")
        self.docs_path.write_text(json.dumps(self._docs,  ensure_ascii=False), encoding="utf-8")
        print(f"[FAISS] 💾 Saved to {self.index_dir}/  ({self.index_path.stat().st_size//1024} KB)")

    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"No FAISS index at {self.index_path}\nRun: python main.py index")
        self._index = self._faiss.read_index(str(self.index_path))
        self._metas = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._docs  = json.loads(self.docs_path.read_text(encoding="utf-8"))
        print(f"[FAISS] 📂 Loaded {self._index.ntotal} vectors from {self.index_dir}/")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        language: Optional[str] = None,
        path_prefix: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[dict]:
        fetch_k = min(top_k * 10, self._index.ntotal)
        scores, indices = self._index.search(
            query_vector.reshape(1, -1).astype("float32"), fetch_k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < min_score:
                continue
            meta = self._metas[idx]
            if language    and meta.get("language")           != language:    continue
            if path_prefix and path_prefix not in meta.get("relative_path", ""): continue
            results.append({"text": self._docs[idx], "score": float(round(score, 4)), **meta})
            if len(results) >= top_k:
                break
        return results

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index else 0


# ─────────────────────────────────────────────────────────────────────────────
# Backend B — ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB persistent collection.
    Requires Python 3.9+ and chromadb>=0.5.0.
    Richer features: native metadata filtering, collections, built-in persistence.
    """

    def __init__(self, db_path: str = CHROMA_DB_PATH, collection_name: str = COLLECTION_NAME):
        try:
            import chromadb as _chromadb
            self._chromadb = _chromadb
        except ImportError:
            raise ImportError("Run:  pip install chromadb")

        self.db_path         = db_path
        self.collection_name = collection_name
        self._collection     = None
        self._client         = None

    def _connect(self, reset: bool = False):
        self._client = self._chromadb.PersistentClient(path=self.db_path)
        if reset:
            try:
                self._client.delete_collection(self.collection_name)
                print(f"[ChromaDB] 🗑  Deleted collection: {self.collection_name}")
            except Exception:
                pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[ChromaDB] 📦 Collection '{self.collection_name}'  ({self._collection.count()} docs)")

    def build(self, embeddings: np.ndarray, metas: List[dict], docs: List[str], reset: bool = True):
        """Build collection by upserting all chunks."""
        self._connect(reset=reset)

        # Upsert in batches of 500 (ChromaDB limit per call)
        batch_size = 500
        total = len(docs)
        for i in range(0, total, batch_size):
            batch_embs  = embeddings[i : i + batch_size]
            batch_metas = metas[i : i + batch_size]
            batch_docs  = docs[i : i + batch_size]
            ids         = [f"chunk_{j}" for j in range(i, min(i + batch_size, total))]

            self._collection.upsert(
                ids        = ids,
                embeddings = batch_embs.tolist(),
                documents  = batch_docs,
                metadatas  = batch_metas,
            )
        print(f"[ChromaDB] ✅ Stored {total} chunks  →  total: {self._collection.count()}")

    def save(self):
        # ChromaDB PersistentClient auto-saves; nothing to do
        print(f"[ChromaDB] 💾 Auto-persisted to {self.db_path}/")

    def load(self):
        import os
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"No ChromaDB at {self.db_path}\nRun: python main.py index")
        self._connect(reset=False)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        language: Optional[str] = None,
        path_prefix: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[dict]:
        # Build ChromaDB where filter
        where = None
        conditions = []
        if language:
            conditions.append({"language": {"$eq": language}})
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        results = self._collection.query(
            query_embeddings = [query_vector.tolist()],
            n_results        = top_k,
            where            = where,
            include          = ["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist    # cosine distance → similarity
            if score < min_score:
                continue
            # path_prefix filter (not natively supported in chroma for "contains")
            if path_prefix and path_prefix not in meta.get("relative_path", ""):
                continue
            output.append({"text": doc, "score": round(score, 4), **meta})
        return output

    @property
    def count(self) -> int:
        return self._collection.count() if self._collection else 0


# ─────────────────────────────────────────────────────────────────────────────
# Factory — returns the right store based on config
# ─────────────────────────────────────────────────────────────────────────────

def get_vector_store(backend: str = VECTOR_BACKEND) -> BaseVectorStore:
    """Return a VectorStore instance for the configured backend."""
    if backend == "faiss":
        print(f"[VectorStore] Backend: FAISS  (index dir: {VECTOR_DB_PATH})")
        return FAISSVectorStore(index_dir=VECTOR_DB_PATH)
    elif backend == "chromadb":
        print(f"[VectorStore] Backend: ChromaDB  (db path: {CHROMA_DB_PATH})")
        return ChromaVectorStore(db_path=CHROMA_DB_PATH, collection_name=COLLECTION_NAME)
    else:
        raise ValueError(f"Unknown VECTOR_BACKEND: {backend!r}. Use 'faiss' or 'chromadb'.")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    print(f"[Embedder] 🔄 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[Embedder] ✅ Model loaded  (dim={model.get_sentence_embedding_dimension()})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Embed + store
# ─────────────────────────────────────────────────────────────────────────────

def embed_and_store(
    chunks: List[Chunk],
    model: SentenceTransformer,
    backend: str = VECTOR_BACKEND,
    batch_size: int = 64,
) -> BaseVectorStore:
    """
    Embed all chunks and store in the configured vector backend.
    Returns the populated store (already saved to disk).
    """
    total    = len(chunks)
    all_embs = []
    metas    = []
    docs     = []
    t_start  = time.time()

    print(f"\n[Embedder] ⚙️  Embedding {total} chunks (batch_size={batch_size})...")

    for batch_num, i in enumerate(range(0, total, batch_size)):
        batch = chunks[i : i + batch_size]
        embs  = model.encode(
            [c.text for c in batch],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        all_embs.append(embs)
        for c in batch:
            metas.append(c.to_metadata_dict())
            docs.append(c.text)

        stored  = min(i + batch_size, total)
        elapsed = time.time() - t_start
        speed   = stored / elapsed if elapsed > 0 else 0
        print(f"  Batch {batch_num+1:>3} | {stored:>5}/{total} chunks | {speed:.0f} chunks/sec")

    all_embs_np = np.vstack(all_embs).astype("float32")

    # Build + save via the chosen backend
    store = get_vector_store(backend)

    if isinstance(store, FAISSVectorStore):
        store.build(all_embs_np, metas, docs)
        store.save()
    elif isinstance(store, ChromaVectorStore):
        store.build(all_embs_np, metas, docs, reset=True)
        store.save()

    print(f"\n[Embedder] ✅ Done!  {total} chunks  |  {time.time()-t_start:.1f}s  |  backend={backend}")
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Full ingestion pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion_pipeline(
    repo_path: Optional[str] = None,
    reset_db: bool = False,
    backend: str = VECTOR_BACKEND,
) -> BaseVectorStore:
    """Full pipeline: load → chunk → embed → store → build BM25 index."""
    from ingestion.loader  import load_repo, summarize_loaded_files
    from ingestion.chunker import chunk_all_files

    print("=" * 55)
    print(f"STEP 1 — Loading files")
    print("=" * 55)
    files = load_repo(repo_path) if repo_path else load_repo()
    summarize_loaded_files(files)

    print("=" * 55)
    print("STEP 2 — Chunking")
    print("=" * 55)
    chunks = chunk_all_files(files)

    print("=" * 55)
    print(f"STEP 3 — Embedding + Storing  [{backend.upper()}]")
    print("=" * 55)
    model = load_embedding_model()
    store = embed_and_store(chunks, model, backend=backend)

    print("=" * 55)
    print("STEP 4 — Building BM25 keyword index")
    print("=" * 55)
    from retrieval.bm25 import BM25Index
    bm25_idx = BM25Index()
    docs  = [c.text for c in chunks]
    metas = [c.to_metadata_dict() for c in chunks]
    bm25_idx.build_from_store(docs, metas)
    print("[BM25] ✅ Keyword index ready")

    return store


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    store = run_ingestion_pipeline()
    model = load_embedding_model()
    q_vec = model.encode("how does httpx handle timeouts?", normalize_embeddings=True).astype("float32")

    print("\n── Test query ───────────────────────────────────")
    for i, r in enumerate(store.search(q_vec, top_k=3), 1):
        print(f"\n[{i}] score={r['score']:.3f}  {r['relative_path']} → {r.get('node_name','')}")
        print(f"     {r['text'][:200].strip()}...")