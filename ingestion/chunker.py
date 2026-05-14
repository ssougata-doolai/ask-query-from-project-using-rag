# ingestion/chunker.py
# Splits raw files into meaningful chunks with rich metadata.

# Strategy:
#   Python files  → AST-aware split (per function / class)
#   Markdown      → Header-aware recursive split
#   YAML / Text   → Recursive character split (fallback)

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, REPO_NAME
from ingestion.loader import RawFile


@dataclass
class Chunk:
    """
    A single chunk of source text with all metadata needed for retrieval.
    This is what gets embedded and stored in the vector DB.
    """
    # ── Content ───────────────────────────────────────────────────────────────
    text: str                        # the actual text to embed

    # ── Source metadata ───────────────────────────────────────────────────────
    relative_path: str               # e.g. "httpx/_client.py"
    language: str                    # "python" | "markdown" | "yaml" | "text"
    chunk_index: int                 # position of this chunk within the file

    # ── Code-specific metadata (Python only) ──────────────────────────────────
    node_type: Optional[str] = None  # "function" | "class" | "module_level"
    node_name: Optional[str] = None  # e.g. "AsyncClient.get"
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    # ── Derived ───────────────────────────────────────────────────────────────
    repo_name: str = REPO_NAME

    def to_metadata_dict(self) -> dict:
        """Flat dict for storing in ChromaDB metadata field."""
        return {
            "relative_path": self.relative_path,
            "language":       self.language,
            "chunk_index":    self.chunk_index,
            "node_type":      self.node_type or "",
            "node_name":      self.node_name or "",
            "start_line":     self.start_line or 0,
            "end_line":       self.end_line or 0,
            "repo_name":      self.repo_name,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Python AST chunker
# ─────────────────────────────────────────────────────────────────────────────

def _get_source_segment(source_lines: List[str], start: int, end: int) -> str:
    """Extract lines [start, end] from source (1-indexed, inclusive)."""
    return "\n".join(source_lines[start - 1 : end])


def chunk_python_file(raw: RawFile) -> List[Chunk]:
    """
    Parse a Python file with the ast module and split into chunks,
    one per top-level function or class.  Methods inside classes are
    kept together with their class body.

    Falls back to recursive text splitting if AST parsing fails.
    """
    source = raw.content
    source_lines = source.splitlines()
    chunks: List[Chunk] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"[Chunker] AST parse failed for {raw.relative_path}: {e}. Using fallback.")
        return chunk_by_text(raw)

    # Collect top-level nodes with line info
    top_level_nodes = [
        node for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and hasattr(node, "lineno")
    ]

    if not top_level_nodes:
        # No functions/classes — treat whole file as one module-level chunk
        return [Chunk(
            text=source,
            relative_path=raw.relative_path,
            language="python",
            chunk_index=0,
            node_type="module_level",
            node_name=Path(raw.relative_path).stem,
            start_line=1,
            end_line=len(source_lines),
        )]

    # Track which lines are already covered by a top-level node
    covered_lines = set()

    for idx, node in enumerate(top_level_nodes):
        start_line = node.lineno
        end_line   = getattr(node, "end_lineno", start_line)

        segment = _get_source_segment(source_lines, start_line, end_line)

        # If segment is very long, sub-split it (large classes)
        if len(segment) > CHUNK_SIZE * 3:
            sub_chunks = _split_large_node(segment, raw, idx, node, start_line)
            chunks.extend(sub_chunks)
        else:
            node_type = (
                "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                else "class"
            )
            chunks.append(Chunk(
                text=segment,
                relative_path=raw.relative_path,
                language="python",
                chunk_index=idx,
                node_type=node_type,
                node_name=node.name,
                start_line=start_line,
                end_line=end_line,
            ))

        covered_lines.update(range(start_line, end_line + 1))

    # Capture module-level code not inside any function/class
    module_lines = [
        (i + 1, line) for i, line in enumerate(source_lines)
        if (i + 1) not in covered_lines and line.strip()
    ]

    if module_lines:
        module_text = "\n".join(line for _, line in module_lines)
        if module_text.strip():
            chunks.append(Chunk(
                text=module_text,
                relative_path=raw.relative_path,
                language="python",
                chunk_index=len(chunks),
                node_type="module_level",
                node_name=Path(raw.relative_path).stem + ":module",
                start_line=module_lines[0][0],
                end_line=module_lines[-1][0],
            ))

    return chunks


def _split_large_node(segment: str, raw: RawFile, base_idx: int,
                       node: ast.AST, start_line: int) -> List[Chunk]:
    """Sub-split a very large AST node by line groups."""
    lines = segment.splitlines()
    sub_chunks = []
    window = CHUNK_SIZE // 80  # approx lines per chunk
    for i in range(0, len(lines), window):
        block = "\n".join(lines[i : i + window])
        sub_chunks.append(Chunk(
            text=block,
            relative_path=raw.relative_path,
            language="python",
            chunk_index=base_idx + i,
            node_type="class_body_part",
            node_name=getattr(node, "name", "unknown") + f"[part{i}]",
            start_line=start_line + i,
            end_line=start_line + i + window,
        ))
    return sub_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Recursive text splitter (Markdown / YAML / plain text fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _recursive_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Tries to split on paragraph breaks, then newlines, then spaces.
    separators = ["\n\n", "\n", " ", ""]

    def split(text: str, separators: List[str]) -> List[str]:
        if len(text) <= chunk_size:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(sep) if sep else list(text)
        chunks, current = [], ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Part itself might be too large → recurse
                if len(part) > chunk_size and remaining_seps:
                    chunks.extend(split(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    raw_chunks = split(text, separators)

    # Apply overlap: prepend tail of previous chunk
    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0 or overlap == 0:
            overlapped.append(chunk)
        else:
            tail = raw_chunks[i - 1][-overlap:]
            overlapped.append(tail + "\n" + chunk)

    return overlapped


def chunk_by_text(raw: RawFile) -> List[Chunk]:
    """Fallback: split any file using recursive character splitting."""
    pieces = _recursive_split(raw.content, CHUNK_SIZE, CHUNK_OVERLAP)
    return [
        Chunk(
            text=piece,
            relative_path=raw.relative_path,
            language=raw.language,
            chunk_index=idx,
            node_type="text_chunk",
        )
        for idx, piece in enumerate(pieces)
        if piece.strip()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def chunk_file(raw: RawFile) -> List[Chunk]:
    """
    Route a RawFile to the right chunking strategy based on language.
    """
    if raw.language == "python":
        return chunk_python_file(raw)
    else:
        return chunk_by_text(raw)


def chunk_all_files(raw_files: List[RawFile]) -> List[Chunk]:
    """
    Chunk all loaded files and return a flat list of Chunk objects.
    """
    all_chunks: List[Chunk] = []

    for raw in raw_files:
        try:
            chunks = chunk_file(raw)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[Chunker] ERROR on {raw.relative_path}: {e}")

    print(f"[Chunker] ✅ Total chunks : {len(all_chunks)}")
    py_chunks = [c for c in all_chunks if c.language == "python"]
    print(f"[Chunker]    Python (AST) : {len(py_chunks)}")
    print(f"[Chunker]    Other        : {len(all_chunks) - len(py_chunks)}")

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ingestion.loader import load_repo, summarize_loaded_files

    files = load_repo()
    summarize_loaded_files(files)

    chunks = chunk_all_files(files)

    # Show a few sample Python chunks
    py_chunks = [c for c in chunks if c.language == "python" and c.node_type == "function"]
    print(f"\n── Sample function chunk ────────────────")
    if py_chunks:
        sample = py_chunks[0]
        print(f"File      : {sample.relative_path}")
        print(f"Function  : {sample.node_name}")
        print(f"Lines     : {sample.start_line}–{sample.end_line}")
        print(f"Text preview:\n{sample.text[:400]}")