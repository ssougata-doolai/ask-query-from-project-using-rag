# ingestion/loader.py
# Walks a local repo directory and loads all supported files with metadata.

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SUPPORTED_EXTENSIONS, EXCLUDED_DIRS, REPO_PATH, REPO_NAME


@dataclass
class RawFile:
    """Represents a single raw file loaded from the repo."""
    file_path: str          # absolute path
    relative_path: str      # path relative to repo root  e.g. "httpx/_client.py"
    language: str           # "python" | "markdown" | "yaml" | "text"
    content: str            # raw text content
    size_bytes: int
    extension: str


def is_excluded(path: Path, excluded_dirs: set) -> bool:
    """Return True if any part of the path is in the excluded set."""
    return any(part in excluded_dirs for part in path.parts)


def load_repo(repo_path: str = REPO_PATH) -> List[RawFile]:
    """
    Walk the repo directory and return a list of RawFile objects
    for every supported file found.

    Args:
        repo_path: Local path to the cloned repository.

    Returns:
        List of RawFile dataclass instances.
    """
    repo_root = Path(repo_path).resolve()

    if not repo_root.exists():
        raise FileNotFoundError(
            f"Repo not found at: {repo_root}\n"
            f"Clone it first:  git clone https://github.com/encode/httpx.git"
        )

    raw_files: List[RawFile] = []
    skipped = 0

    for file_path in repo_root.rglob("*"):
        # Only process files
        if not file_path.is_file():
            continue

        # Skip excluded directories
        relative = file_path.relative_to(repo_root)
        if is_excluded(relative, EXCLUDED_DIRS):
            skipped += 1
            continue

        # Only process supported extensions
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue

        # Read content — skip files that can't be decoded as UTF-8
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] Could not read {file_path}: {e}")
            skipped += 1
            continue

        # Skip empty files
        if not content.strip():
            skipped += 1
            continue

        raw_files.append(RawFile(
            file_path=str(file_path),
            relative_path=str(relative),
            language=SUPPORTED_EXTENSIONS[ext],
            content=content,
            size_bytes=file_path.stat().st_size,
            extension=ext,
        ))

    print(f"[Loader] ✅ Loaded  : {len(raw_files)} files")
    print(f"[Loader] ⏭  Skipped : {skipped} files")
    print(f"[Loader] 📂 Repo    : {repo_root}")

    return raw_files


def summarize_loaded_files(files: List[RawFile]) -> None:
    """Print a breakdown of loaded files by language."""
    from collections import Counter
    counts = Counter(f.language for f in files)
    print("\n── File breakdown ───────────────────────")
    for lang, count in counts.most_common():
        print(f"  {lang:<12} {count:>4} files")
    print(f"  {'TOTAL':<12} {len(files):>4} files")
    print("─────────────────────────────────────────\n")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    files = load_repo()
    summarize_loaded_files(files)

    # Preview first Python file
    py_files = [f for f in files if f.language == "python"]
    if py_files:
        sample = py_files[0]
        print(f"\nSample file : {sample.relative_path}")
        print(f"Size        : {sample.size_bytes} bytes")
        print(f"Preview     :\n{sample.content[:300]}")
        print(f"End preview")