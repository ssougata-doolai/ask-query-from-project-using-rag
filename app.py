# app.py — Streamlit UI for Codebase Q&A
# Run with:  streamlit run app.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_TF"]    = "1"
os.environ["TRANSFORMERS_NO_FLAX"]  = "1"

import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st

sys.path.append(str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title            = "Codebase Q&A",
    page_icon             = "🔍",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — fix #2 (overlapping text) and #3 (sidebar toggle styling)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] label {
    color: #8b949e !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 0.05em;
}

/* Sidebar collapse button — make it visible (#3) */
[data-testid="collapsedControl"],
button[kind="header"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #58a6ff !important;
    border-radius: 6px !important;
}

/* Header */
.app-header { padding: 0 0 1.2rem 0; border-bottom: 1px solid #21262d; margin-bottom: 1.5rem; }
.app-header h1 {
    font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600;
    color: #f0f6fc; margin: 0; letter-spacing: -0.02em;
}
.app-header .subtitle { font-size: 0.8rem; color: #8b949e; font-family: 'JetBrains Mono', monospace; }
.badge {
    display: inline-block; background: #161b22; border: 1px solid #30363d;
    border-radius: 4px; padding: 2px 8px;
    font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #58a6ff;
}
.badge.green { color: #3fb950; border-color: #238636; background: #0d1117; }
.badge.yellow { color: #d29922; border-color: #9e6a03; background: #0d1117; }

/* Chat messages */
.msg-user {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0; font-size: 0.95rem; color: #f0f6fc;
}
.msg-user::before {
    content: "YOU"; display: block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #8b949e; letter-spacing: 0.1em; margin-bottom: 6px;
}
.msg-assistant {
    background: #0d1117; border: 1px solid #1f6feb44;
    border-left: 3px solid #1f6feb; border-radius: 0 8px 8px 0;
    padding: 14px 18px; margin: 10px 0; font-size: 0.95rem;
    color: #c9d1d9; line-height: 1.65;
}
.msg-assistant::before {
    content: "ASSISTANT"; display: block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #1f6feb; letter-spacing: 0.1em; margin-bottom: 6px;
}

/* Source cards */
.sources-header {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #8b949e;
    letter-spacing: 0.1em; text-transform: uppercase; margin: 10px 0 6px 0;
}
.source-card {
    background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
    padding: 8px 12px; margin: 4px 0; font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; color: #8b949e;
    display: flex; justify-content: space-between; align-items: center;
}
.source-card .path { color: #58a6ff; }
.source-card .func { color: #3fb950; }
.source-card .score {
    background: #161b22; border: 1px solid #30363d; border-radius: 3px;
    padding: 1px 6px; color: #d29922; font-size: 0.68rem;
}

/* Stats bar */
.stats-bar {
    display: flex; gap: 16px; flex-wrap: wrap; padding: 8px 12px;
    background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
    margin-bottom: 14px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #8b949e;
}
.stats-bar span b { color: #f0f6fc; }

/* Input */
.stTextInput input {
    background: #0d1117 !important; border: 1px solid #30363d !important;
    border-radius: 6px !important; color: #f0f6fc !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important; padding: 10px 14px !important;
}
.stTextInput input:focus {
    border-color: #1f6feb !important; box-shadow: 0 0 0 3px #1f6feb22 !important;
}

/* Buttons */
.stButton button {
    background: #21262d !important; border: 1px solid #30363d !important;
    color: #c9d1d9 !important; border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important; padding: 6px 14px !important;
    transition: all 0.15s !important;
}
.stButton button:hover {
    background: #30363d !important; border-color: #58a6ff !important;
    color: #58a6ff !important;
}

/* FIX #2: raw chunk expander — prevent text overflow */
.streamlit-expanderContent {
    overflow-x: auto !important;
    max-width: 100% !important;
}
.streamlit-expanderContent pre,
.streamlit-expanderContent code {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    max-width: 100% !important;
}
/* Also fix st.code blocks inside expander */
.streamlit-expanderContent .stCode {
    overflow-x: auto !important;
    max-width: 100% !important;
}
[data-testid="stCodeBlock"] {
    max-width: 100% !important;
    overflow-x: auto !important;
}
[data-testid="stCodeBlock"] pre {
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

/* Expander header */
.streamlit-expanderHeader {
    background: #0d1117 !important; border: 1px solid #21262d !important;
    border-radius: 6px !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important; color: #8b949e !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 8px; padding: 12px 16px;
}
[data-testid="stMetricValue"] { color: #f0f6fc !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.72rem !important; }

hr { border-color: #21262d !important; }
code {
    background: #161b22 !important; border: 1px solid #30363d !important;
    border-radius: 4px !important; padding: 1px 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em !important; color: #ff7b72 !important;
}
pre code { padding: 12px !important; display: block; color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state  — fix #5: cap messages at MAX_MESSAGES
# ─────────────────────────────────────────────────────────────────────────────

MAX_MESSAGES = 10   # max user+assistant pairs stored (5 pairs = 10 entries)

def init_state():
    defaults = {
        "messages":       [],
        "chain":          None,
        "backend":        "chromadb",
        "top_k":          5,
        "min_score":      0.15,
        "search_mode":    "hybrid",
        "lang_filter":    "all",
        "total_queries":  0,
        "total_latency":  0.0,
        "index_count":    0,
        "pending_question": None,   # fix #4: store question from button click
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Load QA chain (cached per backend+mode)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_chain(backend: str, top_k: int, search_mode: str):
    from generation.qa_chain import CodebaseQAChain
    return CodebaseQAChain(top_k=top_k, backend=backend, search_mode=search_mode)


def get_chain():
    try:
        chain = load_chain(
            st.session_state.backend,
            st.session_state.top_k,
            st.session_state.search_mode,
        )
        st.session_state.index_count = chain.retriever.store.count
        return chain, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar  — fix #3: Streamlit has a built-in collapse arrow; we enhance it
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    # Backend selector
    new_backend = st.selectbox(
        "Vector Backend",
        options=["chromadb", "faiss"],
        index=0 if st.session_state.backend == "chromadb" else 1,
    )
    if new_backend != st.session_state.backend:
        st.session_state.backend  = new_backend
        st.session_state.messages = []
        st.cache_resource.clear()
        st.rerun()

    # Search mode
    new_mode = st.selectbox(
        "Search Mode",
        options=["hybrid", "semantic", "bm25"],
        index=["hybrid","semantic","bm25"].index(st.session_state.search_mode),
        help="hybrid=best quality | semantic=concept match | bm25=exact keywords",
    )
    if new_mode != st.session_state.search_mode:
        st.session_state.search_mode = new_mode
        st.cache_resource.clear()
        st.rerun()

    from config import LLM_MODEL, LLM_BACKEND, REPO_NAME, USE_RERANKER
    st.markdown(f"""
    <div style='margin:8px 0'>
        <div style='font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px'>LLM</div>
        <span class='badge'>{LLM_BACKEND}/{LLM_MODEL[:20]}</span>
    </div>
    <div style='margin:8px 0'>
        <div style='font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px'>Reranker</div>
        <span class='badge {"green" if USE_RERANKER else "yellow"}'>{"✓ enabled" if USE_RERANKER else "✗ off"}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.session_state.top_k = st.slider("Top-K chunks", 1, 10, st.session_state.top_k)
    st.session_state.min_score = st.slider("Min similarity", 0.0, 1.0,
                                            st.session_state.min_score, 0.05)
    st.session_state.lang_filter = st.selectbox(
        "Language filter",
        ["all","python","markdown","yaml","text"], index=0,
    )

    st.markdown("---")

    # Index status
    chain, err = get_chain()
    if chain:
        st.markdown(f"""
        <span class='badge green'>✓ {st.session_state.index_count} chunks</span>
        <span class='badge' style='margin-left:4px'>{REPO_NAME}</span>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge yellow'>⚠ not indexed</span>", unsafe_allow_html=True)
        st.caption("Run: `python main.py index`")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear chat"):
            st.session_state.messages      = []
            st.session_state.total_queries = 0
            st.session_state.total_latency = 0.0
            st.rerun()
    with col2:
        # History cap info
        n_pairs = len(st.session_state.messages) // 2
        st.caption(f"{n_pairs}/{MAX_MESSAGES//2} chats")

    if st.session_state.total_queries > 0:
        avg = st.session_state.total_latency / st.session_state.total_queries
        st.markdown(f"""
        <div style='margin-top:8px;font-size:.72rem;color:#8b949e'>
            Queries: <b style='color:#f0f6fc'>{st.session_state.total_queries}</b>
            &nbsp;|&nbsp; Avg: <b style='color:#f0f6fc'>{avg:.1f}s</b>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

from config import REPO_NAME
st.markdown(f"""
<div class="app-header">
    <h1>🔍 Codebase Q&A</h1>
    <div class="subtitle">RAG · {REPO_NAME} · {st.session_state.backend} · {st.session_state.search_mode}</div>
</div>
""", unsafe_allow_html=True)

if err:
    st.error(f"**Index not loaded:** {err}")
    st.info("Run `python main.py index --backend chromadb` to index the repo first.")
    st.stop()

# ── Suggested questions  FIX #4: use session_state flag, not rerun trick ──────
SUGGESTIONS = [
    "How does httpx handle timeouts?",
    "What authentication methods does httpx support?",
    "How does AsyncClient differ from the regular Client?",
    "How are redirects handled?",
    "How does httpx manage connection pooling?",
    "What does the _auth.py module do?",
]

if not st.session_state.messages:
    st.markdown("#### Try asking...")
    cols = st.columns(3)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 3].button(s, key=f"sug_{i}"):
            st.session_state.pending_question = s   # store, don't rerun yet


# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

        if msg.get("latency"):
            top_score = msg["sources"][0].score if msg.get("sources") else 0
            st.markdown(f"""
            <div class="stats-bar">
                <span>⏱ <b>{msg['latency']:.2f}s</b></span>
                <span>📄 <b>{len(msg.get('sources',[]))}</b> sources</span>
                <span>🎯 top score <b>{top_score:.3f}</b></span>
                <span>🔍 <b>{msg.get('search_mode','hybrid')}</b></span>
            </div>
            """, unsafe_allow_html=True)

        if msg.get("sources"):
            st.markdown('<div class="sources-header">Retrieved Sources</div>', unsafe_allow_html=True)
            for i, src in enumerate(msg["sources"], 1):
                # FIX #1: show full path including directory
                full_path  = src.relative_path
                func       = src.node_name or "—"
                lines      = f"L{src.start_line}–{src.end_line}" if src.start_line else ""
                st.markdown(f"""
                <div class="source-card">
                    <span>[{i}] <span class="path">{full_path}</span>
                    → <span class="func">{func}</span>
                    <span style='color:#484f58'>{lines}</span></span>
                    <span class="score">{src.score:.4f}</span>
                </div>
                """, unsafe_allow_html=True)

            # FIX #2: wrap code in container with overflow handling
            with st.expander("📋 View raw chunks", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}]** `{src.relative_path}` → `{src.node_name}`  "
                        f"*(lines {src.start_line}–{src.end_line})*"
                    )
                    # Truncate long chunks and wrap in scrollable container
                    chunk_text = src.text if len(src.text) <= 800 else src.text[:800] + "\n... (truncated)"
                    st.code(chunk_text, language=src.language)
                    st.markdown("---")

        st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Input area  — FIX #4 + FIX #5
# ─────────────────────────────────────────────────────────────────────────────

# FIX #4: pick up pending question from suggestion button
default_val = st.session_state.pop("pending_question", "") or ""

col_input, col_btn = st.columns([10, 1])
with col_input:
    question = st.text_input(
        label            = "question",
        value            = default_val,
        placeholder      = "Ask anything about the codebase…",
        label_visibility = "collapsed",
        key              = "question_input",
    )
with col_btn:
    send = st.button("→", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Handle query  — FIX #5: cap history + prevent duplicate submission
# ─────────────────────────────────────────────────────────────────────────────

def should_process(q: str) -> bool:
    """Return True only if this is a genuinely new question to process."""
    if not q or not q.strip():
        return False
    # Don't reprocess if the last user message is identical (prevents loops)
    if st.session_state.messages:
        last_user = next(
            (m for m in reversed(st.session_state.messages) if m["role"] == "user"), None
        )
        if last_user and last_user["content"] == q.strip():
            return False
    return True


if (send or default_val) and should_process(question):
    q = question.strip()

    # FIX #5: enforce message cap — remove oldest pair if needed
    while len(st.session_state.messages) >= MAX_MESSAGES:
        st.session_state.messages.pop(0)   # remove oldest user msg
        if st.session_state.messages:
            st.session_state.messages.pop(0)  # remove its assistant reply

    st.session_state.messages.append({"role": "user", "content": q})

    with st.spinner("Searching codebase and generating answer…"):
        t0 = time.time()
        try:
            lang = None if st.session_state.lang_filter == "all" else st.session_state.lang_filter
            result = chain.ask(
                question    = q,
                top_k       = st.session_state.top_k,
                mode        = st.session_state.search_mode,
                language    = lang,
                min_score   = st.session_state.min_score,
            )
            latency = time.time() - t0

            st.session_state.messages.append({
                "role":        "assistant",
                "content":     result.answer,
                "sources":     result.sources,
                "latency":     latency,
                "search_mode": st.session_state.search_mode,
            })
            st.session_state.total_queries += 1
            st.session_state.total_latency += latency

        except Exception as e:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"❌ Error: {e}",
                "sources": [],
                "latency": 0,
                "search_mode": "",
            })

    st.rerun()