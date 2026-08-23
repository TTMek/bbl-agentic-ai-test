"""Keyword-based retrieval over a plain-text knowledge base.

This module implements the RAG retrieval layer: it loads the knowledge base,
splits it into paragraph-level chunks, and ranks those chunks against a query
using token overlap scoring.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base.txt"

# Common English words that carry no retrieval signal.
STOP_WORDS = frozenset(
    """
    a an and are as at be by can do does for from has have how i in is it its
    of on or that the this to what when where which who why with you your
    """.split()
)


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, and drop stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in STOP_WORDS and len(word) > 2}


@lru_cache(maxsize=1)
def load_chunks(path: Path = KNOWLEDGE_BASE_PATH) -> tuple[str, ...]:
    """Read the knowledge base and split it into paragraph chunks.

    Chunks are separated by blank lines. The result is cached so repeated
    tool calls within one run do not re-read the file from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found at: {path}")

    raw_text = path.read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", raw_text)]
    return tuple(chunk for chunk in chunks if chunk)


def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Return the knowledge base chunks most relevant to `query`.

    Scoring is the overlap between query tokens and chunk tokens, normalised
    by query length so longer queries are not unfairly favoured.

    Args:
        query: The user's information request.
        top_k: Maximum number of chunks to return.

    Returns:
        The matching chunks as raw text, or a not-found message.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return "No searchable terms found in the query."

    scored: list[tuple[float, str]] = []
    for chunk in load_chunks():
        overlap = len(query_tokens & _tokenize(chunk))
        if overlap:
            scored.append((overlap / len(query_tokens), chunk))

    if not scored:
        return "No relevant information found in the knowledge base."

    scored.sort(key=lambda item: item[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:top_k]]

    return "\n\n---\n\n".join(
        f"[Snippet {index}]\n{chunk}" for index, chunk in enumerate(top_chunks, start=1)
    )


if __name__ == "__main__":
    # Quick manual check of the retrieval layer, no LLM required.
    for demo_query in (
        "What is the policy on international travel?",
        "How many days of annual leave do I get?",
        "Can I work from home?",
    ):
        print(f"\n{'=' * 70}\nQUERY: {demo_query}\n{'=' * 70}")
        print(search_knowledge_base(demo_query))