import re
from typing import Dict, List, Pattern, Tuple

# -----------------------------
# Text preprocessing utilities (same style as evaluator)
# -----------------------------
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks (reasoning) from model output."""
    if not text:
        return ""
    return _THINK_BLOCK_RE.sub("", text).strip()


_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+|\n+")

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _normalize_lex_item(item: str) -> str:
    # Remove parenthetical hints like "children (offspring)"
    item = re.sub(r"\s*\(.*?\)\s*", "", item)
    item = item.strip().lower()
    item = re.sub(r"\s+", " ", item)
    return item


def _compile_lexicon_patterns(items: List[str]) -> List[Pattern]:
    """
    Compile regex patterns with word boundaries.
    Supports multi-word phrases by allowing flexible whitespace.
    """
    patterns: List[Pattern] = []
    for raw in items:
        item = _normalize_lex_item(raw)
        if not item:
            continue

        if " " in item:
            tokens = item.split()
            pat = r"\b" + r"\s+".join(re.escape(t) for t in tokens) + r"\b"
        else:
            pat = r"\b" + re.escape(item) + r"\b"

        patterns.append(re.compile(pat, flags=re.IGNORECASE))
    return patterns