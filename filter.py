"""
Profanity filter module — imported by both app.py and tests.

Keeps filter logic independent of Flask/gevent so tests can import it
without triggering monkey-patching or requiring a running server.

All filtering is done with explicit word lists in wordlists/*.txt.
Latin-script languages (EN, ES, PT, HT) use \\b word-boundary regex so
substrings are never matched ("assessment" is never flagged for "ass").
Mandarin (ZH) uses substring matching on multi-character sequences only.
"""

import os
import re
import logging

log = logging.getLogger(__name__)

_WORDLIST_DIR = os.path.join(os.path.dirname(__file__), 'wordlists')


def _load_wordlist(filename: str) -> list:
    """Load a word list file, stripping blank lines and # comments (inline too)."""
    path = os.path.join(_WORDLIST_DIR, filename)
    try:
        words = []
        with open(path, encoding='utf-8') as f:
            for ln in f:
                word = ln.split('#')[0].strip()
                if word:
                    words.append(word)
        return words
    except FileNotFoundError:
        log.warning("Word list not found: %s", filename)
        return []


def _build_latin_regex(words: list):
    """
    Build a case-insensitive word-boundary regex from a list of words/phrases.
    Sorted longest-first so multi-word phrases match before their component words.
    Returns None if the list is empty.
    """
    if not words:
        return None
    words = sorted(set(words), key=len, reverse=True)
    # \\b works at the edge of both single words and phrases
    pattern = r'\b(?:' + '|'.join(re.escape(w) for w in words) + r')\b'
    return re.compile(pattern, re.IGNORECASE)


# ── Load word lists ───────────────────────────────────────────────────────────
_latin_words = (
    _load_wordlist('en.txt') +
    _load_wordlist('es.txt') +
    _load_wordlist('pt.txt') +
    _load_wordlist('ht.txt')
)
_latin_re = _build_latin_regex(_latin_words)

# ZH: no word boundaries in Chinese; use substring matching on explicit sequences.
# Single characters with common non-vulgar uses (干, 日, 操) are excluded from zh.txt.
_zh_words = _load_wordlist('zh.txt')
_zh_re = re.compile('|'.join(re.escape(w) for w in _zh_words)) if _zh_words else None

log.info(
    "Filter ready: %d latin-script words/phrases, %d ZH sequences",
    len(_latin_words), len(_zh_words),
)


def filter_text(text: str) -> str:
    """Censor profanity across all supported languages, replacing matches with a space."""
    if _latin_re:
        text = _latin_re.sub(' ', text)
    if _zh_re:
        text = _zh_re.sub(' ', text)
    return text
