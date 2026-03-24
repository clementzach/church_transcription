"""
Profanity filter module — imported by both app.py and tests.

Keeps filter logic independent of Flask/gevent so tests can import it
without triggering monkey-patching or requiring a running server.
"""

import os
import re
import logging
from better_profanity import profanity as _bp_filter

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


# ── Bootstrap better-profanity ─────────────────────────────────────────────
# Whitelist first: biblical/LDS vocabulary must never be censored.
_whitelist = _load_wordlist('whitelist.txt')
_bp_filter.load_censor_words(whitelist_words=_whitelist)

# Add multilingual profanity (ES, PT, HT).
# better-profanity uses whole-word matching so "assessment" ≠ "ass".
_custom_words = (
    _load_wordlist('es.txt') +
    _load_wordlist('pt.txt') +
    _load_wordlist('ht.txt')
)
_bp_filter.add_censor_words(_custom_words)

_n_builtin = len(_bp_filter.CENSOR_WORDSET) - len(_custom_words)
log.info(
    "Filter ready: %d EN built-in + %d multilingual custom + %d whitelisted",
    _n_builtin, len(_custom_words), len(_whitelist),
)

# ── ZH substring regex ────────────────────────────────────────────────────
# Chinese has no word boundaries; handled separately.
# Single characters with common non-vulgar uses (干, 日, 操) are excluded.
_zh_words = _load_wordlist('zh.txt')
_zh_re = re.compile('|'.join(re.escape(w) for w in _zh_words)) if _zh_words else None


def filter_text(text: str) -> str:
    """Censor profanity across all supported languages.

    Latin-script languages (EN, ES, PT, HT): better-profanity with whole-word
    matching and KJV/LDS whitelist.
    Mandarin (ZH): substring regex on multi-character sequences only.
    """
    text = _bp_filter.censor(text, censor_char=' ')
    if _zh_re:
        text = _zh_re.sub(' ', text)
    return text
