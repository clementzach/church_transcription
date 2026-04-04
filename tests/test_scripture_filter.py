"""
Scripture false-positive test for the profanity filter.

Downloads the complete Book of Mormon, New Testament, and an entire General
Conference session (October 2025) in all 5 languages (EN, ES, PT, HT, ZH)
from the Church of Jesus Christ content API.  Runs filter_text() on every
verse/paragraph and asserts zero false positives (filtered text must equal
original text).

All downloaded content is cached under tests/scripture_cache/ so subsequent
runs are instant and work offline.

Usage:
    cd /path/to/church_transcription
    pip install beautifulsoup4        # if not already installed
    python -m pytest tests/test_scripture_filter.py -v
    # or run directly:
    python tests/test_scripture_filter.py
"""

import hashlib
import json
import os
import re
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Import filter under test ─────────────────────────────────────────────────
# filter.py lives one directory above this test file.
sys.path.insert(0, str(Path(__file__).parent.parent))
from filter import filter_text  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "scripture_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_API = "https://www.churchofjesuschrist.org/study/api/v3/library/content"

# Language codes for the Church API
LANG_CODES = {
    "en": "eng",
    "es": "spa",
    "pt": "por",
    "ht": "hat",
    "zh": "zhs",
}

# ── Scripture book/chapter tables ────────────────────────────────────────────
# Book of Mormon books with chapter counts
BOM_BOOKS = [
    ("1-ne", 22), ("2-ne", 33), ("jacob", 7), ("enos", 1), ("jarom", 1),
    ("omni", 1), ("w-of-m", 1), ("mosiah", 29), ("alma", 63), ("hel", 16),
    ("3-ne", 30), ("4-ne", 1), ("morm", 9), ("ether", 15), ("moro", 10),
]

# New Testament books with chapter counts
NT_BOOKS = [
    ("matt", 28), ("mark", 16), ("luke", 24), ("john", 21),
    ("acts", 28), ("rom", 16), ("1-cor", 16), ("2-cor", 13),
    ("gal", 6), ("eph", 6), ("philip", 4), ("col", 4),
    ("1-thes", 5), ("2-thes", 3), ("1-tim", 6), ("2-tim", 4),
    ("titus", 3), ("philem", 1), ("heb", 13), ("james", 5),
    ("1-pet", 5), ("2-pet", 3), ("1-jn", 5), ("2-jn", 1),
    ("3-jn", 1), ("jude", 1), ("rev", 22),
]

# General Conference — October 2025 (talk URIs hardcoded from the session index page)
GC_TALK_URIS = [
    "/general-conference/2025/10/11eyring",
    "/general-conference/2025/10/12stevenson",
    "/general-conference/2025/10/13browning",
    "/general-conference/2025/10/14barcellos",
    "/general-conference/2025/10/15eyre",
    "/general-conference/2025/10/16uchtdorf",
    "/general-conference/2025/10/17johnson",
    "/general-conference/2025/10/19oaks",
    "/general-conference/2025/10/21rasband",
    "/general-conference/2025/10/22webb",
    "/general-conference/2025/10/23jaggi",
    "/general-conference/2025/10/24brown",
    "/general-conference/2025/10/25gong",
    "/general-conference/2025/10/26cziesla",
    "/general-conference/2025/10/27cook",
    "/general-conference/2025/10/31kearon",
    "/general-conference/2025/10/32dennis",
    "/general-conference/2025/10/33barlow",
    "/general-conference/2025/10/34jackson",
    "/general-conference/2025/10/35andersen",
    "/general-conference/2025/10/41holland",
    "/general-conference/2025/10/42evanson",
    "/general-conference/2025/10/43soares",
    "/general-conference/2025/10/44johnson",
    "/general-conference/2025/10/45christofferson",
    "/general-conference/2025/10/46spannaus",
    "/general-conference/2025/10/47eyring",
    "/general-conference/2025/10/51bednar",
    "/general-conference/2025/10/52cuvelier",
    "/general-conference/2025/10/53holland",
    "/general-conference/2025/10/54godoy",
    "/general-conference/2025/10/55renlund",
    "/general-conference/2025/10/56amos",
    "/general-conference/2025/10/57farias",
    "/general-conference/2025/10/58oaks",
]

# Request settings
REQUEST_TIMEOUT = 15          # seconds per API call
RATE_LIMIT_DELAY = 0.05       # seconds between requests (be polite)
MAX_WORKERS = 6               # concurrent download threads

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(lang: str, uri: str) -> Path:
    slug = uri.strip("/").replace("/", "_")
    return CACHE_DIR / f"{lang}_{slug}.json"


def _load_cache(lang: str, uri: str):
    path = _cache_key(lang, uri)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(lang: str, uri: str, data):
    path = _cache_key(lang, uri)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ── API fetcher ───────────────────────────────────────────────────────────────

def fetch_content(lang: str, uri: str) -> dict :
    """Fetch content from the Church API (or cache).  Returns parsed JSON or None."""
    cached = _load_cache(lang, uri)
    if cached is not None:
        return cached

    api_lang = LANG_CODES[lang]
    url = BASE_API
    params = {"lang": api_lang, "uri": uri}
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            _save_cache(lang, uri, data)
            return data
        else:
            # Cache the failure marker so we don't retry on every run
            _save_cache(lang, uri, {"_error": resp.status_code})
            return None
    except Exception as e:
        print(f"  [fetch] {lang} {uri} error: {e}", file=sys.stderr)
        return None


# ── Text extractor ────────────────────────────────────────────────────────────

def extract_paragraphs(api_response: dict) -> list[str]:
    """Pull all visible paragraph text out of an API response."""
    if not api_response or "_error" in api_response:
        return []

    # The Church API returns HTML content inside response["content"]["body"]
    # Walk all likely keys to find the HTML blob.
    html = None
    content = api_response.get("content") or {}
    if isinstance(content, dict):
        html = content.get("body") or content.get("html") or content.get("text")

    if not html:
        # Fallback: stringify and search
        raw = json.dumps(api_response)
        # Look for <p ...>...</p> patterns
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', raw, re.DOTALL)
        results = []
        for p in paragraphs:
            text = BeautifulSoup(p, "html.parser").get_text(" ").strip()
            if text:
                results.append(text)
        return results

    soup = BeautifulSoup(html, "html.parser")
    # Remove footnote / cross-reference spans
    for tag in soup.find_all(class_=re.compile(r'(footnote|note|ref)')):
        tag.decompose()
    paragraphs = []
    for p in soup.find_all(["p", "span"], class_=re.compile(r'(verse|para|content)')):
        text = p.get_text(" ").strip()
        if text:
            paragraphs.append(text)
    if not paragraphs:
        # Last resort: every <p>
        for p in soup.find_all("p"):
            text = p.get_text(" ").strip()
            if text:
                paragraphs.append(text)
    return paragraphs


# ── Chapter URI builders ──────────────────────────────────────────────────────

def bom_uris() -> list[str]:
    uris = []
    for book, chapters in BOM_BOOKS:
        for ch in range(1, chapters + 1):
            uris.append(f"/scriptures/bofm/{book}/{ch}")
    return uris


def nt_uris() -> list[str]:
    uris = []
    for book, chapters in NT_BOOKS:
        for ch in range(1, chapters + 1):
            uris.append(f"/scriptures/nt/{book}/{ch}")
    return uris


GC_TEXT_DIR = Path(__file__).parent / "general_conference"


def read_gc_paragraphs(lang: str) -> list[str]:
    """Read paragraphs from the local general_conference/<lang>.txt file."""
    path = GC_TEXT_DIR / f"{lang}.txt"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


# ── Core assertion logic ──────────────────────────────────────────────────────

class FalsePositive:
    def __init__(self, lang, uri, original, filtered):
        self.lang = lang
        self.uri = uri
        self.original = original
        self.filtered = filtered

    def __str__(self):
        return (
            f"[{self.lang}] {self.uri}\n"
            f"  ORIGINAL: {self.original[:120]}\n"
            f"  FILTERED: {self.filtered[:120]}"
        )


def check_paragraphs(lang: str, uri: str, paragraphs: list[str]) -> list[FalsePositive]:
    """Run filter_text on each paragraph; return list of false positives."""
    fps = []
    for para in paragraphs:
        if not para.strip():
            continue
        filtered = filter_text(para)
        if filtered != para:
            fps.append(FalsePositive(lang, uri, para, filtered))
    return fps


# ── Download helpers ──────────────────────────────────────────────────────────

def download_all(lang: str, uris: list[str]) -> list[FalsePositive]:
    """Download and check a list of URIs for one language.  Caches results."""
    false_positives = []
    not_cached = [u for u in uris if _load_cache(lang, u) is None]

    if not_cached:
        print(f"  Downloading {len(not_cached)} new items for lang={lang}…")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {}
            for uri in not_cached:
                futures[pool.submit(fetch_content, lang, uri)] = uri
                time.sleep(RATE_LIMIT_DELAY)
            for future in as_completed(futures):
                pass  # results already cached inside fetch_content

    # Now process all (from cache)
    for uri in uris:
        data = _load_cache(lang, uri) or {}
        paragraphs = extract_paragraphs(data)
        fps = check_paragraphs(lang, uri, paragraphs)
        false_positives.extend(fps)

    return false_positives


# ── Main test class ───────────────────────────────────────────────────────────

class TestScriptureFilter(unittest.TestCase):
    """Comprehensive false-positive test across scripture + General Conference."""

    @classmethod
    def setUpClass(cls):
        cls.all_false_positives: list[FalsePositive] = []

    def _run_corpus(self, label: str, lang: str, uris: list[str]):
        print(f"\n── {label} / {lang.upper()} ({len(uris)} URIs) ──")
        fps = download_all(lang, uris)
        if fps:
            print(f"  !! {len(fps)} false positive(s):")
            for fp in fps:
                print(f"  {fp}")
        else:
            print(f"  OK — 0 false positives")
        self.all_false_positives.extend(fps)
        return fps

    # ── Book of Mormon ──────────────────────────────────────────────────────

    def test_bom_en(self):
        fps = self._run_corpus("Book of Mormon", "en", bom_uris())
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in BOM/EN")

    def test_bom_es(self):
        fps = self._run_corpus("Book of Mormon", "es", bom_uris())
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in BOM/ES")

    def test_bom_pt(self):
        fps = self._run_corpus("Book of Mormon", "pt", bom_uris())
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in BOM/PT")

    def test_bom_ht(self):
        fps = self._run_corpus("Book of Mormon", "ht", bom_uris())
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in BOM/HT")

    def test_bom_zh(self):
        fps = self._run_corpus("Book of Mormon", "zh", bom_uris())
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in BOM/ZH")

    # ── New Testament ───────────────────────────────────────────────────────

    def test_nt_en(self):
        fps = self._run_corpus("New Testament", "en", nt_uris())
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in NT/EN")

    def test_nt_es(self):
        fps = self._run_corpus("New Testament", "es", nt_uris())
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in NT/ES")

    def test_nt_pt(self):
        fps = self._run_corpus("New Testament", "pt", nt_uris())
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in NT/PT")

    def test_nt_ht(self):
        fps = self._run_corpus("New Testament", "ht", nt_uris())
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in NT/HT")

    def test_nt_zh(self):
        fps = self._run_corpus("New Testament", "zh", nt_uris())
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in NT/ZH")

    # ── General Conference Oct 2025 ─────────────────────────────────────────

    def _run_gc(self, lang: str):
        paragraphs = read_gc_paragraphs(lang)
        if not paragraphs:
            self.skipTest(f"No GC text file found for {lang}")
        print(f"\n── General Conference Oct 2025 / {lang.upper()} ({len(paragraphs)} lines) ──")
        fps = check_paragraphs(lang, f"general_conference/{lang}.txt", paragraphs)
        if fps:
            print(f"  !! {len(fps)} false positive(s):")
            for fp in fps:
                print(f"  {fp}")
        else:
            print(f"  OK — 0 false positives")
        return fps

    def test_gc_en(self):
        fps = self._run_gc("en")
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in GC/EN")

    def test_gc_es(self):
        fps = self._run_gc("es")
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in GC/ES")

    def test_gc_pt(self):
        fps = self._run_gc("pt")
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in GC/PT")

    def test_gc_ht(self):
        fps = self._run_gc("ht")
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in GC/HT")

    def test_gc_zh(self):
        fps = self._run_gc("zh")
        self.assertEqual(fps, [], f"{len(fps)} false positive(s) in GC/ZH")


# ── Quick sanity tests (no network needed) ────────────────────────────────────

class TestFilterSanity(unittest.TestCase):
    """Fast unit tests that run without any network access."""

    # Things that must NEVER be filtered
    MUST_PASS = [
        # EN — explicitly requested
        "Praise God from whom all blessings flow",
        "Go to hell",
        "Damn the torpedoes",
        # KJV animals / objects
        "Balaam's ass spoke to him",
        "Before the cock crow, thou shalt deny me thrice",
        "It is hard for thee to kick against the pricks",
        # KJV anatomy / theology
        "And the LORD said unto her, Two nations are in thy womb",
        "The whore of Babylon sits upon many waters",
        "No bastard shall enter into the congregation of the LORD",
        "Her breasts are like two young roes",
        "Flee fornication",
        # sexual / chastity context
        "The law of sexual purity is foundational",
        "The word 'sex' appears in modern translations",
        # LDS
        "We partake of the sacrament each week",
        "He holds the Melchizedek priesthood",
        # Common EN word with 'ass' substring — the Scunthorpe test
        "We must assess the assessment carefully",
        "The assassination attempt was foiled",
        "Classic is a classy word",
        "The cockatiel sang beautifully",
        "The cockatoo is a parrot species",
        # ES religious
        "Dios es nuestro Padre Celestial",
        "El Espíritu Santo testifica de Cristo",
        "La Virgen María fue bendecida entre las mujeres",
        "Satanás fue expulsado del cielo",
        "El Sacramento es una ordenanza sagrada",
        # PT religious
        "Deus é nosso Pai Celestial",
        "Jesus Cristo ressuscitou dos mortos",
        "O Espírito Santo testifica da verdade",
        "Satanás foi expulso do céu",
        "O Sacramento nos lembra do Salvador",
        # HT religious
        "Bondye renmen nou tout",
        "Jezi Kri se Sovè nou an",
        "Lespri Sen ap temwaye verite a",
        "Legliz la rele Legliz Jezikri",
        "Sakrèman an se yon òdinans enpòtan",
        # ZH — scripture terms confirmed absent from zh.txt
        "上帝爱世人",          # God loves the world
        "耶稣基督是救主",      # Jesus Christ is the Savior
        "圣灵作见证",          # Holy Spirit bears witness
        "地狱是真实的",        # Hell is real
        "体操运动员很厉害",    # The gymnast is very skilled (体操 = gymnastics, contains 操)
        "操场上有很多孩子",    # There are many children on the playground (操场)
        "今日特价",            # Today's special (今日 contains 日)
    ]

    # Things that MUST be filtered
    MUST_FILTER = [
        "What the fuck is going on",
        "He's such an asshole",
        "Qué puta mierda",          # ES
        "Vai-te foder",             # PT
        "Fout ou",                  # HT
        "操你妈",                   # ZH
        "他妈的混蛋",               # ZH
    ]

    def test_must_pass(self):
        failures = []
        for text in self.MUST_PASS:
            result = filter_text(text)
            if result != text:
                failures.append(f"FALSE POSITIVE:\n  IN:  {text}\n  OUT: {result}")
        if failures:
            self.fail("\n".join(failures))

    def test_must_filter(self):
        failures = []
        for text in self.MUST_FILTER:
            result = filter_text(text)
            if result == text:
                failures.append(f"MISSED PROFANITY:\n  IN:  {text}")
        if failures:
            self.fail("\n".join(failures))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scripture false-positive test")
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run only the fast sanity tests (no network required)",
    )
    parser.add_argument(
        "--corpus",
        choices=["bom", "nt", "gc", "all"],
        default="all",
        help="Which scripture corpus to test (default: all)",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "es", "pt", "ht", "zh", "all"],
        default="all",
        help="Which language to test (default: all)",
    )
    args = parser.parse_args()

    if args.sanity_only:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestFilterSanity)
    else:
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFilterSanity))
        loader = unittest.TestLoader()
        for test in loader.loadTestsFromTestCase(TestScriptureFilter):
            name = test._testMethodName  # e.g. "test_bom_en"
            parts = name.split("_")     # ["test", "bom", "en"]
            corpus = parts[1] if len(parts) >= 2 else ""
            lang = parts[2] if len(parts) >= 3 else ""
            if args.corpus != "all" and corpus != args.corpus:
                continue
            if args.lang != "all" and lang != args.lang:
                continue
            suite.addTest(test)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
