"""Configuration for the Britannica OCR article parser."""

from pathlib import Path

# Repository root (one level up from scripts/)
REPO_DIR = Path(__file__).resolve().parent.parent

# Paths — relative to repo root
INPUT_DIR = REPO_DIR / "data" / "ocr"
OUTPUT_DIR = REPO_DIR / "data"
PARAGRAPHS_DIR = OUTPUT_DIR / "paragraphs"
CLASSIFICATIONS_DIR = OUTPUT_DIR / "classifications"
ARTICLES_DIR = OUTPUT_DIR / "articles"
EXPORT_DIR = OUTPUT_DIR / "export"
SITE_DIR = REPO_DIR / "docs"
DEDUP_MANIFEST = OUTPUT_DIR / "dedup_manifest.json"
OCR_MANIFEST = INPUT_DIR / "ocr_manifest.json"
COMPARISON_REPORT = OUTPUT_DIR / "comparison_report.json"
ORDER_AUDIT_REPORT = OUTPUT_DIR / "order_audit_report.json"

# LIS parser — 1842 index for Title Case validation
INDEX_1842_PATH = REPO_DIR / "old" / "data" / "output_v2" / "index_1842.jsonl"

# Headword dictionary: consolidated headwords from LLM, Gemini, and docs_old sources
HEADWORD_DICT_PATH = REPO_DIR / "data" / "headword_dictionary.json"

# Supplementary headings: pre-validated missed headings (Gemini-classified + alpha-filtered)
SUPPLEMENTARY_HEADINGS_PATH = REPO_DIR / "data" / "missed_headings_filtered.jsonl"

# docs_old directory: fallback articles for volumes without OCR files
DOCS_OLD_DIR = REPO_DIR / "old" / "data" / "docs_old"

# Cross-edition confidence threshold (articles below this are low-confidence)
CONFIDENCE_THRESHOLD = 0.6

# Editions with full OCR coverage — DO NOT use docs_old fallback for these.
# docs_old was DERIVED from the same OCR files, so falling back to it is circular.
# Based on ocr_manifest.json + ocr_nibi ingest (2026-02-27):
#   1771: 3/3 vols, 1778: 10/10, 1797: 18/18, 1823: 20/20, 1842: 21/21, 1860: 21/21
# Editions with genuinely missing OCR (NLS PDFs not available):
#   1810: 19/20 vols (vol 20 missing)
#   1815: 11/20 vols (9 vols missing: 3,7,9,10,12,13,15,19,20)
FULL_OCR_EDITIONS = {1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860}

# LLM API (for classify.py and LIS parser)
API_BASE = "http://platogpu001:8000/v1"
API_URL = f"{API_BASE}/chat/completions"
MODEL = "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"

# Batching
BATCH_SIZE = 20         # paragraphs per LLM call
OVERLAP = 2             # paragraphs overlap between windows
STEP_SIZE = BATCH_SIZE - OVERLAP  # effective advancement per call
PREVIEW_LENGTH = 300    # chars to include in LLM prompt

# Concurrency
MAX_CONCURRENT = 10     # simultaneous API requests
REQUEST_TIMEOUT = 120   # seconds per API call
MAX_RETRIES = 3         # retries on failure

# LLM parameters
LLM_TEMPERATURE = 0.1   # low temperature for consistent classification
LLM_MAX_TOKENS = 2048   # guided_json = no thinking tokens, all useful output

# Edition metadata (for prompt context)
EDITIONS = {
    "1st": {"year": 1771, "name": "1st", "full_name": "First Edition"},
    "2nd": {"year": 1778, "name": "2nd", "full_name": "Second Edition"},
    "3rd": {"year": 1797, "name": "3rd", "full_name": "Third Edition"},
    "4th": {"year": 1810, "name": "4th", "full_name": "Fourth Edition"},
    "5th": {"year": 1815, "name": "5th", "full_name": "Fifth Edition"},
    "6th": {"year": 1823, "name": "6th", "full_name": "Sixth Edition"},
    "7th": {"year": 1842, "name": "7th", "full_name": "Seventh Edition"},
    "8th": {"year": 1860, "name": "8th", "full_name": "Eighth Edition"},
}


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [OUTPUT_DIR, PARAGRAPHS_DIR, CLASSIFICATIONS_DIR, ARTICLES_DIR, EXPORT_DIR, SITE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
