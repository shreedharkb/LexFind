"""
LexFind — Metadata Extractor v2 (Legal-Aware Chunking)
=========================================================
Processes all 46k PDFs in data/pdfs/, extracts text + metadata via regex,
and chunks full text using a two-tier legal-aware strategy:

  Tier 1 — Legal-Aware:  Detects JUDIS section markers (HELD, FACTS, etc.)
                          and chunks within each section using NLTK sentences.
                          Tags every chunk with section_type.

  Tier 2 — Sentence-Aware Fallback: Used when doc structure is not detectable.
                          Chunks by sentence boundary with word-count limit.

Resume capability:
  - Default: skip PDFs already in legal_documents (their chunks are intact).
  - --rechunk: Wipe all legal_chunks for all docs and re-chunk with the new
               legal-aware strategy. Use this after the migration if you want
               to upgrade existing naive chunks.

Run from backend/:
  python scripts/qdrant_ingestion/metadata_extractor.py
  python scripts/qdrant_ingestion/metadata_extractor.py --limit 500   # test run
  python scripts/qdrant_ingestion/metadata_extractor.py --rechunk      # upgrade existing chunks
"""

import argparse
import logging
import os
import re
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

import fitz           # PyMuPDF
import ftfy
import nltk
from tqdm import tqdm
from sqlalchemy import create_engine, text

# Download NLTK punkt tokenizer (silent if already present)
nltk.download("punkt",          quiet=True)
nltk.download("punkt_tab",      quiet=True)

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("[WARN] langdetect not installed. pip install langdetect")

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
PDF_DIR      = Path(os.getenv("PDF_DIR", str(BASE_DIR / "data" / "pdfs")))
LOG_FILE     = BASE_DIR / "scripts" / "qdrant_ingestion" / "ingestion_errors.log"
BATCH_SIZE   = 100
MAX_CHUNK_WORDS      = 280
OVERLAP_SENTENCES    = 2   # sentences carried over as context

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_text(pdf_path: Path) -> tuple[str, int]:
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    page_count = len(doc)
    doc.close()
    full_text = ftfy.fix_text("\n".join(pages))
    return " ".join(full_text.split()), page_count


# ═════════════════════════════════════════════════════════════════════════════
# LEGAL-AWARE CHUNKING (TIER 1)
# ═════════════════════════════════════════════════════════════════════════════

LEGAL_SECTION_MARKERS = {
    "header": [
        r"PETITIONER\s*:",
        r"RESPONDENT\s*:",
        r"DATE\s+OF\s+JUDGMENT",
        r"DATE\s+OF\s+DECISION",
    ],
    "citation": [
        r"CITATION\s*:",
        r"\d{4}\s+AIR\s+\d+",
        r"AIR\s+\d{4}\s+SC",
        r"\(\d{4}\)\s+\d+\s+SCC",
    ],
    "bench": [
        r"^BENCH\s*:",
        r"^CORAM\s*:",
        r"HON['']?BLE",
    ],
    "headnote": [
        r"HEADNOTE\s*:",
        r"HEAD\s*NOTE\s*:",
        r"^HEADNOTES",
    ],
    "acts": [
        r"^ACT\s*:",
        r"^ACTS\s*:",
        r"ACTS\s+REFERRED",
    ],
    "facts": [
        r"\bFACTS\s*:",
        r"\bBACKGROUND\s*:",
        r"THE\s+FACTS\b",
        r"STATEMENT\s+OF\s+FACTS",
        r"BRIEF\s+FACTS",
    ],
    "issues": [
        r"\bISSUE\s*:",
        r"\bISSUES\s*:",
        r"\bQUESTION\s+OF\s+LAW",
        r"POINT\s+FOR\s+DETERMINATION",
        r"POINTS?\s+FOR\s+CONSIDERATION",
    ],
    "held": [
        r"\bHELD\b\s*:",
        r"\bHELD\s*,",
        r"\bIT\s+IS\s+HELD\b",
        r"\bWE\s+HOLD\b",
        r"\bTHE\s+COURT\s+HELD\b",
        r"\bCOURT\s+HELD\b",
    ],
    "judgment": [
        r"^JUDGMENT\s*:",
        r"^J\.\s*[-–—]",
        r"^[A-Z][A-Z\s]+,\s*J\.\s*[-–—:]",
        r"^OPINION\s+OF\s+THE\s+COURT",
    ],
    "ratio": [
        r"RATIO\s+DECIDENDI",
        r"RATIO\s*:",
        r"PRINCIPLE\s+LAID\s+DOWN",
    ],
    "order": [
        r"\bIN\s+THE\s+RESULT\b",
        r"\bIN\s+RESULT\b",
        r"\bACCORDINGLY\b",
        r"\bDISPOSED\s+OF\b",
        r"\bAPPEAL\s+(IS\s+)?DISMISSED\b",
        r"\bAPPEAL\s+(IS\s+)?ALLOWED\b",
        r"^ORDER\s*:",
    ],
}

# Pre-compile all patterns (case insensitive, multiline for ^ anchors)
_COMPILED_MARKERS: list[tuple[str, re.Pattern]] = []
for section, patterns in LEGAL_SECTION_MARKERS.items():
    for pat in patterns:
        _COMPILED_MARKERS.append((section, re.compile(pat, re.IGNORECASE | re.MULTILINE)))


def detect_legal_sections(text: str) -> list[tuple[int, str]]:
    """
    Scan text for section markers. Returns a list of (char_offset, section_type)
    sorted by offset. Deduplicates overlapping matches, keeping the one with
    the lowest offset for each 100-char window.
    """
    hits: list[tuple[int, str]] = []
    for section_type, pattern in _COMPILED_MARKERS:
        for m in pattern.finditer(text):
            hits.append((m.start(), section_type))

    if not hits:
        return []

    # Sort by position
    hits.sort(key=lambda x: x[0])

    # Deduplicate: if two hits are within 80 chars, keep first
    deduped: list[tuple[int, str]] = [hits[0]]
    for pos, sec in hits[1:]:
        if pos - deduped[-1][0] > 80:
            deduped.append((pos, sec))

    return deduped


def _sentences_to_chunks(
    sentences: list[str],
    section_type: str,
    strategy: str,
    max_words: int = MAX_CHUNK_WORDS,
    overlap_sents: int = OVERLAP_SENTENCES,
) -> list[dict]:
    """Accumulate sentences into word-capped chunks with sentence-boundary overlap."""
    chunks: list[dict] = []
    current: list[str] = []
    current_wc = 0

    for sent in sentences:
        wc = len(sent.split())
        if current_wc + wc > max_words and current:
            chunks.append({
                "text":          " ".join(current),
                "section_type":  section_type,
                "chunk_strategy": strategy,
            })
            # Keep last N sentences as overlap context
            current   = current[-overlap_sents:]
            current_wc = sum(len(s.split()) for s in current)

        current.append(sent)
        current_wc += wc

    if current:
        chunks.append({
            "text":          " ".join(current),
            "section_type":  section_type,
            "chunk_strategy": strategy,
        })
    return chunks


def legal_aware_chunk(text: str, section_hits: list[tuple[int, str]]) -> list[dict]:
    """
    Tier 1: Split text at detected section boundaries, then chunk each
    section with sentence-aware logic. Tags every chunk with section_type.
    """
    all_chunks: list[dict] = []

    # Build section slices: (start, end, section_type)
    slices: list[tuple[int, int, str]] = []
    for i, (start, sec) in enumerate(section_hits):
        end = section_hits[i + 1][0] if i + 1 < len(section_hits) else len(text)
        slices.append((start, end, sec))

    # Add any leading text before first marker as "header"
    if section_hits and section_hits[0][0] > 0:
        slices.insert(0, (0, section_hits[0][0], "header"))

    for start, end, sec in slices:
        segment = text[start:end].strip()
        if not segment:
            continue
        sentences = nltk.sent_tokenize(segment)
        all_chunks.extend(
            _sentences_to_chunks(sentences, sec, "legal_aware")
        )

    return all_chunks


def sentence_aware_chunk(text: str) -> list[dict]:
    """
    Tier 2 Fallback: No section structure detected. Chunk by sentence
    boundary with word-count limit. All chunks tagged section_type='unknown'.
    """
    sentences = nltk.sent_tokenize(text)
    return _sentences_to_chunks(sentences, "unknown", "sentence_aware")


def chunk_document(text: str) -> tuple[list[dict], str]:
    """
    Dispatcher: tries Tier 1 (legal-aware), falls back to Tier 2.
    Returns (chunks_list, majority_strategy).
    """
    section_hits = detect_legal_sections(text)

    # Need at least 3 distinct sections for Tier 1 to be meaningful
    unique_sections = {s for _, s in section_hits}
    if len(unique_sections) >= 3:
        chunks = legal_aware_chunk(text, section_hits)
        strategy = "legal_aware"
    else:
        chunks = sentence_aware_chunk(text)
        strategy = "sentence_aware"

    return chunks, strategy


# ═════════════════════════════════════════════════════════════════════════════
# COURT → STATE MAPPING
# ═════════════════════════════════════════════════════════════════════════════

COURT_STATE_MAP: dict[str, Optional[str]] = {
    "Supreme Court":                          None,
    "High Court of Bombay":                   "Maharashtra",
    "High Court of Madras":                   "Tamil Nadu",
    "High Court of Calcutta":                 "West Bengal",
    "High Court of Delhi":                    "Delhi",
    "High Court of Allahabad":                "Uttar Pradesh",
    "High Court of Kerala":                   "Kerala",
    "High Court of Gujarat":                  "Gujarat",
    "High Court of Rajasthan":                "Rajasthan",
    "High Court of Punjab and Haryana":       "Punjab",
    "High Court of Madhya Pradesh":           "Madhya Pradesh",
    "High Court of Karnataka":                "Karnataka",
    "High Court of Andhra Pradesh":           "Andhra Pradesh",
    "High Court of Telangana":                "Telangana",
    "High Court of Orissa":                   "Odisha",
    "High Court of Gauhati":                  "Assam",
    "High Court of Patna":                    "Bihar",
    "High Court of Himachal Pradesh":         "Himachal Pradesh",
    "High Court of Jammu and Kashmir":        "Jammu and Kashmir",
    "High Court of Jharkhand":                "Jharkhand",
    "High Court of Chhattisgarh":             "Chhattisgarh",
    "High Court of Uttarakhand":              "Uttarakhand",
    "High Court of Manipur":                  "Manipur",
    "High Court of Tripura":                  "Tripura",
    "High Court of Meghalaya":                "Meghalaya",
    "High Court of Sikkim":                   "Sikkim",
}

MONTH_MAP: dict[str, int] = {
    "january": 1,  "february": 2,  "march": 3,    "april": 4,
    "may": 5,      "june": 6,      "july": 7,     "august": 8,
    "september": 9,"october": 10,  "november": 11,"december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


# ═════════════════════════════════════════════════════════════════════════════
# METADATA EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_metadata(text: str, filename: str) -> dict:
    header     = text[:3000]
    header_up  = header.upper()
    result = dict(
        title=None, petitioner=None, respondent=None,
        court=None, state=None, year=None, date_of_judgment=None,
        judges=[], citation=None, acts_referred=[], bench_strength=0,
        case_type=None,
    )

    # ── Court ─────────────────────────────────────────────────────────────────
    court_patterns: list[tuple[str, str]] = [
        (r"SUPREME\s+COURT\s+OF\s+INDIA",                    "Supreme Court"),
        (r"IN\s+THE\s+SUPREME\s+COURT",                      "Supreme Court"),
        (r"HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+BOMBAY",   "High Court of Bombay"),
        (r"HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+MADRAS",   "High Court of Madras"),
        (r"HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+CALCUTTA", "High Court of Calcutta"),
        (r"HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+ALLAHABAD","High Court of Allahabad"),
        (r"HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+JABALPUR", "High Court of Madhya Pradesh"),
        (r"HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+JODHPUR",  "High Court of Rajasthan"),
        (r"HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+HYDERABAD","High Court of Andhra Pradesh"),
        (r"HIGH\s+COURT\s+OF\s+DELHI",                       "High Court of Delhi"),
        (r"HIGH\s+COURT\s+OF\s+BOMBAY",                      "High Court of Bombay"),
        (r"HIGH\s+COURT\s+OF\s+MADRAS",                      "High Court of Madras"),
        (r"HIGH\s+COURT\s+OF\s+CALCUTTA",                    "High Court of Calcutta"),
        (r"HIGH\s+COURT\s+OF\s+ALLAHABAD",                   "High Court of Allahabad"),
        (r"HIGH\s+COURT\s+OF\s+KERALA",                      "High Court of Kerala"),
        (r"HIGH\s+COURT\s+OF\s+GUJARAT",                     "High Court of Gujarat"),
        (r"HIGH\s+COURT\s+OF\s+RAJASTHAN",                   "High Court of Rajasthan"),
        (r"HIGH\s+COURT\s+OF\s+PUNJAB\s+AND\s+HARYANA",     "High Court of Punjab and Haryana"),
        (r"HIGH\s+COURT\s+OF\s+MADHYA\s+PRADESH",           "High Court of Madhya Pradesh"),
        (r"HIGH\s+COURT\s+OF\s+KARNATAKA",                   "High Court of Karnataka"),
        (r"HIGH\s+COURT\s+OF\s+ANDHRA\s+PRADESH",           "High Court of Andhra Pradesh"),
        (r"HIGH\s+COURT\s+OF\s+TELANGANA",                   "High Court of Telangana"),
        (r"HIGH\s+COURT\s+OF\s+ORISSA",                      "High Court of Orissa"),
        (r"HIGH\s+COURT\s+OF\s+GAUHATI",                     "High Court of Gauhati"),
        (r"HIGH\s+COURT\s+OF\s+PATNA",                       "High Court of Patna"),
        (r"HIGH\s+COURT\s+OF\s+HIMACHAL\s+PRADESH",         "High Court of Himachal Pradesh"),
        (r"HIGH\s+COURT\s+OF\s+JAMMU\s+AND\s+KASHMIR",      "High Court of Jammu and Kashmir"),
        (r"HIGH\s+COURT\s+OF\s+JHARKHAND",                   "High Court of Jharkhand"),
        (r"HIGH\s+COURT\s+OF\s+CHHATTISGARH",               "High Court of Chhattisgarh"),
        (r"HIGH\s+COURT\s+OF\s+UTTARAKHAND",                 "High Court of Uttarakhand"),
        (r"HIGH\s+COURT\s+OF\s+MANIPUR",                     "High Court of Manipur"),
        (r"HIGH\s+COURT\s+OF\s+TRIPURA",                     "High Court of Tripura"),
        (r"HIGH\s+COURT\s+OF\s+MEGHALAYA",                   "High Court of Meghalaya"),
        (r"HIGH\s+COURT\s+OF\s+SIKKIM",                      "High Court of Sikkim"),
    ]
    for pat, normalized in court_patterns:
        if re.search(pat, header_up):
            result["court"] = normalized
            result["state"] = COURT_STATE_MAP.get(normalized)
            break

    # ── Date and Year ─────────────────────────────────────────────────────────
    date_obj = None
    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    m = re.search(
        r"DATE\s+OF\s+(?:JUDGMENT|DECISION|ORDER)\s*[:\-]\s*"
        r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})",
        header_up,
    )
    if m:
        try:
            date_obj = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    # DD Month YYYY
    if not date_obj:
        m = re.search(
            r"DATE\s+OF\s+(?:JUDGMENT|DECISION|ORDER)\s*[:\-]\s*"
            r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)[,\s]+(\d{4})",
            header, re.IGNORECASE,
        )
        if m and m.group(2).lower() in MONTH_MAP:
            try:
                date_obj = date(int(m.group(3)), MONTH_MAP[m.group(2).lower()], int(m.group(1)))
            except ValueError:
                pass
    # Decided on: DD.MM.YYYY
    if not date_obj:
        m = re.search(
            r"(?:Decided\s+on|Date\s+of\s+Decision)\s*[:\-]?\s*(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})",
            header, re.IGNORECASE,
        )
        if m:
            try:
                date_obj = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                pass
    # Month D, YYYY
    if not date_obj:
        m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", header)
        if m and m.group(1).lower() in MONTH_MAP:
            try:
                date_obj = date(int(m.group(3)), MONTH_MAP[m.group(1).lower()], int(m.group(2)))
            except ValueError:
                pass

    if date_obj:
        result["date_of_judgment"] = date_obj
        result["year"] = date_obj.year
    else:
        m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", filename)
        if m:
            result["year"] = int(m.group(1))
        else:
            m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", header)
            if m:
                result["year"] = int(m.group(1))

    # ── Petitioner / Respondent ────────────────────────────────────────────────
    m = re.search(
        r"PETITIONER\s*[:\-]\s*(.+?)\s*RESPONDENT\s*[:\-]\s*(.+?)(?:\n|$|BENCH|DATE|JUDGE)",
        header, re.IGNORECASE | re.DOTALL,
    )
    if m:
        p = re.sub(r"\s+", " ", m.group(1)).strip()[:300]
        r_ = re.sub(r"\s+", " ", m.group(2)).strip()[:300]
        result["petitioner"] = p
        result["respondent"] = r_
        result["title"] = f"{p} vs {r_}"

    if not result["petitioner"]:
        m = re.search(
            r"((?:M/s\.?\s*|Smt\.?\s*|Shri\s*|Dr\.?\s*|Sri\s*)?[A-Z][^\n]{3,80}?)\s+"
            r"\.{3,}\s*(?:Appellant|Plaintiff|Petitioner)\s+"
            r"(?:Versus|Vs\.?|V\.?)\s+"
            r"((?:M/s\.?\s*|Smt\.?\s*|Shri\s*|Dr\.?\s*|Sri\s*)?[A-Z][^\n]{3,80}?)\s+"
            r"\.{3,}\s*(?:Respondent|Defendant|Opposite\s+Party)",
            header, re.IGNORECASE | re.DOTALL,
        )
        if m:
            p = re.sub(r"\s+", " ", m.group(1)).strip()[:300]
            r_ = re.sub(r"\s+", " ", m.group(2)).strip()[:300]
            result["petitioner"] = p
            result["respondent"] = r_
            result["title"] = f"{p} vs {r_}"

    if not result["petitioner"]:
        m = re.search(
            r"((?:(?:M/s\.?|Smt\.?|Shri|Dr\.?|Sri|Union\s+of\s+India|State\s+of\s+\w+)\s+)?[A-Z][A-Za-z\s\.\,\(\)&]{5,100}?)"
            r"\s+(?:Vs\.?|V\.?|VERSUS|versus)\s+"
            r"((?:(?:M/s\.?|Smt\.?|Shri|Dr\.?|Sri|Union\s+of\s+India|State\s+of\s+\w+)\s+)?[A-Z][A-Za-z\s\.\,\(\)&]{5,100})",
            header,
        )
        if m:
            p = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(",")[:300]
            r_ = re.sub(r"\s+", " ", m.group(2)).strip().rstrip(",")[:300]
            result["petitioner"] = p
            result["respondent"] = r_
            result["title"] = f"{p} vs {r_}"

    # ── Judges ─────────────────────────────────────────────────────────────────
    judges: set[str] = set()
    for m in re.finditer(r"BENCH\s*[:\-]\s*([A-Z][A-Z\s\.\,]+)", header_up):
        raw = m.group(1).strip().rstrip(",")
        if raw and len(raw) > 2:
            judges.add(_clean_judge_name(raw))

    m = re.search(
        r"CORAM\s*[:\-]\s*(.+?)(?:\n\n|\.|DATE|BENCH)",
        header, re.IGNORECASE | re.DOTALL,
    )
    if m:
        for part in re.split(r",|AND|&", m.group(1), flags=re.IGNORECASE):
            part = re.sub(r"\b(?:J\.|CJ\.|JJ\.|HON'BLE|MR\.|MS\.|JUSTICE|MRS\.)\b", "", part, flags=re.IGNORECASE)
            part = re.sub(r"\s+", " ", part).strip().rstrip(".")
            if part and len(part) > 3:
                judges.add(_clean_judge_name(part))

    for m in re.finditer(r"HON'?BLE\s+(?:MR\.?|MS\.?|MRS\.?)?\s*JUSTICE\s+([A-Z][A-Z\s\.]+)", header_up):
        name = _clean_judge_name(m.group(1))
        if name:
            judges.add(name)

    result["judges"] = sorted(judges)
    result["bench_strength"] = len(judges)

    # ── Citation ───────────────────────────────────────────────────────────────
    citation_patterns = [
        r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",
        r"\d{4}\s+\(\d+\)\s+SCC\s+\d+",
        r"\d{4}\s+AIR\s+\d+",
        r"AIR\s+\d{4}\s+SC\s+\d+",
        r"\d{4}\s+SCR\s+(?:\(\d+\)\s+)?\d+",
        r"MANU/[A-Z]+/\d+/\d{4}",
        r"\(\d{4}\)\s+\d+\s+(?:SCR|SCC|AIR|SCJ)\s+\d+",
    ]
    for pat in citation_patterns:
        m = re.search(pat, header, re.IGNORECASE)
        if m:
            result["citation"] = m.group(0).strip()
            break

    # ── Case Type ──────────────────────────────────────────────────────────────
    case_type_patterns = [
        (r"CRIMINAL\s+APPELLATE\s+JURISDICTION", "Criminal Appeal"),
        (r"CRIMINAL\s+APPEAL",                   "Criminal Appeal"),
        (r"CIVIL\s+APPELLATE\s+JURISDICTION",    "Civil Appeal"),
        (r"CIVIL\s+APPEAL",                      "Civil Appeal"),
        (r"SPECIAL\s+LEAVE\s+PETITION|SLP\s*\(","SLP"),
        (r"WRIT\s+PETITION",                     "Writ Petition"),
        (r"ORIGINAL\s+JURISDICTION",             "Original"),
        (r"REVIEW\s+PETITION",                   "Review"),
        (r"TRANSFER\s+PETITION",                 "Transfer"),
        (r"CONTEMPT\s+PETITION",                 "Contempt"),
        (r"CURATIVE\s+PETITION",                 "Curative"),
        (r"ELECTION\s+PETITION",                 "Election"),
    ]
    for pat, case_type in case_type_patterns:
        if re.search(pat, header_up):
            result["case_type"] = case_type
            break

    # ── Acts Referred ──────────────────────────────────────────────────────────
    acts: set[str] = set()
    abbrev_map = {
        r"\bIPC\b":                  "Indian Penal Code, 1860",
        r"\bCPC\b":                  "Code of Civil Procedure, 1908",
        r"\bCrPC\b|Cr\.P\.C\.":     "Code of Criminal Procedure, 1973",
        r"Constitution\s+of\s+India":"Constitution of India",
    }
    for pat, full in abbrev_map.items():
        if re.search(pat, text[:5000], re.IGNORECASE):
            acts.add(full)
    for m in re.finditer(r"(?:the\s+)?([A-Z][A-Za-z\s\(\)]{5,60}?Act(?:,\s*\d{4})?)", text[:5000]):
        act = re.sub(r"\s+", " ", m.group(1)).strip()
        if 8 < len(act) < 100:
            acts.add(act)
    result["acts_referred"] = sorted(acts)[:20]

    return result


def _clean_judge_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", raw).strip()
    name = re.sub(r"[,\.]+$", "", name).strip()
    return name.title() if len(name) > 2 else ""


# ═════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    if not LANGDETECT_AVAILABLE:
        return "en"
    try:
        sample = " ".join(text.split()[:100])
        return detect(sample)
    except Exception:
        return "unknown"


# ═════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═════════════════════════════════════════════════════════════════════════════

def get_db_engine():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set in .env")
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def get_already_processed(conn) -> set[str]:
    rows = conn.execute(text("SELECT filename FROM legal_documents")).fetchall()
    return {r[0] for r in rows}


def upsert_document(conn, doc: dict) -> Optional[str]:
    doc_id = str(uuid.uuid4())
    conn.execute(
        text("""
            INSERT INTO legal_documents
                (id, filename, title, petitioner, respondent, court, state,
                 year, date_of_judgment, judges, citation, acts_referred,
                 bench_strength, case_type, page_count, full_text,
                 language, quality_flag, chunk_strategy, qdrant_synced)
            VALUES
                (:id, :filename, :title, :petitioner, :respondent, :court, :state,
                 :year, :date_of_judgment, :judges, :citation, :acts_referred,
                 :bench_strength, :case_type, :page_count, :full_text,
                 :language, :quality_flag, :chunk_strategy, false)
            ON CONFLICT (filename) DO UPDATE SET
                chunk_strategy = EXCLUDED.chunk_strategy,
                qdrant_synced  = false
        """),
        {
            "id":               doc_id,
            "filename":         doc["filename"],
            "title":            doc["title"],
            "petitioner":       doc["petitioner"],
            "respondent":       doc["respondent"],
            "court":            doc["court"],
            "state":            doc["state"],
            "year":             doc["year"],
            "date_of_judgment": doc["date_of_judgment"],
            "judges":           doc["judges"] or [],
            "citation":         doc["citation"],
            "acts_referred":    doc["acts_referred"] or [],
            "bench_strength":   doc["bench_strength"],
            "case_type":        doc["case_type"],
            "page_count":       doc["page_count"],
            "full_text":        doc["full_text"],
            "language":         doc["language"],
            "quality_flag":     doc["quality_flag"],
            "chunk_strategy":   doc["chunk_strategy"],
        },
    )
    row = conn.execute(
        text("SELECT id FROM legal_documents WHERE filename = :fn"),
        {"fn": doc["filename"]},
    ).fetchone()
    return str(row[0]) if row else None


def delete_chunks_for_document(conn, document_id: str):
    conn.execute(
        text("DELETE FROM legal_chunks WHERE document_id = :did::uuid"),
        {"did": document_id},
    )


def insert_chunks(conn, document_id: str, chunks: list[dict]):
    if not chunks:
        return
    rows = [
        {
            "id":             str(uuid.uuid4()),
            "document_id":    document_id,
            "chunk_index":    i,
            "chunk_text":     c["text"],
            "section_type":   c["section_type"],
            "chunk_strategy": c["chunk_strategy"],
            "page_number":    None,
        }
        for i, c in enumerate(chunks)
    ]
    conn.execute(
        text("""
            INSERT INTO legal_chunks
                (id, document_id, chunk_index, chunk_text, section_type, chunk_strategy, page_number)
            VALUES
                (:id, :document_id, :chunk_index, :chunk_text, :section_type, :chunk_strategy, :page_number)
        """),
        rows,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LexFind Metadata Extractor v2")
    parser.add_argument("--pdf-dir", default=str(PDF_DIR))
    parser.add_argument("--limit",   type=int, default=None, help="Process only N PDFs (test run)")
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help="Delete and re-create chunks for ALL documents with the new legal-aware strategy. "
             "Use after running migration 0004 to upgrade existing naive chunks.",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)

    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.limit:
        all_pdfs = all_pdfs[: args.limit]

    logger.info(f"PDF directory  : {pdf_dir}")
    logger.info(f"Total PDFs     : {len(all_pdfs)}")
    logger.info(f"Rechunk mode   : {args.rechunk}")

    engine = get_db_engine()

    stats = dict(
        total=len(all_pdfs), skipped=0, success=0, failed=0,
        scanned=0, non_english=0,
        has_title=0, has_court=0, has_year=0, has_judges=0, has_citation=0,
        used_legal_aware=0, used_sentence_aware=0,
    )

    with engine.begin() as conn:
        already_done = get_already_processed(conn)
    logger.info(f"Already in DB  : {len(already_done)} files")

    doc_batch: list[dict] = []

    def flush_batch(conn, batch: list[dict]):
        for doc in batch:
            try:
                doc_id = upsert_document(conn, doc)
                if doc_id:
                    if args.rechunk:
                        delete_chunks_for_document(conn, doc_id)
                    insert_chunks(conn, doc_id, doc.get("chunks", []))
            except Exception as e:
                logger.error(f"DB error for {doc['filename']}: {e}")
                stats["failed"] += 1

    for pdf_path in tqdm(all_pdfs, desc="Extracting", unit="pdf"):
        filename = pdf_path.name

        # Skip if already processed AND not in rechunk mode
        if filename in already_done and not args.rechunk:
            stats["skipped"] += 1
            continue

        try:
            full_text, page_count = extract_text(pdf_path)

            quality_flag = "clean"
            if len(full_text.strip()) < 200:
                quality_flag = "scanned"
                stats["scanned"] += 1

            language = detect_language(full_text)
            if language != "en" and quality_flag == "clean":
                quality_flag = "non_english"
                stats["non_english"] += 1

            meta = extract_metadata(full_text, filename)

            if meta["title"]:     stats["has_title"]    += 1
            if meta["court"]:     stats["has_court"]    += 1
            if meta["year"]:      stats["has_year"]     += 1
            if meta["judges"]:    stats["has_judges"]   += 1
            if meta["citation"]:  stats["has_citation"] += 1

            # Chunk only clean/encoding_fixed docs
            chunks: list[dict] = []
            chunk_strategy = "none"
            if quality_flag in ("clean", "encoding_fixed"):
                chunks, chunk_strategy = chunk_document(full_text)
                if chunk_strategy == "legal_aware":
                    stats["used_legal_aware"] += 1
                else:
                    stats["used_sentence_aware"] += 1

            doc_batch.append({
                "filename":       filename,
                "title":          meta["title"],
                "petitioner":     meta["petitioner"],
                "respondent":     meta["respondent"],
                "court":          meta["court"],
                "state":          meta["state"],
                "year":           meta["year"],
                "date_of_judgment": meta["date_of_judgment"],
                "judges":         meta["judges"],
                "citation":       meta["citation"],
                "acts_referred":  meta["acts_referred"],
                "bench_strength": meta["bench_strength"],
                "case_type":      meta["case_type"],
                "page_count":     page_count,
                "full_text":      full_text if quality_flag != "scanned" else None,
                "language":       language,
                "quality_flag":   quality_flag,
                "chunk_strategy": chunk_strategy,
                "chunks":         chunks,
            })
            stats["success"] += 1

        except Exception as e:
            logger.error(f"FAILED: {filename} — {e}", exc_info=False)
            stats["failed"] += 1
            continue

        if len(doc_batch) >= BATCH_SIZE:
            with engine.begin() as txn:
                flush_batch(txn, doc_batch)
            doc_batch.clear()

    if doc_batch:
        with engine.begin() as txn:
            flush_batch(txn, doc_batch)

    # ── Summary ───────────────────────────────────────────────────────────────
    processed = stats["success"] + stats["failed"]
    def pct(n, d): return f"{n/d*100:.1f}%" if d > 0 else "0.0%"

    print("\n" + "=" * 60)
    print("  LexFind — Metadata Extraction v2 Complete")
    print("=" * 60)
    print(f"  Total PDFs         : {stats['total']}")
    print(f"  Skipped            : {stats['skipped']} (already in DB)")
    print(f"  Processed          : {processed}")
    print(f"  Success            : {stats['success']}")
    print(f"  Failed             : {stats['failed']}")
    print(f"  Scanned            : {stats['scanned']}")
    print(f"  Non-English        : {stats['non_english']}")
    print()
    print(f"  Chunking Strategy:")
    print(f"    Legal-Aware      : {stats['used_legal_aware']}")
    print(f"    Sentence-Aware   : {stats['used_sentence_aware']}")
    if stats["success"] > 0:
        s = stats["success"]
        print()
        print("  Metadata Extraction Rates (of successful):")
        print(f"    Title    : {stats['has_title']}/{s} = {pct(stats['has_title'], s)}")
        print(f"    Court    : {stats['has_court']}/{s} = {pct(stats['has_court'], s)}")
        print(f"    Year     : {stats['has_year']}/{s} = {pct(stats['has_year'], s)}")
        print(f"    Judges   : {stats['has_judges']}/{s} = {pct(stats['has_judges'], s)}")
        print(f"    Citation : {stats['has_citation']}/{s} = {pct(stats['has_citation'], s)}")
    print(f"\n  Error log: {LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
