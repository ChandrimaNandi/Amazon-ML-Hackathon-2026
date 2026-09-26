"""
Unicode-Safe Multilingual Text Normalization & Transliteration Module.
Supports English, Hindi (Devanagari transliteration), and French (open-set diacritic normalization).
"""

import unicodedata
import re
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("FastHybridER.Normalization")

# Zero-width and control characters to strip
ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\x00-\x1f\x7f-\x9f]")
# Whitespace pattern
MULTIPLE_SPACES = re.compile(r"\s+")
# Safe punctuation replacement: preserves unicode letters, marks (Devanagari vowel signs), and numbers
PUNCT_PATTERN = re.compile(r"[^\w\s\u0900-\u097F\u0980-\u09FF\u0600-\u06FF\u0400-\u04FF\u4E00-\u9FFF]")

# Devanagari to Latin phonetic transliteration dictionary
DEVANAGARI_TO_LATIN = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'an', 'अः': 'ah',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': '', 'ं': 'n', 'ँ': 'n', 'ः': 'h',
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# Legal suffixes & abbreviations across US, India, France
EXPANSIONS = [
    (re.compile(r"\bpvt\s+ltd\b", re.I), "private limited"),
    (re.compile(r"\bpvt\b", re.I), "private"),
    (re.compile(r"\bltd\b", re.I), "limited"),
    (re.compile(r"\binc\b", re.I), "incorporated"),
    (re.compile(r"\bcorp\b", re.I), "corporation"),
    (re.compile(r"\bllc\b", re.I), "limited liability company"),
    (re.compile(r"\bsarl\b", re.I), "societe a responsabilite limitee"),
    (re.compile(r"\bsas\b", re.I), "societe par actions simplifiee"),
    (re.compile(r"\bsa\b", re.I), "societe anonyme"),
    (re.compile(r"\brd\b", re.I), "road"),
    (re.compile(r"\bst\b", re.I), "street"),
    (re.compile(r"\bave\b", re.I), "avenue"),
    (re.compile(r"\bbvd?\b", re.I), "boulevard"),
]


def normalize_text(text: str) -> str:
    """
    Standard Unicode NFKC normalization:
    - Removes zero-width/control characters
    - NFKC decomposition/recomposition
    - Casefolding
    - Replaces punctuation with space while preserving characters across all scripts
    - Collapses whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    text = ZERO_WIDTH_CHARS.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = PUNCT_PATTERN.sub(" ", text)
    return MULTIPLE_SPACES.sub(" ", text).strip()


def transliterate_to_latin(text: str) -> str:
    """
    Transliterates Devanagari to Latin and strips French/Latin accents:
    - é, è, ê, ç, à, ô -> e, e, e, c, a, o
    - Fast ASCII short-circuit for high throughput
    """
    if not text or not isinstance(text, str):
        return ""

    text = normalize_text(text)
    if text.isascii():
        return text

    # Transliterate Devanagari if present
    if any(0x0900 <= ord(c) <= 0x097F for c in text):
        chars = [DEVANAGARI_TO_LATIN.get(c, c) for c in text]
        text = "".join(chars)

    # Decompose French / European accents and drop combining marks
    if not text.isascii():
        nfkd = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in nfkd if not unicodedata.combining(c))

    return MULTIPLE_SPACES.sub(" ", text).strip()


def create_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds normalized and transliterated fields to DataFrame:
    - name_normalized: NFKC + casefolded original script
    - address_normalized: NFKC + casefolded original script
    - name_transliterated: Latin phonetic representation
    - address_transliterated: Latin phonetic representation
    - combined_normalized: name_normalized + ' ' + address_normalized
    - combined_transliterated: name_transliterated + ' ' + address_transliterated
    """
    df = df.copy()
    name_col = df["business_name"].fillna("").astype(str)
    addr_col = df["business_address"].fillna("").astype(str)

    df["name_normalized"] = name_col.map(normalize_text)
    df["address_normalized"] = addr_col.map(normalize_text)
    df["name_transliterated"] = name_col.map(transliterate_to_latin)
    df["address_transliterated"] = addr_col.map(transliterate_to_latin)

    df["combined_normalized"] = (df["name_normalized"] + " " + df["address_normalized"]).str.strip()
    df["combined_transliterated"] = (df["name_transliterated"] + " " + df["address_transliterated"]).str.strip()
    return df
