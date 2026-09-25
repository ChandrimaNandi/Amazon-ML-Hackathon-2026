"""
Unicode-Safe Multilingual Text Normalization & Transliteration Module.

Provides robust normalization while preserving original representations:
1. Unicode NFKC Normalization & Zero-width/Control Character Cleaning
2. Unicode Casefolding (case-insensitive across all scripts)
3. Whitespace & Punctuation Normalization
4. Phonetic Script Transliteration (Devanagari -> Latin & Latin accent stripping)
5. Business entity suffix standardization (Pvt Ltd, LLC, Inc, Corp, LLP)
"""

import unicodedata
import re
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Zero-width and control characters to strip
ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\x00-\x1f\x7f-\x9f]")
# Whitespace pattern
MULTIPLE_SPACES = re.compile(r"\s+")
# Safe punctuation replacement (preserves unicode word chars, marks, and numbers in all scripts)
# Crucial: \w alone does NOT include Devanagari/Indic nonspacing vowel marks (Mn/Mc), so we explicitly include them.
PUNCT_PATTERN = re.compile(r"[^\w\s\u0900-\u097F\u0980-\u09FF\u0600-\u06FF\u0400-\u04FF\u4E00-\u9FFF]")

# Devanagari to Latin phonetic transliteration map
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

# Common business suffixes and abbreviations
BUSINESS_EXPANSIONS = [
    (re.compile(r"\bpvt\s+ltd\b"), "private limited"),
    (re.compile(r"\bpvt\b"), "private"),
    (re.compile(r"\bltd\b"), "limited"),
    (re.compile(r"\binc\b"), "incorporated"),
    (re.compile(r"\bco\b"), "company"),
    (re.compile(r"\bcorp\b"), "corporation"),
    (re.compile(r"\bllc\b"), "limited liability company"),
    (re.compile(r"\bllp\b"), "limited liability partnership"),
    (re.compile(r"\bप्रा\s*लि\b"), "private limited"),
    (re.compile(r"\bलिमिटेड\b"), "limited"),
    (re.compile(r"\bप्राइवेट\b"), "private"),
    (re.compile(r"\bकंपनी\b"), "company"),
    (re.compile(r"\bएलएलपी\b"), "llp"),
]


def normalize_text(text: str) -> str:
    """
    Standard Unicode NFKC normalization:
    - Removes zero-width/control characters
    - Decomposes and recomposes compatibility characters via NFKC
    - Casefolds
    - Replaces punctuation with space while preserving characters across all scripts
    - Collapses whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Clean zero-width and control characters
    text = ZERO_WIDTH_CHARS.sub("", text)
    
    # 2. Unicode NFKC Normalization
    text = unicodedata.normalize("NFKC", text)
    
    # 3. Unicode-aware case folding
    text = text.casefold()
    
    # 4. Safe punctuation removal (preserves unicode word characters in any script)
    text = PUNCT_PATTERN.sub(" ", text)
    
    # 5. Collapse spaces
    text = MULTIPLE_SPACES.sub(" ", text).strip()
    
    return text


def transliterate_to_latin(text: str) -> str:
    """
    Phonetic transliteration to Latin representation:
    - Transliterates Devanagari characters phonetically
    - Decomposes Latin accents (e.g. é -> e, ç -> c for French/European text)
    - Fast ASCII short-circuit for high throughput on million-record datasets.
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = normalize_text(text)
    if text.isascii():
        return text
    
    # Transliterate Devanagari phonetically if Devanagari characters are present
    has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text)
    if has_devanagari:
        chars = [DEVANAGARI_TO_LATIN.get(c, c) for c in text]
        text = "".join(chars)
    
    # Strip Latin diacritics / accents (NFKD decompose and strip non-ASCII combining marks)
    if not text.isascii():
        nfkd = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in nfkd if not unicodedata.combining(c))
        
    return MULTIPLE_SPACES.sub(" ", text).strip()


def create_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds normalized and transliterated fields to DataFrame without overwriting original columns:
    - name_normalized: NFKC + casefolded original script
    - address_normalized: NFKC + casefolded original script
    - name_transliterated: Latin phonetic representation
    - address_transliterated: Latin phonetic representation
    - combined_normalized: name_normalized + ' ' + address_normalized
    - combined_transliterated: name_transliterated + ' ' + address_transliterated
    """
    df = df.copy()
    
    # Ensure string types
    name_col = df["business_name"].fillna("").astype(str)
    addr_col = df["business_address"].fillna("").astype(str)
    
    # 1. Native script Unicode NFKC normalization
    df["name_normalized"] = name_col.map(normalize_text)
    df["address_normalized"] = addr_col.map(normalize_text)
    
    # 2. Transliterated / accent-stripped representation
    df["name_transliterated"] = name_col.map(transliterate_to_latin)
    df["address_transliterated"] = addr_col.map(transliterate_to_latin)
    
    # 3. Combined representations
    df["combined_normalized"] = (df["name_normalized"] + " " + df["address_normalized"]).str.strip()
    df["combined_transliterated"] = (df["name_transliterated"] + " " + df["address_transliterated"]).str.strip()
    
    return df
