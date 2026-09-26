"""
Unicode-Safe Multilingual Text Normalization & Transliteration Module for ColBERT-Ditto ER.
Handles Indic scripts (Devanagari to Latin), French/European diacritics, and Unicode NFKC normalization.
"""

import unicodedata
import re
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("ColBERT_Ditto.Normalization")

ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\x00-\x1f\x7f-\x9f]")
MULTIPLE_SPACES = re.compile(r"\s+")
# Preserves alphanumeric characters and Indic non-spacing marks (Mn/Mc)
PUNCT_PATTERN = re.compile(r"[^\w\s\u0900-\u097F\u0980-\u09FF\u0600-\u06FF\u0400-\u04FF\u4E00-\u9FFF]")

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

BUSINESS_EXPANSIONS = [
    (re.compile(r"\bpvt\s+ltd\b", re.I), "private limited"),
    (re.compile(r"\bpvt\b", re.I), "private"),
    (re.compile(r"\bltd\b", re.I), "limited"),
    (re.compile(r"\binc\b", re.I), "incorporated"),
    (re.compile(r"\bco\b", re.I), "company"),
    (re.compile(r"\bcorp\b", re.I), "corporation"),
    (re.compile(r"\bllc\b", re.I), "limited liability company"),
    (re.compile(r"\bllp\b", re.I), "limited liability partnership"),
    (re.compile(r"\bste\b", re.I), "suite"),
    (re.compile(r"\bapt\b", re.I), "apartment"),
    (re.compile(r"\brd\b", re.I), "road"),
    (re.compile(r"\bst\b", re.I), "street"),
    (re.compile(r"\bave\b", re.I), "avenue"),
    (re.compile(r"\bblvd\b", re.I), "boulevard"),
    (re.compile(r"\bdr\b", re.I), "drive"),
]


def normalize_text(text: str) -> str:
    """
    Standard Unicode NFKC normalization:
    - Removes zero-width/control characters
    - NFKC decomposition
    - Unicode casefold
    - Replaces punctuation with space
    - Collapses whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = ZERO_WIDTH_CHARS.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = PUNCT_PATTERN.sub(" ", text)
    text = MULTIPLE_SPACES.sub(" ", text).strip()
    return text


def transliterate_to_latin(text: str) -> str:
    """
    Phonetic transliteration to Latin representation:
    - Devanagari transliteration via char map
    - Latin accent decomposition (é -> e, ç -> c)
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Fast path for pure ASCII
    if text.isascii():
        return text
    
    # 1. Transliterate Devanagari if present
    chars = []
    has_devanagari = False
    for ch in text:
        if '\u0900' <= ch <= '\u097f':
            has_devanagari = True
            chars.append(DEVANAGARI_TO_LATIN.get(ch, ''))
        else:
            chars.append(ch)
            
    res = "".join(chars)
    
    # 2. Decompose European accents (NFD decomposition + drop nonspacing marks)
    nfd = unicodedata.normalize('NFD', res)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != 'Mn')
    return stripped


def clean_entity_field(text: str, expand_business: bool = True) -> str:
    """Full normalization and expansion pipeline for a single text field."""
    norm = normalize_text(text)
    trans = transliterate_to_latin(norm)
    if expand_business:
        for pattern, repl in BUSINESS_EXPANSIONS:
            trans = pattern.sub(repl, trans)
    return MULTIPLE_SPACES.sub(" ", trans).strip()


def create_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds normalized and transliterated fields to an entity DataFrame."""
    df = df.copy()
    for col in ["business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
        
    df["name_norm"] = df["business_name"].apply(clean_entity_field)
    df["addr_norm"] = df["business_address"].apply(clean_entity_field)
    df["country_clean"] = df["country"].str.strip().str.upper()
    return df
