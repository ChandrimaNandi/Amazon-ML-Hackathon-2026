"""
Unicode-Safe Multilingual Text Normalization & Transliteration Module.
Preserves Indic vowels (Devanagari to Latin), decomposes European accents, and standardizes business legal forms.
"""

import unicodedata
import re
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("PrecisionSieve.Normalization")

ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\x00-\x1f\x7f-\x9f]")
MULTIPLE_SPACES = re.compile(r"\s+")
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
    if not text or not isinstance(text, str):
        return ""
    text = ZERO_WIDTH_CHARS.sub("", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = PUNCT_PATTERN.sub(" ", text)
    return MULTIPLE_SPACES.sub(" ", text).strip()


def transliterate_to_latin(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    if text.isascii():
        return text
    chars = [DEVANAGARI_TO_LATIN.get(ch, ch) if '\u0900' <= ch <= '\u097f' else ch for ch in text]
    res = "".join(chars)
    nfd = unicodedata.normalize('NFD', res)
    return "".join(c for c in nfd if unicodedata.category(c) != 'Mn')


def clean_entity_field(text: str, expand_business: bool = True) -> str:
    norm = normalize_text(text)
    trans = transliterate_to_latin(norm)
    if expand_business:
        for pat, rep in BUSINESS_EXPANSIONS:
            trans = pat.sub(rep, trans)
    return MULTIPLE_SPACES.sub(" ", trans).strip()


def create_normalized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
        
    df["name_norm"] = df["business_name"].apply(clean_entity_field)
    df["addr_norm"] = df["business_address"].apply(clean_entity_field)
    df["country_clean"] = df["country"].str.strip().str.upper()
    df["comb_norm"] = (df["name_norm"] + " " + df["addr_norm"]).str.strip()
    return df
