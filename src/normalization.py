"""
Unicode-Safe Text Normalization Module.
"""

import unicodedata
import re
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Punctuation pattern that replaces noise symbols with space, preserving letters, marks, digits in any script
# \w matches unicode word characters in Python 3 re when default flags apply.
PUNCT_PATTERN = re.compile(r"[^\w\s]")
MULTIPLE_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Unicode-safe normalization:
    1. Unicode NFKC normalization
    2. Case folding (Unicode-aware lowercasing)
    3. Safe punctuation removal / replacement with space
    4. Collapse whitespace
    """
    if not text:
        return ""
    
    # 1. NFKC Normalization
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Case folding
    text = text.casefold()
    
    # 3. Clean noise symbols but preserve letters/digits across all scripts
    text = PUNCT_PATTERN.sub(" ", text)
    
    # 4. Collapse spaces
    text = MULTIPLE_SPACES.sub(" ", text).strip()
    
    return text


def create_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds normalized fields to DataFrame without overwriting original columns:
    - name_normalized
    - address_normalized
    - combined_normalized
    """
    df = df.copy()
    
    logger.info("Applying Unicode-safe normalization to names and addresses...")
    df["name_normalized"] = df["business_name"].apply(normalize_text)
    df["address_normalized"] = df["business_address"].apply(normalize_text)
    
    # Combined representation
    df["combined_normalized"] = (df["name_normalized"] + " " + df["address_normalized"]).str.strip()
    
    return df
