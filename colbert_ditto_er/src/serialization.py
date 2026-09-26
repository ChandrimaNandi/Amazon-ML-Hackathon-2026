"""
Ditto-Style Entity Record Serialization Module.
Converts tabular business records into structured token sequences for Transformer and ColBERT models.
Reference: "Deep Entity Matching with Pre-Trained Language Models" (Li et al., EMNLP 2020).
"""

import pandas as pd
from typing import List, Dict, Any, Union
from colbert_ditto_er.src.normalization import clean_entity_field


def serialize_record(
    name: str,
    address: str = "",
    country: str = "",
    tag_style: str = "compact"
) -> str:
    """
    Serializes a business entity record into a structured sequence.
    
    Styles:
    - 'compact': "[NAME] {name} [ADDR] {address} [CTRY] {country}"
      (Saves token budget for long street addresses and legal names)
    - 'ditto': "[COL] name [VAL] {name} [COL] address [VAL] {address} [COL] country [VAL] {country}"
    """
    name_clean = clean_entity_field(name)
    addr_clean = clean_entity_field(address)
    ctry_clean = str(country).strip().upper() if country else ""
    
    if tag_style == "ditto":
        parts = []
        if name_clean:
            parts.extend(["[COL]", "name", "[VAL]", name_clean])
        if addr_clean:
            parts.extend(["[COL]", "address", "[VAL]", addr_clean])
        if ctry_clean:
            parts.extend(["[COL]", "country", "[VAL]", ctry_clean])
        return " ".join(parts)
    else:  # compact
        parts = []
        if name_clean:
            parts.append(f"[NAME] {name_clean}")
        if addr_clean:
            parts.append(f"[ADDR] {addr_clean}")
        if ctry_clean:
            parts.append(f"[CTRY] {ctry_clean}")
        return " ".join(parts) if parts else "[NAME] unknown"


def serialize_dataframe(df: pd.DataFrame, tag_style: str = "compact") -> List[str]:
    """Serializes an entire DataFrame of entity records into a list of strings."""
    names = df["business_name"].fillna("").astype(str).tolist() if "business_name" in df.columns else [""] * len(df)
    addresses = df["business_address"].fillna("").astype(str).tolist() if "business_address" in df.columns else [""] * len(df)
    countries = df["country"].fillna("").astype(str).tolist() if "country" in df.columns else [""] * len(df)
    
    return [
        serialize_record(n, a, c, tag_style=tag_style)
        for n, a, c in zip(names, addresses, countries)
    ]
