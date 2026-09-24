"""
Dataset Profiling & Multilingual/Script Analysis Module.
"""

import unicodedata
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_script(text: str) -> str:
    """
    Classifies a string into Unicode script category based on non-ASCII characters or primary block.
    Returns one of: 'Latin', 'Devanagari', 'Arabic', 'Cyrillic', 'CJK', 'Greek', 'Hebrew', 'Other', 'Mixed'.
    """
    if not text:
        return "Empty"
    
    script_counts = {
        "Latin": 0,
        "Devanagari": 0,
        "Arabic": 0,
        "Cyrillic": 0,
        "CJK": 0,
        "Greek": 0,
        "Hebrew": 0,
        "Other": 0
    }
    
    for char in text:
        if not char.isalpha():
            continue
        code = ord(char)
        if (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A) or (0x00C0 <= code <= 0x024F) or (0x1E00 <= code <= 0x1EFF):
            script_counts["Latin"] += 1
        elif 0x0900 <= code <= 0x097F:
            script_counts["Devanagari"] += 1
        elif (0x0600 <= code <= 0x06FF) or (0x0750 <= code <= 0x077F) or (0x08A0 <= code <= 0x08FF):
            script_counts["Arabic"] += 1
        elif (0x0400 <= code <= 0x04FF) or (0x0500 <= code <= 0x052F):
            script_counts["Cyrillic"] += 1
        elif (0x4E00 <= code <= 0x9FFF) or (0x3040 <= code <= 0x30FF) or (0xAC00 <= code <= 0xD7AF):
            script_counts["CJK"] += 1
        elif 0x0370 <= code <= 0x03FF:
            script_counts["Greek"] += 1
        elif 0x0590 <= code <= 0x05FF:
            script_counts["Hebrew"] += 1
        else:
            script_counts["Other"] += 1
            
    total_alpha = sum(script_counts.values())
    if total_alpha == 0:
        return "Numeric/Symbol"
    
    # Check dominant script
    active_scripts = [s for s, cnt in script_counts.items() if cnt > 0]
    if len(active_scripts) == 1:
        return active_scripts[0]
    
    primary = max(script_counts, key=script_counts.get)
    if script_counts[primary] / total_alpha >= 0.7:
        return primary
    return "Mixed"


def profile_dataframe(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """
    Profiles basic statistics for a source DataFrame.
    """
    num_rows = len(df)
    unique_ids = df["entity_id"].nunique()
    missing_names = (df["business_name"] == "").sum()
    missing_addresses = (df["business_address"] == "").sum()
    missing_countries = (df["country"] == "").sum()
    
    country_counts = df["country"].value_counts().to_dict()
    
    # Script breakdown for business_name
    name_scripts = df["business_name"].apply(detect_script).value_counts().to_dict()
    addr_scripts = df["business_address"].apply(detect_script).value_counts().to_dict()
    
    profile = {
        "dataset": name,
        "num_rows": num_rows,
        "unique_ids": unique_ids,
        "missing_name_count": missing_names,
        "missing_name_pct": round(missing_names / max(num_rows, 1) * 100, 2),
        "missing_address_count": missing_addresses,
        "missing_address_pct": round(missing_addresses / max(num_rows, 1) * 100, 2),
        "missing_country_count": missing_countries,
        "missing_country_pct": round(missing_countries / max(num_rows, 1) * 100, 2),
        "countries": country_counts,
        "name_scripts": name_scripts,
        "address_scripts": addr_scripts,
    }
    
    return profile


def profile_ground_truth(gt_df: pd.DataFrame, s1_to_matches: Dict[str, Set[str]]) -> Dict[str, Any]:
    """
    Profiles ground truth statistics: match counts per S1 entity.
    """
    total_s1 = len(gt_df)
    match_counts = [len(m) for m in s1_to_matches.values()]
    
    zero_matches = sum(1 for c in match_counts if c == 0)
    one_match = sum(1 for c in match_counts if c == 1)
    multi_matches = sum(1 for c in match_counts if c > 1)
    
    gt_stats = {
        "total_s1_entities": total_s1,
        "zero_match_count": zero_matches,
        "zero_match_pct": round(zero_matches / max(total_s1, 1) * 100, 2),
        "one_match_count": one_match,
        "one_match_pct": round(one_match / max(total_s1, 1) * 100, 2),
        "multi_match_count": multi_matches,
        "multi_match_pct": round(multi_matches / max(total_s1, 1) * 100, 2),
        "avg_matches_per_s1": round(float(np.mean(match_counts)), 3),
        "max_matches_per_s1": max(match_counts) if match_counts else 0,
    }
    
    return gt_stats
