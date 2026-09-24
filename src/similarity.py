"""
String Distance and Text Similarity Helper Functions.
"""

import numpy as np
from rapidfuzz import distance, fuzz
from typing import Set, Tuple


def compute_token_jaccard(str1: str, str2: str) -> float:
    """Computes Jaccard similarity of words in str1 and str2."""
    if not str1 or not str2:
        return 0.0
    set1 = set(str1.split())
    set2 = set(str2.split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / float(union)


def compute_string_similarities(str1: str, str2: str) -> Tuple[float, float, float, float, float, int, float]:
    """
    Computes rapid similarity metrics between str1 and str2:
    Returns (levenshtein_ratio, jaro_winkler, token_set_ratio, token_sort_ratio, token_jaccard, length_diff, length_ratio)
    """
    if not str1 or not str2:
        l1, l2 = len(str1), len(str2)
        diff = abs(l1 - l2)
        ratio = 0.0 if (l1 == 0 and l2 == 0) else min(l1, l2) / max(l1, l2, 1)
        return (0.0, 0.0, 0.0, 0.0, 0.0, diff, ratio)
    
    lev_ratio = distance.Levenshtein.normalized_similarity(str1, str2)
    jw_sim = distance.JaroWinkler.similarity(str1, str2)
    ts_ratio = fuzz.token_set_ratio(str1, str2) / 100.0
    tsort_ratio = fuzz.token_sort_ratio(str1, str2) / 100.0
    jaccard = compute_token_jaccard(str1, str2)
    
    l1, l2 = len(str1), len(str2)
    diff = abs(l1 - l2)
    ratio = min(l1, l2) / max(l1, l2, 1)
    
    return (lev_ratio, jw_sim, ts_ratio, tsort_ratio, jaccard, diff, ratio)
