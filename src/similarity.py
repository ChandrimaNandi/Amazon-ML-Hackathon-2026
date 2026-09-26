"""
String Distance and Text Similarity Module for Business Entity Resolution.
Uses RapidFuzz for high-speed C++ string metric evaluations.
"""

import numpy as np
from rapidfuzz import distance, fuzz
from typing import Set, Tuple


def compute_token_metrics(str1: str, str2: str) -> Tuple[float, int, int]:
    """
    Computes token-level Jaccard similarity, token overlap count, and token count difference.
    Returns: (jaccard_sim, overlap_count, token_count_diff)
    """
    if not str1 or not str2:
        tokens1 = str1.split() if str1 else []
        tokens2 = str2.split() if str2 else []
        return (0.0, 0, abs(len(tokens1) - len(tokens2)))
        
    set1 = set(str1.split())
    set2 = set(str2.split())
    
    if not set1 or not set2:
        return (0.0, 0, abs(len(set1) - len(set2)))
        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard = intersection / float(union) if union > 0 else 0.0
    token_diff = abs(len(set1) - len(set2))
    
    return (jaccard, intersection, token_diff)


def compute_string_similarities(str1: str, str2: str) -> Tuple[float, float, float, float, float, float, int, float, int, int]:
    """
    Computes pairwise string similarity metrics:
    Returns:
    - lev_ratio: Levenshtein normalized similarity [0, 1]
    - jw_sim: Jaro-Winkler similarity [0, 1]
    - ratio: RapidFuzz simple ratio [0, 1]
    - partial_ratio: RapidFuzz partial ratio [0, 1]
    - token_sort: Token sort ratio [0, 1]
    - token_set: Token set ratio [0, 1]
    - char_len_diff: Absolute character length difference
    - char_len_ratio: Length ratio min(l1, l2) / max(l1, l2)
    - token_overlap: Number of shared tokens
    - token_diff: Difference in number of tokens
    """
    if not str1 or not str2:
        l1 = len(str1) if str1 else 0
        l2 = len(str2) if str2 else 0
        diff = abs(l1 - l2)
        ratio = 0.0 if (l1 == 0 and l2 == 0) else min(l1, l2) / float(max(l1, l2, 1))
        t1 = len(str1.split()) if str1 else 0
        t2 = len(str2.split()) if str2 else 0
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, diff, ratio, 0, abs(t1 - t2))
        
    if str1 == str2:
        tokens = set(str1.split())
        return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 1.0, len(tokens), 0)

    l1, l2 = len(str1), len(str2)
    diff = abs(l1 - l2)
    len_ratio = min(l1, l2) / float(max(l1, l2, 1))
    
    lev_ratio = distance.Levenshtein.normalized_similarity(str1, str2)
    jw_sim = distance.JaroWinkler.similarity(str1, str2)
    ratio = fuzz.ratio(str1, str2) / 100.0
    partial_ratio = fuzz.partial_ratio(str1, str2) / 100.0
    token_sort = fuzz.token_sort_ratio(str1, str2) / 100.0
    token_set = fuzz.token_set_ratio(str1, str2) / 100.0
    
    jaccard, overlap, token_diff = compute_token_metrics(str1, str2)
    
    return (
        lev_ratio, jw_sim, ratio, partial_ratio, token_sort, token_set,
        diff, len_ratio, overlap, token_diff
    )
