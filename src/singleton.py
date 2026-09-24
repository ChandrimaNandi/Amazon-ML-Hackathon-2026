"""
Singleton Entity & No-Match Analysis Module.
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def analyze_singleton_performance(
    all_s1_ids: Set[str],
    s1_to_true_matches: Dict[str, Set[str]],
    s1_to_pred_matches: Dict[str, Set[str]]
) -> Dict[str, Any]:
    """
    Analyzes matching quality specifically on singleton (zero true matches), one match, and multi match entities.
    """
    zero_match_s1 = {s1 for s1 in all_s1_ids if len(s1_to_true_matches.get(s1, set())) == 0}
    one_match_s1 = {s1 for s1 in all_s1_ids if len(s1_to_true_matches.get(s1, set())) == 1}
    multi_match_s1 = {s1 for s1 in all_s1_ids if len(s1_to_true_matches.get(s1, set())) > 1}
    
    # Zero match performance (singleton rejection accuracy)
    zero_match_correct = sum(1 for s1 in zero_match_s1 if len(s1_to_pred_matches.get(s1, set())) == 0)
    zero_match_fp = len(zero_match_s1) - zero_match_correct
    
    # One match performance
    one_match_correct = sum(1 for s1 in one_match_s1 if s1_to_pred_matches.get(s1, set()) == s1_to_true_matches[s1])
    
    analysis = {
        "total_zero_match_entities": len(zero_match_s1),
        "zero_match_correctly_empty": zero_match_correct,
        "zero_match_false_positives": zero_match_fp,
        "zero_match_accuracy_pct": round(zero_match_correct / max(len(zero_match_s1), 1) * 100, 2),
        "total_one_match_entities": len(one_match_s1),
        "one_match_exact_correct": one_match_correct,
        "one_match_accuracy_pct": round(one_match_correct / max(len(one_match_s1), 1) * 100, 2),
        "total_multi_match_entities": len(multi_match_s1),
    }
    
    return analysis
