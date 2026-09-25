"""
Evaluation Module for Business Entity Resolution.
Computes Macro F0.5 metric, Entity-level Precision & Recall, and Candidate Recall@K.
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple, List, Any, Optional
import logging

from src.config import BETA

logger = logging.getLogger(__name__)


def compute_entity_f_beta(
    true_set: Set[str],
    pred_set: Set[str],
    beta: float = BETA
) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F_beta for a single S1 entity.
    Special cases:
    - true_set is empty and pred_set is empty: (1.0, 1.0, 1.0)
    - true_set is empty and pred_set is non-empty: (0.0, 1.0, 0.0)
    - true_set is non-empty and pred_set is empty: (1.0, 0.0, 0.0)
    """
    beta_sq = beta ** 2
    
    if not true_set and not pred_set:
        return (1.0, 1.0, 1.0)
    if not true_set and pred_set:
        return (0.0, 1.0, 0.0)
    if true_set and not pred_set:
        return (1.0, 0.0, 0.0)
        
    tp = len(true_set.intersection(pred_set))
    prec = tp / float(len(pred_set))
    rec = tp / float(len(true_set))
    
    if (beta_sq * prec + rec) == 0:
        f_beta = 0.0
    else:
        f_beta = (1.0 + beta_sq) * prec * rec / ((beta_sq * prec) + rec)
        
    return (prec, rec, f_beta)


def evaluate_macro_metrics(
    all_s1_ids: Set[str],
    s1_to_true_matches: Dict[str, Set[str]],
    s1_to_pred_matches: Dict[str, Set[str]],
    beta: float = BETA
) -> Dict[str, float]:
    """
    Computes Macro F0.5, Macro Precision, Macro Recall across evaluated S1 entities.
    """
    precisions, recalls, f_betas = [], [], []
    
    for s1_id in all_s1_ids:
        true_set = s1_to_true_matches.get(s1_id, set())
        pred_set = s1_to_pred_matches.get(s1_id, set())
        
        prec, rec, f_b = compute_entity_f_beta(true_set, pred_set, beta=beta)
        precisions.append(prec)
        recalls.append(rec)
        f_betas.append(f_b)
        
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    macro_f_beta = float(np.mean(f_betas)) if f_betas else 0.0
    
    results = {
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        f"macro_f{beta}": round(macro_f_beta, 4),
    }
    
    logger.info(f"Evaluation Results -> Macro Precision: {macro_prec:.4f}, Macro Recall: {macro_rec:.4f}, Macro F{beta}: {macro_f_beta:.4f}")
    return results


def evaluate_candidate_recall(
    cand_df: pd.DataFrame,
    s1_to_true_matches: Dict[str, Set[str]],
    evaluated_query_ids: Optional[Set[str]] = None
) -> Dict[str, float]:
    """
    Evaluates candidate recall on training data.
    Measures proportion of true match pairs (s1_id, query_id) present in candidate set.
    """
    if cand_df.empty:
        return {"candidate_recall": 0.0, "total_true_pairs": 0, "found_pairs": 0}
        
    if evaluated_query_ids is None:
        evaluated_query_ids = set(cand_df["query_id"].unique())
        
    # Extract true pairs belonging to the evaluated query IDs
    true_pair_set = set()
    for s1_id, q_set in s1_to_true_matches.items():
        for q_id in q_set:
            if q_id in evaluated_query_ids:
                true_pair_set.add((s1_id, q_id))
                
    total_true_pairs = len(true_pair_set)
    if total_true_pairs == 0:
        return {"candidate_recall": 1.0, "total_true_pairs": 0, "found_pairs": 0}
            
    # Set of candidate (s1_id, query_id) pairs
    cand_pair_set = set(zip(cand_df["s1_id"], cand_df["query_id"]))
    
    found_pairs = len(true_pair_set.intersection(cand_pair_set))
    recall = found_pairs / float(total_true_pairs)
    
    results = {
        "candidate_recall": round(recall, 4),
        "total_true_pairs": total_true_pairs,
        "found_pairs": found_pairs,
        "missed_pairs": total_true_pairs - found_pairs,
    }
    
    logger.info(f"Candidate Recall: {recall:.4f} ({found_pairs:,} / {total_true_pairs:,} true pairs recovered for {len(evaluated_query_ids):,} queries)")
    return results
