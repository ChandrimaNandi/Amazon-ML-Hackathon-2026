"""
Constrained Bipartite Assignment, Query Exclusivity & Macro F0.5 Optimization Module.
Guarantees query exclusivity (each query -> at most 1 S1) and optimizes precision on singletons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

from fast_hybrid_er.configs.default_config import BETA

logger = logging.getLogger("FastHybridER.Assignment")


def compute_entity_f05(
    true_matches: Set[str],
    pred_matches: Set[str],
    beta: float = BETA
) -> float:
    """
    Computes F_beta (default beta=0.5) for a single reference S1 entity.
    Singletons:
      - If true is empty and pred is empty -> 1.0
      - If true is empty and pred is non-empty -> 0.0
    Non-singletons:
      - Precision = |true & pred| / |pred|
      - Recall = |true & pred| / |true|
      - F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if len(true_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0

    if len(pred_matches) == 0:
        return 0.0

    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0

    prec = tp / len(pred_matches)
    rec = tp / len(true_matches)
    beta_sq = beta ** 2
    f_beta = ((1 + beta_sq) * prec * rec) / ((beta_sq * prec) + rec)
    return float(f_beta)


def evaluate_macro_f05(
    all_s1_ids: List[str],
    ground_truth_map: Dict[str, Set[str]],
    predicted_map: Dict[str, Set[str]],
    beta: float = BETA
) -> Dict[str, float]:
    """
    Computes macro-averaged F0.5, Precision, and Recall across all reference S1 entities.
    """
    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    n_entities = len(all_s1_ids)

    if n_entities == 0:
        return {"macro_f0.5": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

    for s1_id in all_s1_ids:
        true_set = ground_truth_map.get(s1_id, set())
        pred_set = predicted_map.get(s1_id, set())

        # F0.5
        f05 = compute_entity_f05(true_set, pred_set, beta=beta)
        total_f05 += f05

        # Precision & Recall tracking
        if len(true_set) == 0:
            total_prec += 1.0 if len(pred_set) == 0 else 0.0
            total_rec += 1.0
        else:
            tp = len(true_set & pred_set)
            prec = (tp / len(pred_set)) if len(pred_set) > 0 else 0.0
            rec = tp / len(true_set)
            total_prec += prec
            total_rec += rec

    return {
        "macro_f0.5": round(total_f05 / n_entities, 5),
        "macro_precision": round(total_prec / n_entities, 5),
        "macro_recall": round(total_rec / n_entities, 5)
    }


def apply_query_exclusivity(
    scored_pairs_df: pd.DataFrame,
    abs_threshold: float = 0.50,
    margin_threshold: float = 0.05
) -> Dict[str, str]:
    """
    Assigns each query to at most one reference S1 entity based on confidence and margin:
    - Group by query_id
    - Sort scores descending
    - If top_score >= abs_threshold and (top_score - second_score) >= margin_threshold:
        assign query -> top S1
    Returns:
        Mapping of query_id -> s1_id
    """
    if len(scored_pairs_df) == 0:
        return {}

    # Sort descending by score
    df_sorted = scored_pairs_df.sort_values(by=["query_id", "score"], ascending=[True, False])
    assignments: Dict[str, str] = {}

    for qid, group in df_sorted.groupby("query_id", sort=False):
        scores = group["score"].values
        s1_ids = group["s1_id"].values

        top_score = scores[0]
        if top_score >= abs_threshold:
            if len(scores) > 1:
                margin = top_score - scores[1]
                if margin >= margin_threshold:
                    assignments[qid] = s1_ids[0]
            else:
                assignments[qid] = s1_ids[0]

    return assignments


def optimize_thresholds_grid(
    scored_pairs_df: pd.DataFrame,
    all_val_s1_ids: List[str],
    ground_truth_map: Dict[str, Set[str]],
    abs_thresholds: Optional[List[float]] = None,
    margin_thresholds: Optional[List[float]] = None
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Runs fine-grained 2D grid search on validation set to find thresholds maximizing Macro F0.5.
    """
    if abs_thresholds is None:
        abs_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    if margin_thresholds is None:
        margin_thresholds = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]

    best_f05 = -1.0
    best_abs = 0.50
    best_margin = 0.05
    best_metrics = {}

    for a_th in abs_thresholds:
        for m_th in margin_thresholds:
            assignments = apply_query_exclusivity(scored_pairs_df, abs_threshold=a_th, margin_threshold=m_th)
            
            # Group predictions by s1_id
            pred_map: Dict[str, Set[str]] = {s1: set() for s1 in all_val_s1_ids}
            for qid, s1_id in assignments.items():
                if s1_id in pred_map:
                    pred_map[s1_id].add(qid)

            metrics = evaluate_macro_f05(all_val_s1_ids, ground_truth_map, pred_map)
            f05 = metrics["macro_f0.5"]

            if f05 > best_f05:
                best_f05 = f05
                best_abs = a_th
                best_margin = m_th
                best_metrics = metrics

    logger.info(f"[OPTIMIZATION] Optimal Thresholds: abs={best_abs:.2f}, margin={best_margin:.2f} -> Macro F0.5={best_f05:.4f}")
    return best_abs, best_margin, best_f05, best_metrics
