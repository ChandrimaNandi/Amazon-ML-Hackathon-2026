"""
Threshold Optimization & Multi-Match Decision Rule Module.

Features:
- Independent probability thresholding and query-margin decision logic.
- Respects multi-match structure: reference S1 entities can link to multiple queries.
- Optimizes both absolute threshold (abs_threshold) and margin threshold (margin_threshold)
  strictly on validation set to maximize Macro F0.5.
- Freezes and logs optimal parameters for test inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

from src.evaluation import evaluate_macro_metrics

logger = logging.getLogger(__name__)


def apply_decision_rules(
    cand_df_with_probs: pd.DataFrame,
    abs_threshold: float = 0.50,
    margin_threshold: float = 0.0,
    enforce_query_exclusivity: bool = True
) -> Dict[str, Set[str]]:
    """
    Applies decision rules to scored candidate pairs to produce {s1_id: set(matched_query_ids)}.
    
    Structure:
    - Each candidate pair (query_id, s1_id) has pred_score.
    - If enforce_query_exclusivity is True: Each query record can match at most ONE reference S1 entity
      (the highest-scoring S1 with pred_score >= abs_threshold, subject to margin_threshold).
    - If enforce_query_exclusivity is False: Every pair with pred_score >= abs_threshold is accepted.
    - An S1 reference entity can accumulate zero, one, or multiple matching queries.
    
    Returns:
        s1_to_pred_matches: Dict mapping S1 ID -> set of matched query IDs.
    """
    s1_to_pred: Dict[str, Set[str]] = {}
    
    if cand_df_with_probs.empty or "pred_score" not in cand_df_with_probs.columns:
        return s1_to_pred
        
    # Filter candidates meeting minimum absolute threshold
    above = cand_df_with_probs[cand_df_with_probs["pred_score"] >= abs_threshold]
    if above.empty:
        return s1_to_pred
        
    if not enforce_query_exclusivity:
        # Pure independent thresholding
        for row in above.itertuples():
            s1_to_pred.setdefault(row.s1_id, set()).add(row.query_id)
        return s1_to_pred
        
    # Group by query_id to find top candidate and evaluate margin
    for qid, group in above.groupby("query_id"):
        sorted_group = group.sort_values("pred_score", ascending=False)
        top_row = sorted_group.iloc[0]
        top_score = float(top_row["pred_score"])
        top_s1 = top_row["s1_id"]
        
        # Check margin against runner-up candidate for the same query
        if margin_threshold > 0.0 and len(sorted_group) > 1:
            second_score = float(sorted_group.iloc[1]["pred_score"])
            margin = top_score - second_score
            if margin < margin_threshold:
                # Ambiguous query — reject to protect high precision required by Macro F0.5
                continue
                
        s1_to_pred.setdefault(top_s1, set()).add(qid)
        
    return s1_to_pred


def optimize_threshold_grid(
    val_cand_df_with_probs: pd.DataFrame,
    val_s1_ids: Set[str],
    s1_to_true_matches: Dict[str, Set[str]],
    abs_thresholds: Optional[List[float]] = None,
    margin_thresholds: Optional[List[float]] = None,
    beta: float = 0.5
) -> Dict[str, Any]:
    """
    Grid sweep over absolute and margin thresholds strictly on validation set.
    Selects parameters maximizing Macro F0.5.
    
    Returns:
        best_params: Dict containing best_threshold, best_margin, best_macro_f05,
                     best_macro_precision, best_macro_recall, and the full sweep history.
    """
    if abs_thresholds is None:
        abs_thresholds = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    if margin_thresholds is None:
        margin_thresholds = [0.00, 0.02, 0.05, 0.10, 0.15]
        
    logger.info("=" * 60)
    logger.info(f"[THRESHOLD OPTIMIZATION] Starting grid sweep: {len(abs_thresholds)} abs thresholds x {len(margin_thresholds)} margins")
    logger.info("=" * 60)
    
    best_f_beta = -1.0
    best_abs = 0.50
    best_margin = 0.00
    best_prec = 0.0
    best_rec = 0.0
    
    sweep_records = []
    
    for abs_th in abs_thresholds:
        for m_th in margin_thresholds:
            s1_preds = apply_decision_rules(
                cand_df_with_probs=val_cand_df_with_probs,
                abs_threshold=abs_th,
                margin_threshold=m_th,
                enforce_query_exclusivity=True
            )
            
            metrics = evaluate_macro_metrics(
                all_s1_ids=val_s1_ids,
                s1_to_true_matches=s1_to_true_matches,
                s1_to_pred_matches=s1_preds,
                beta=beta
            )
            
            f_score = metrics[f"macro_f{beta}"]
            prec = metrics["macro_precision"]
            rec = metrics["macro_recall"]
            n_preds = sum(len(m) for m in s1_preds.values())
            
            sweep_records.append({
                "abs_threshold": abs_th,
                "margin_threshold": m_th,
                f"macro_f{beta}": f_score,
                "macro_precision": prec,
                "macro_recall": rec,
                "predicted_links": n_preds
            })
            
            if f_score > best_f_beta:
                best_f_beta = f_score
                best_abs = abs_th
                best_margin = m_th
                best_prec = prec
                best_rec = rec
                
    sweep_df = pd.DataFrame(sweep_records).sort_values(f"macro_f{beta}", ascending=False).reset_index(drop=True)
    
    logger.info("=" * 60)
    logger.info(f"[THRESHOLD OPTIMIZATION] Optimal Configuration Found:")
    logger.info(f"  Best Absolute Threshold: {best_abs:.2f}")
    logger.info(f"  Best Margin Threshold:   {best_margin:.2f}")
    logger.info(f"  Best Validation Macro F{beta}: {best_f_beta:.4f} (Precision: {best_prec:.4f}, Recall: {best_rec:.4f})")
    logger.info("=" * 60)
    
    return {
        "best_threshold": best_abs,
        "best_margin": best_margin,
        f"best_macro_f{beta}": best_f_beta,
        "best_precision": best_prec,
        "best_recall": best_rec,
        "sweep_history": sweep_df
    }
