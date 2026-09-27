"""
Evaluation and Precision Threshold Optimization Module for 99+ Macro F0.5.
Directly optimizes absolute score and ambiguity margin thresholds on unseen validation entities.
"""

import numpy as np
from typing import Dict, Set, Tuple, List, Any, Optional
import logging

logger = logging.getLogger("PrecisionSieve.Evaluation")

BETA = 0.5
BETA_SQ = BETA ** 2


def compute_entity_f_beta(
    true_set: Set[str],
    pred_set: Set[str],
    beta: float = BETA
) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F_beta for a single reference S1 entity.
    Singleton logic:
    - true_set empty, pred_set empty: (1.0, 1.0, 1.0) [Correct singleton rejection]
    - true_set empty, pred_set non-empty: (0.0, 0.0, 0.0) [Catastrophic false merge]
    - true_set non-empty, pred_set empty: (0.0, 0.0, 0.0) [Missed match]
    """
    beta_sq = beta ** 2
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not true_set and pred_set:
        return 0.0, 0.0, 0.0
    if true_set and not pred_set:
        return 0.0, 0.0, 0.0
        
    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0, 0.0, 0.0
        
    prec = tp / float(len(pred_set))
    rec = tp / float(len(true_set))
    denom = (beta_sq * prec) + rec
    f_beta = (1.0 + beta_sq) * (prec * rec) / denom if denom > 0 else 0.0
    return prec, rec, f_beta


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Computes Macro F0.5, Macro Precision, and Macro Recall across all S1 entities."""
    all_s1_ids = list(ground_truth.keys())
    prec_list, rec_list, f05_list = [], [], []
    
    for s1_id in all_s1_ids:
        true_matches = ground_truth.get(s1_id, set())
        pred_matches = predictions.get(s1_id, set())
        p, r, f = compute_entity_f_beta(true_matches, pred_matches)
        prec_list.append(p)
        rec_list.append(r)
        f05_list.append(f)
        
    return {
        "macro_f0.5": float(np.mean(f05_list)),
        "macro_precision": float(np.mean(prec_list)),
        "macro_recall": float(np.mean(rec_list)),
        "num_entities": len(all_s1_ids)
    }


def optimize_thresholds_grid(
    scored_candidates: List[Dict[str, Any]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: List[str],
    abs_range: Optional[List[float]] = None,
    margin_range: Optional[List[float]] = None
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    2D Grid Search over high-precision thresholds to maximize Macro F0.5.
    """
    if abs_range is None:
        abs_range = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    if margin_range is None:
        margin_range = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
        
    logger.info("[EVALUATION] Starting 2D high-precision grid search...")
    
    query_cand_map: Dict[str, List[Tuple[str, float]]] = {}
    for item in scored_candidates:
        query_cand_map.setdefault(item["query_id"], []).append((item["s1_id"], float(item["score"])))
        
    for qid in query_cand_map:
        query_cand_map[qid].sort(key=lambda x: x[1], reverse=True)
        
    best_f05 = -1.0
    best_abs = 0.60
    best_margin = 0.10
    best_metrics = {}
    
    for abs_th in abs_range:
        for m_th in margin_range:
            preds: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}
            for qid, cands in query_cand_map.items():
                if not cands:
                    continue
                top1_s1, top1_score = cands[0]
                if top1_score < abs_th:
                    continue
                # Pillar 5: Ambiguity rejection via margin gating
                if len(cands) > 1 and m_th > 0:
                    top2_score = cands[1][1]
                    if (top1_score - top2_score) < m_th:
                        continue
                if top1_s1 in preds:
                    preds[top1_s1].add(qid)
                    
            metrics = evaluate_macro_f05(ground_truth, preds)
            f05 = metrics["macro_f0.5"]
            if f05 > best_f05:
                best_f05 = f05
                best_abs = abs_th
                best_margin = m_th
                best_metrics = metrics
                
    logger.info(f"[EVALUATION] Optimal Boundaries: Abs={best_abs:.2f}, Margin={best_margin:.2f} -> Macro F0.5={best_f05:.4f}")
    return best_abs, best_margin, best_f05, best_metrics
