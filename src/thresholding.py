"""
Threshold Optimization & Decision Rule Module for Business Entity Resolution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


def apply_decision_rules(
    cand_df_with_probs: pd.DataFrame,
    abs_threshold: float = 0.50,
    margin_threshold: float = 0.05,
    min_agreement_count: int = 1
) -> Dict[str, Set[str]]:
    """
    Applies decision rules on scored candidates to generate predicted matches.
    
    cand_df_with_probs must contain:
    - query_id
    - s1_id
    - pred_score
    - retrieval_agreement_count
    
    Returns:
        s1_to_predicted_matches: Dict mapping S1 ID -> set of matched query IDs (S2/S3)
    """
    s1_to_predicted: Dict[str, Set[str]] = {}
    
    if cand_df_with_probs.empty:
        return s1_to_predicted
        
    # Group candidates by query_id
    for q_id, group in cand_df_with_probs.groupby("query_id"):
        # Sort candidates descending by predicted probability score
        sorted_cands = group.sort_values("pred_score", ascending=False)
        
        selected_s1_ids = set()
        
        scores = sorted_cands["pred_score"].values
        agree_counts = sorted_cands["retrieval_agreement_count"].values
        s1_ids = sorted_cands["s1_id"].values
        
        top1_score = scores[0] if len(scores) > 0 else 0.0
        top2_score = scores[1] if len(scores) > 1 else 0.0
        margin = top1_score - top2_score
        
        for idx in range(len(scores)):
            sc = scores[idx]
            ag = agree_counts[idx]
            s1_id = s1_ids[idx]
            
            # Acceptance conditions
            if sc >= abs_threshold and ag >= min_agreement_count:
                # If top candidate, check margin unless score is extremely high (e.g. >= 0.85)
                if idx == 0:
                    if sc >= 0.85 or margin >= margin_threshold or len(scores) == 1:
                        selected_s1_ids.add(s1_id)
                else:
                    # Multi-match candidate: must also satisfy threshold
                    if sc >= abs_threshold + 0.1:
                        selected_s1_ids.add(s1_id)
                        
        for s1_id in selected_s1_ids:
            s1_to_predicted.setdefault(s1_id, set()).add(q_id)
            
    return s1_to_predicted
