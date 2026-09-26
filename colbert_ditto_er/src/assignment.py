"""
Assignment Engine for ColBERT-Ditto Entity Resolution.
Enforces query exclusivity (at most one match per query), margin filtering, and candidate inclusion rules.
"""

from typing import Dict, Set, List, Tuple, Any, Optional
import logging

logger = logging.getLogger("ColBERT_Ditto.Assignment")


def apply_assignment_rules(
    query_scores: Dict[str, List[Tuple[str, float]]],
    all_s1_ids: List[str],
    abs_threshold: float = 0.50,
    margin_threshold: float = 0.05
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Applies decision rules:
    - Sorts candidate reference entities per query by score descending.
    - Top candidate must satisfy score >= abs_threshold.
    - If second candidate exists, (top1 - top2) >= margin_threshold.
    - Query exclusivity: each query assigned to at most one reference entity.
    
    Returns:
      (matches_map, candidates_map) where keys are all S1 IDs.
    """
    matches_map: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}
    candidates_map: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}
    
    for qid, scored_cands in query_scores.items():
        if not scored_cands:
            continue
            
        # Register all candidates in candidates_map
        for s1_id, _ in scored_cands:
            if s1_id in candidates_map:
                candidates_map[s1_id].add(qid)
                
        # Sort descending by score
        scored_cands.sort(key=lambda x: x[1], reverse=True)
        top1_s1, top1_score = scored_cands[0]
        
        # Absolute threshold check
        if top1_score < abs_threshold:
            continue
            
        # Margin check if second candidate present
        if len(scored_cands) > 1 and margin_threshold > 0:
            top2_score = scored_cands[1][1]
            if (top1_score - top2_score) < margin_threshold:
                continue
                
        # Assign query to top S1
        if top1_s1 in matches_map:
            matches_map[top1_s1].add(qid)
            
    return matches_map, candidates_map
