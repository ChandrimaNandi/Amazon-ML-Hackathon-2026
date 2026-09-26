"""
Assignment Engine for ColBERT-Ditto Entity Resolution.
Enforces query exclusivity (at most one match per query), margin filtering, and candidate inclusion rules.
"""

from typing import Dict, Set, List, Tuple, Any, Optional
import logging

logger = logging.getLogger("ColBERT_Ditto.Assignment")


def apply_assignment_rules(
    query_scores: Dict[str, List[Tuple[str, float]]],
    abs_threshold: float = 0.50,
    margin_threshold: float = 0.05
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Applies decision rules sparsely for the current chunk.
    Only returns entities touched by the queries in this chunk, avoiding
    allocating millions of empty sets per chunk.
    
    Returns:
      (chunk_matches, chunk_candidates): sparse {s1_id: set(qids)} maps.
    """
    matches_map: Dict[str, Set[str]] = {}
    candidates_map: Dict[str, Set[str]] = {}
    
    for qid, scored_cands in query_scores.items():
        if not scored_cands:
            continue
            
        # Register all candidates in candidates_map
        for s1_id, _ in scored_cands:
            candidates_map.setdefault(s1_id, set()).add(qid)
                
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
        matches_map.setdefault(top1_s1, set()).add(qid)
            
    return matches_map, candidates_map
