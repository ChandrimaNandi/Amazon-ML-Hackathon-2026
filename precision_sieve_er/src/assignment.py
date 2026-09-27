"""
Pillar 5: Ambiguity Rejection Engine & Precision Margin Gating.
Enforces query exclusivity (at most 1 match per query) and refuses ambiguous guesses to protect singletons.
"""

from typing import Dict, Set, List, Tuple, Any, Optional
import logging

logger = logging.getLogger("PrecisionSieve.Assignment")


def apply_assignment_rules(
    query_scores: Dict[str, List[Tuple[str, float]]],
    tier1_exact_matches: Optional[Dict[str, str]] = None,
    abs_threshold: float = 0.60,
    margin_threshold: float = 0.10
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Applies high-precision assignment rules:
    1. Tier 1 exact matches are locked in deterministically.
    2. Candidate scores must exceed abs_threshold.
    3. Ambiguity rejection: if top1 - top2 < margin_threshold, reject query (refuse to guess).
    4. Query exclusivity: each query matched to at most one reference entity.
    
    Returns:
      (chunk_matches, chunk_candidates): sparse {s1_id: set(qids)}
    """
    matches_map: Dict[str, Set[str]] = {}
    candidates_map: Dict[str, Set[str]] = {}
    
    tier1_map = tier1_exact_matches or {}
    
    for qid, scored_cands in query_scores.items():
        # Register candidates
        for s1_id, _ in scored_cands:
            candidates_map.setdefault(s1_id, set()).add(qid)
            
        # 1. Tier 1 Exact Sieve Resolution (100% precision)
        if qid in tier1_map:
            exact_s1 = tier1_map[qid]
            candidates_map.setdefault(exact_s1, set()).add(qid)
            matches_map.setdefault(exact_s1, set()).add(qid)
            continue
            
        if not scored_cands:
            continue
            
        # 2. Sort candidates by score descending
        scored_cands.sort(key=lambda x: x[1], reverse=True)
        top1_s1, top1_score = scored_cands[0]
        
        # Absolute threshold check
        if top1_score < abs_threshold:
            continue
            
        # Pillar 5: Ambiguity Margin Check
        if len(scored_cands) > 1 and margin_threshold > 0:
            top2_score = scored_cands[1][1]
            if (top1_score - top2_score) < margin_threshold:
                # Ambiguous collision: reject to protect singletons from false merges
                continue
                
        # Assign query to top S1
        matches_map.setdefault(top1_s1, set()).add(qid)
        
    return matches_map, candidates_map
