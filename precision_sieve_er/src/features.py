"""
Pillar 3: High-Precision Feature Engineering for 99+ Macro F0.5.
Combines RapidFuzz distances with numerical disambiguation, PIN code verification, and brand prefix salience.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
from rapidfuzz import fuzz, distance

from precision_sieve_er.src.veto_gates import (
    extract_numbers, extract_pin_codes, extract_us_state,
    is_numeric_conflict, is_pin_conflict, is_geographic_conflict
)

logger = logging.getLogger("PrecisionSieve.Features")


def extract_pair_features(
    q_name: str,
    s_name: str,
    q_addr: str,
    s_addr: str,
    q_country: str = "",
    s_country: str = "",
    bm25_score: float = 0.0,
    retrieval_rank: int = 1
) -> Dict[str, float]:
    """Extracts high-discriminator pairwise features for a candidate pair."""
    feat: Dict[str, float] = {}

    # 1. High-Discriminator Numeric & Physical Features
    q_nums = extract_numbers(q_addr)
    s_nums = extract_numbers(s_addr)
    if q_nums and s_nums:
        inter = len(q_nums.intersection(s_nums))
        union = len(q_nums.union(s_nums))
        feat["num_jaccard"] = inter / float(union) if union > 0 else 0.0
        feat["num_conflict"] = 1.0 if inter == 0 else 0.0
    else:
        feat["num_jaccard"] = 1.0 if not q_nums and not s_nums else 0.5
        feat["num_conflict"] = 0.0

    # PIN code match
    q_pins = extract_pin_codes(q_addr)
    s_pins = extract_pin_codes(s_addr)
    if q_pins and s_pins:
        feat["pin_match"] = 1.0 if not q_pins.isdisjoint(s_pins) else -1.0
    else:
        feat["pin_match"] = 0.0

    # Geographic signals
    qc = str(q_country).strip().upper() if q_country else ""
    sc = str(s_country).strip().upper() if s_country else ""
    feat["country_match"] = 1.0 if qc and sc and qc == sc else 0.0
    feat["country_mismatch"] = 1.0 if qc and sc and qc != sc else 0.0

    q_state = extract_us_state(q_addr)
    s_state = extract_us_state(s_addr)
    if q_state and s_state:
        feat["state_match"] = 1.0 if q_state == s_state else 0.0
        feat["state_mismatch"] = 1.0 if q_state != s_state else 0.0
    else:
        feat["state_match"] = 0.5
        feat["state_mismatch"] = 0.0

    # 2. Brand Prefix Salience (First two words)
    q_tokens = q_name.split()
    s_tokens = s_name.split()
    q_prefix = " ".join(q_tokens[:2]) if q_tokens else ""
    s_prefix = " ".join(s_tokens[:2]) if s_tokens else ""
    feat["brand_prefix_jw"] = distance.JaroWinkler.similarity(q_prefix, s_prefix) if q_prefix and s_prefix else 0.0
    feat["first_word_match"] = 1.0 if q_tokens and s_tokens and q_tokens[0] == s_tokens[0] else 0.0

    # 3. Core RapidFuzz String Similarities (Top-ranked drivers)
    feat["name_jw"] = distance.JaroWinkler.similarity(q_name, s_name)
    feat["name_sort"] = fuzz.token_sort_ratio(q_name, s_name) / 100.0
    feat["name_set"] = fuzz.token_set_ratio(q_name, s_name) / 100.0
    feat["name_ratio"] = fuzz.ratio(q_name, s_name) / 100.0
    feat["name_partial"] = fuzz.partial_ratio(q_name, s_name) / 100.0

    feat["addr_jw"] = distance.JaroWinkler.similarity(q_addr, s_addr)
    feat["addr_sort"] = fuzz.token_sort_ratio(q_addr, s_addr) / 100.0
    feat["addr_set"] = fuzz.token_set_ratio(q_addr, s_addr) / 100.0
    feat["addr_ratio"] = fuzz.ratio(q_addr, s_addr) / 100.0
    feat["addr_partial"] = fuzz.partial_ratio(q_addr, s_addr) / 100.0

    q_comb = f"{q_name} {q_addr}"
    s_comb = f"{s_name} {s_addr}"
    feat["comb_sort"] = fuzz.token_sort_ratio(q_comb, s_comb) / 100.0
    feat["comb_set"] = fuzz.token_set_ratio(q_comb, s_comb) / 100.0

    # Length discrepancies
    len_qn, len_sn = len(q_name), len(s_name)
    feat["name_len_diff"] = abs(len_qn - len_sn)
    feat["name_len_ratio"] = min(len_qn, len_sn) / max(len_qn, len_sn, 1)

    len_qa, len_sa = len(q_addr), len(s_addr)
    feat["addr_len_diff"] = abs(len_qa - len_sa)
    feat["addr_len_ratio"] = min(len_qa, len_sa) / max(len_qa, len_sa, 1)

    # 4. Retrieval Rank & BM25 Signals
    feat["bm25_score"] = float(bm25_score)
    feat["retrieval_rank"] = float(retrieval_rank)
    feat["is_top1"] = 1.0 if retrieval_rank == 1 else 0.0

    return feat


def extract_candidate_features_dataframe(
    pairs_list: List[Dict[str, Any]],
    s1_lookup: Dict[str, Dict[str, str]],
    query_lookup: Dict[str, Dict[str, str]]
) -> pd.DataFrame:
    """
    Vectorized extraction of features for a list of candidate pair dicts.
    Each pair dict: {'query_id': ..., 's1_id': ..., 'bm25_score': ..., 'rank': ...}
    """
    rows = []
    for p in pairs_list:
        qid = p["query_id"]
        s1id = p["s1_id"]
        q_data = query_lookup.get(qid, {})
        s_data = s1_lookup.get(s1id, {})

        f = extract_pair_features(
            q_name=q_data.get("name_norm", ""),
            s_name=s_data.get("name_norm", ""),
            q_addr=q_data.get("addr_norm", ""),
            s_addr=s_data.get("addr_norm", ""),
            q_country=q_data.get("country", ""),
            s_country=s_data.get("country", ""),
            bm25_score=p.get("bm25_score", 0.0),
            retrieval_rank=p.get("rank", 1)
        )
        f["query_id"] = qid
        f["s1_id"] = s1id
        rows.append(f)

    return pd.DataFrame(rows)
