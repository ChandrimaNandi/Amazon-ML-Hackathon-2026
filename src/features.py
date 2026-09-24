"""
Pairwise Feature Engineering Module for Business Entity Resolution.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
import logging

from src.similarity import compute_string_similarities
from src.profiling import detect_script

logger = logging.getLogger(__name__)


def extract_candidate_features(
    candidate_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    query_df: pd.DataFrame,
    s1_to_matches: Optional[Dict[str, Set[str]]] = None
) -> pd.DataFrame:
    """
    Extracts pairwise features for candidate pairs.
    
    If s1_to_matches (ground truth) is provided, adds a binary target column 'is_match' (1 if s1_id in s1_to_matches[query_id], else 0).
    """
    logger.info(f"Extracting features for {len(candidate_df):,} candidate pairs...")
    start_t = time.time()
    
    # Fast indexing lookup dicts
    s1_data = s1_df.set_index("entity_id")[["name_normalized", "address_normalized", "country"]].to_dict("index")
    query_data = query_df.set_index("entity_id")[["name_normalized", "address_normalized", "country"]].to_dict("index")
    
    feature_rows = []
    
    for row in candidate_df.itertuples():
        q_id = row.query_id
        s1_id = row.s1_id
        
        q_info = query_data.get(q_id, {"name_normalized": "", "address_normalized": "", "country": ""})
        s1_info = s1_data.get(s1_id, {"name_normalized": "", "address_normalized": "", "country": ""})
        
        q_name = q_info["name_normalized"]
        s1_name = s1_info["name_normalized"]
        
        q_addr = q_info["address_normalized"]
        s1_addr = s1_info["address_normalized"]
        
        q_country = q_info["country"]
        s1_country = s1_info["country"]
        
        # Name similarities
        name_lev, name_jw, name_ts, name_tsort, name_jacc, name_len_diff, name_len_ratio = compute_string_similarities(q_name, s1_name)
        
        # Address similarities
        addr_lev, addr_jw, addr_ts, addr_tsort, addr_jacc, addr_len_diff, addr_len_ratio = compute_string_similarities(q_addr, s1_addr)
        
        # Country match features
        c_missing = 1 if (not q_country or not s1_country) else 0
        c_match = 1 if (q_country and s1_country and q_country.casefold() == s1_country.casefold()) else 0
        
        # Script match features
        q_script = detect_script(q_name)
        s1_script = detect_script(s1_name)
        script_match = 1 if q_script == s1_script else 0
        
        feat = {
            "query_id": q_id,
            "s1_id": s1_id,
            
            # Retrieval features
            "by_exact_name": getattr(row, "by_exact_name", 0),
            "by_exact_address": getattr(row, "by_exact_address", 0),
            "by_bm25_name": getattr(row, "by_bm25_name", 0),
            "bm25_name_score": getattr(row, "bm25_name_score", 0.0),
            "bm25_name_rank": getattr(row, "bm25_name_rank", 999),
            "by_bm25_combined": getattr(row, "by_bm25_combined", 0),
            "bm25_comb_score": getattr(row, "bm25_comb_score", 0.0),
            "bm25_comb_rank": getattr(row, "bm25_comb_rank", 999),
            "by_char_name": getattr(row, "by_char_name", 0),
            "char_name_score": getattr(row, "char_name_score", 0.0),
            "char_name_rank": getattr(row, "char_name_rank", 999),
            "by_char_address": getattr(row, "by_char_address", 0),
            "char_addr_score": getattr(row, "char_addr_score", 0.0),
            "char_addr_rank": getattr(row, "char_addr_rank", 999),
            "retrieval_agreement_count": getattr(row, "retrieval_agreement_count", 0),
            "best_retrieval_rank": getattr(row, "best_retrieval_rank", 999),
            
            # String exact match features
            "exact_name_match": 1 if (q_name and q_name == s1_name) else 0,
            "exact_address_match": 1 if (q_addr and q_addr == s1_addr) else 0,
            
            # Name Similarity Features
            "name_levenshtein": name_lev,
            "name_jaro_winkler": name_jw,
            "name_token_set": name_ts,
            "name_token_sort": name_tsort,
            "name_token_jaccard": name_jacc,
            "name_len_diff": name_len_diff,
            "name_len_ratio": name_len_ratio,
            
            # Address Similarity Features
            "address_levenshtein": addr_lev,
            "address_jaro_winkler": addr_jw,
            "address_token_set": addr_ts,
            "address_token_sort": addr_tsort,
            "address_token_jaccard": addr_jacc,
            "address_len_diff": addr_len_diff,
            "address_len_ratio": addr_len_ratio,
            
            # Country & Script Features
            "country_match": c_match,
            "country_missing": c_missing,
            "script_match": script_match,
        }
        
        if s1_to_matches is not None:
            true_matches = s1_to_matches.get(s1_id, set())
            feat["is_match"] = 1 if q_id in true_matches else 0
            
        feature_rows.append(feat)
        
    feat_df = pd.DataFrame(feature_rows)
    elapsed = time.time() - start_t
    
    num_positives = feat_df["is_match"].sum() if "is_match" in feat_df.columns else 0
    logger.info(f"Extracted {feat_df.shape[1]} features for {len(feat_df):,} pairs ({num_positives:,} positive matches) in {elapsed:.2f}s.")
    return feat_df


FEATURE_COLUMNS = [
    "by_exact_name", "by_exact_address", "by_bm25_name", "bm25_name_score", "bm25_name_rank",
    "by_bm25_combined", "bm25_comb_score", "bm25_comb_rank", "by_char_name", "char_name_score",
    "char_name_rank", "by_char_address", "char_addr_score", "char_addr_rank",
    "retrieval_agreement_count", "best_retrieval_rank", "exact_name_match", "exact_address_match",
    "name_levenshtein", "name_jaro_winkler", "name_token_set", "name_token_sort", "name_token_jaccard",
    "name_len_diff", "name_len_ratio", "address_levenshtein", "address_jaro_winkler",
    "address_token_set", "address_token_sort", "address_token_jaccard", "address_len_diff",
    "address_len_ratio", "country_match", "country_missing", "script_match"
]
