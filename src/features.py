"""
Pairwise Feature Engineering Module for Business Entity Resolution.

Extracts deterministic, high-signal similarity, retrieval, and metadata features
for candidate pairs (query_id, s1_id).

Feature Groups:
1. Name Similarity: Exact match, normalized match, Levenshtein, Jaro-Winkler,
   RapidFuzz ratio/partial/token-sort/token-set, token Jaccard, token overlap,
   length difference, token count difference, transliterated similarity.
2. Address Similarity: Equivalent full feature suite for address.
3. Combined Name + Address: Combined token overlap, weighted composite similarity.
4. Retrieval Metadata: Retrieval agreement count, channel boolean flags,
   BM25 scores/ranks, Char-TFIDF scores/ranks, best retrieval rank, reciprocal rank.
5. Country & Script: Country exact match, country mismatch, country missing indicators,
   script match, script mismatch, missing name/address flags, source indicator (S2 vs S3).
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
import logging

import re
from src.similarity import compute_string_similarities
from src.profiling import detect_script

logger = logging.getLogger(__name__)

US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}
GENERIC_BUSINESS_WORDS = {
    "inc", "corp", "llc", "ltd", "group", "co", "company", "school", "public",
    "center", "services", "service", "pvt", "limited", "optical", "studio",
    "care", "international", "holding", "holdings", "the", "and", "of", "in"
}


def extract_us_state(addr: str) -> Optional[str]:
    """Extracts 2-letter US state code from address string if present."""
    if not addr:
        return None
    tokens = re.findall(r"\b([A-Za-z]{2})\b", addr.upper())
    for t in reversed(tokens):
        if t in US_STATES:
            return t
    return None


def check_acronym_match(q_name: str, s1_name: str) -> int:
    """Checks if query name is an acronym for the multi-word S1 business name."""
    if not q_name or not s1_name:
        return 0
    q_clean = "".join(re.findall(r"[A-Za-z]", q_name.upper()))
    if 2 <= len(q_clean) <= 6:
        words = [w for w in re.findall(r"[A-Za-z]+", s1_name.upper()) if w.lower() not in GENERIC_BUSINESS_WORDS]
        initials = "".join(w[0] for w in words if w)
        if q_clean == initials or (len(q_clean) >= 2 and q_clean == initials[:len(q_clean)]):
            return 1
    return 0


def check_distinct_name_mismatch(q_norm: str, s1_norm: str) -> int:
    """Checks if distinctive non-generic tokens in the two names completely disagree."""
    if not q_norm or not s1_norm:
        return 0
    q_words = {w for w in q_norm.split() if w not in GENERIC_BUSINESS_WORDS and len(w) > 1}
    s1_words = {w for w in s1_norm.split() if w not in GENERIC_BUSINESS_WORDS and len(w) > 1}
    if q_words and s1_words and not (q_words & s1_words):
        return 1
    return 0


FEATURE_COLUMNS = [
    # Retrieval Channel Flags & Scores
    "by_exact_name", "by_exact_address", "by_exact_combined",
    "by_bm25_name", "bm25_name_score", "bm25_name_rank",
    "by_bm25_combined", "bm25_comb_score", "bm25_comb_rank",
    "by_tfidf_name", "tfidf_name_score", "tfidf_name_rank",
    "by_tfidf_address", "tfidf_addr_score", "tfidf_addr_rank",
    "retrieval_agreement_count", "best_retrieval_rank", "best_reciprocal_rank",
    
    # Exact Matches
    "exact_name_raw", "exact_name_normalized",
    "exact_addr_raw", "exact_addr_normalized",
    "exact_combined_normalized",
    
    # Name Similarities
    "name_levenshtein", "name_jaro_winkler", "name_fuzz_ratio", "name_partial_ratio",
    "name_token_sort", "name_token_set", "name_char_len_diff", "name_char_len_ratio",
    "name_token_overlap", "name_token_count_diff", "name_translit_lev",
    
    # Address Similarities
    "addr_levenshtein", "addr_jaro_winkler", "addr_fuzz_ratio", "addr_partial_ratio",
    "addr_token_sort", "addr_token_set", "addr_char_len_diff", "addr_char_len_ratio",
    "addr_token_overlap", "addr_token_count_diff", "addr_translit_lev",
    
    # Combined Similarities
    "combined_token_set", "weighted_composite_similarity",
    
    # Country & Script Metadata (Soft Features)
    "country_exact_match", "country_mismatch", "country_missing",
    "script_match", "script_mismatch",
    
    # Missing Field & Source Indicators
    "query_is_s2", "missing_name_q", "missing_addr_q", "missing_name_s1", "missing_addr_s1",
    
    # High-Precision Disambiguation Signals
    "is_acronym_match", "distinct_name_mismatch", "us_state_mismatch", "missing_addr_penalty"
]


def extract_candidate_features(
    candidate_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    query_df: pd.DataFrame,
    s1_to_matches: Optional[Dict[str, Set[str]]] = None
) -> pd.DataFrame:
    """
    Extracts deterministic pairwise features for candidate pairs.
    Memory-efficient: indexes only the subset of records present in candidate_df.
    
    Args:
        candidate_df: DataFrame with query_id, s1_id, and retrieval flags.
        s1_df: Reference S1 DataFrame.
        query_df: Query DataFrame (S2 / S3).
        s1_to_matches: Optional ground truth dict mapping S1 ID -> set of true query IDs.
                       If provided, adds 'is_match' target column (1 or 0).
    Returns:
        DataFrame containing all FEATURE_COLUMNS plus query_id, s1_id (and is_match if ground truth provided).
    """
    if candidate_df.empty:
        cols = ["query_id", "s1_id"] + FEATURE_COLUMNS
        if s1_to_matches is not None:
            cols.append("is_match")
        return pd.DataFrame(columns=cols)
        
    start_t = time.time()
    num_candidates = len(candidate_df)
    logger.info(f"[FEATURES] Extracting pairwise features for {num_candidates:,} candidate pairs...")
    
    # Filter reference and query DataFrames to only entities appearing in candidate_df
    needed_s1 = set(candidate_df["s1_id"].unique())
    needed_q = set(candidate_df["query_id"].unique())
    
    s1_sub = s1_df[s1_df["entity_id"].isin(needed_s1)]
    q_sub = query_df[query_df["entity_id"].isin(needed_q)]
    
    # Build fast lookup dictionaries
    def build_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        lookup = {}
        for row in df.itertuples():
            lookup[row.entity_id] = {
                "name_raw": getattr(row, "business_name", ""),
                "addr_raw": getattr(row, "business_address", ""),
                "name_norm": getattr(row, "name_normalized", ""),
                "addr_norm": getattr(row, "address_normalized", ""),
                "name_trans": getattr(row, "name_transliterated", ""),
                "addr_trans": getattr(row, "address_transliterated", ""),
                "country": getattr(row, "country", "").strip(),
            }
        return lookup
        
    s1_data = build_lookup(s1_sub)
    q_data = build_lookup(q_sub)
    
    default_record = {
        "name_raw": "", "addr_raw": "", "name_norm": "", "addr_norm": "",
        "name_trans": "", "addr_trans": "", "country": ""
    }
    
    feature_rows = []
    
    for row in candidate_df.itertuples():
        qid = row.query_id
        s1id = row.s1_id
        
        q_info = q_data.get(qid, default_record)
        s1_info = s1_data.get(s1id, default_record)
        
        # Name similarities
        qn_norm = q_info["name_norm"]
        s1n_norm = s1_info["name_norm"]
        n_lev, n_jw, n_ratio, n_part, n_tsort, n_tset, n_diff, n_len_ratio, n_overlap, n_tdiff = compute_string_similarities(qn_norm, s1n_norm)
        
        # Address similarities
        qa_norm = q_info["addr_norm"]
        s1a_norm = s1_info["addr_norm"]
        a_lev, a_jw, a_ratio, a_part, a_tsort, a_tset, a_diff, a_len_ratio, a_overlap, a_tdiff = compute_string_similarities(qa_norm, s1a_norm)
        
        # Transliterated similarities (for multilingual / cross-script pairs)
        qn_trans = q_info["name_trans"]
        s1n_trans = s1_info["name_trans"]
        n_trans_lev, _, _, _, _, _, _, _, _, _ = compute_string_similarities(qn_trans, s1n_trans)
        
        qa_trans = q_info["addr_trans"]
        s1a_trans = s1_info["addr_trans"]
        a_trans_lev, _, _, _, _, _, _, _, _, _ = compute_string_similarities(qa_trans, s1a_trans)
        
        # Combined similarities
        q_comb = (qn_norm + " " + qa_norm).strip()
        s1_comb = (s1n_norm + " " + s1a_norm).strip()
        _, _, _, _, _, comb_tset, _, _, _, _ = compute_string_similarities(q_comb, s1_comb)
        
        # Weighted composite similarity
        weighted_sim = 0.55 * n_jw + 0.35 * a_jw + 0.10 * comb_tset
        
        # Country features (soft, open-set friendly)
        q_country = q_info["country"]
        s1_country = s1_info["country"]
        c_missing = 1 if (not q_country or not s1_country) else 0
        c_match = 1 if (not c_missing and q_country.casefold() == s1_country.casefold()) else 0
        c_mismatch = 1 if (not c_missing and q_country.casefold() != s1_country.casefold()) else 0
        
        # Script features (soft, never hard reject)
        q_script = detect_script(q_info["name_raw"])
        s1_script = detect_script(s1_info["name_raw"])
        script_match = 1 if (q_script == s1_script and q_script != "Empty") else 0
        script_mismatch = 1 if (q_script != s1_script and q_script != "Empty" and s1_script != "Empty") else 0
        
        # Missing field & source indicators
        is_s2 = 1 if qid.startswith("S2-") else 0
        m_name_q = 1 if not q_info["name_raw"] else 0
        m_addr_q = 1 if not q_info["addr_raw"] else 0
        m_name_s1 = 1 if not s1_info["name_raw"] else 0
        m_addr_s1 = 1 if not s1_info["addr_raw"] else 0
        
        # Exact match indicators
        ex_name_raw = 1 if (q_info["name_raw"] and q_info["name_raw"] == s1_info["name_raw"]) else 0
        ex_name_norm = 1 if (qn_norm and qn_norm == s1n_norm) else 0
        ex_addr_raw = 1 if (q_info["addr_raw"] and q_info["addr_raw"] == s1_info["addr_raw"]) else 0
        ex_addr_norm = 1 if (qa_norm and qa_norm == s1a_norm) else 0
        ex_comb_norm = 1 if (q_comb and q_comb == s1_comb) else 0
        
        feat = {
            "query_id": qid,
            "s1_id": s1id,
            
            # Retrieval Channels
            "by_exact_name": getattr(row, "by_exact_name", 0),
            "by_exact_address": getattr(row, "by_exact_address", 0),
            "by_exact_combined": getattr(row, "by_exact_combined", 0),
            "by_bm25_name": getattr(row, "by_bm25_name", 0),
            "bm25_name_score": getattr(row, "bm25_name_score", 0.0),
            "bm25_name_rank": getattr(row, "bm25_name_rank", 999),
            "by_bm25_combined": getattr(row, "by_bm25_combined", 0),
            "bm25_comb_score": getattr(row, "bm25_comb_score", 0.0),
            "bm25_comb_rank": getattr(row, "bm25_comb_rank", 999),
            "by_tfidf_name": getattr(row, "by_tfidf_name", 0),
            "tfidf_name_score": getattr(row, "tfidf_name_score", 0.0),
            "tfidf_name_rank": getattr(row, "tfidf_name_rank", 999),
            "by_tfidf_address": getattr(row, "by_tfidf_address", 0),
            "tfidf_addr_score": getattr(row, "tfidf_addr_score", 0.0),
            "tfidf_addr_rank": getattr(row, "tfidf_addr_rank", 999),
            "retrieval_agreement_count": getattr(row, "retrieval_agreement_count", 0),
            "best_retrieval_rank": getattr(row, "best_retrieval_rank", 999),
            "best_reciprocal_rank": getattr(row, "best_reciprocal_rank", 0.0),
            
            # Exact Matches
            "exact_name_raw": ex_name_raw,
            "exact_name_normalized": ex_name_norm,
            "exact_addr_raw": ex_addr_raw,
            "exact_addr_normalized": ex_addr_norm,
            "exact_combined_normalized": ex_comb_norm,
            
            # Name Similarities
            "name_levenshtein": n_lev,
            "name_jaro_winkler": n_jw,
            "name_fuzz_ratio": n_ratio,
            "name_partial_ratio": n_part,
            "name_token_sort": n_tsort,
            "name_token_set": n_tset,
            "name_char_len_diff": n_diff,
            "name_char_len_ratio": n_len_ratio,
            "name_token_overlap": n_overlap,
            "name_token_count_diff": n_tdiff,
            "name_translit_lev": n_trans_lev,
            
            # Address Similarities
            "addr_levenshtein": a_lev,
            "addr_jaro_winkler": a_jw,
            "addr_fuzz_ratio": a_ratio,
            "addr_partial_ratio": a_part,
            "addr_token_sort": a_tsort,
            "addr_token_set": a_tset,
            "addr_char_len_diff": a_diff,
            "addr_char_len_ratio": a_len_ratio,
            "addr_token_overlap": a_overlap,
            "addr_token_count_diff": a_tdiff,
            "addr_translit_lev": a_trans_lev,
            
            # Combined
            "combined_token_set": comb_tset,
            "weighted_composite_similarity": weighted_sim,
            
            # Metadata
            "country_exact_match": c_match,
            "country_mismatch": c_mismatch,
            "country_missing": c_missing,
            "script_match": script_match,
            "script_mismatch": script_mismatch,
            "query_is_s2": is_s2,
            "missing_name_q": m_name_q,
            "missing_addr_q": m_addr_q,
            "missing_name_s1": m_name_s1,
            "missing_addr_s1": m_addr_s1,
            
            # High-Precision Disambiguation Signals
            "is_acronym_match": check_acronym_match(q_info["name_raw"], s1_info["name_raw"]),
            "distinct_name_mismatch": check_distinct_name_mismatch(qn_norm, s1n_norm),
            "us_state_mismatch": (
                1 if (
                    q_info["country"].upper() == "US" and s1_info["country"].upper() == "US"
                    and extract_us_state(q_info["addr_raw"])
                    and extract_us_state(s1_info["addr_raw"])
                    and extract_us_state(q_info["addr_raw"]) != extract_us_state(s1_info["addr_raw"])
                ) else 0
            ),
            "missing_addr_penalty": 1 if ((m_addr_q ^ m_addr_s1) == 1) else 0,
        }
        
        if s1_to_matches is not None:
            true_queries = s1_to_matches.get(s1id, set())
            feat["is_match"] = 1 if qid in true_queries else 0
            
        feature_rows.append(feat)
        
    feat_df = pd.DataFrame(feature_rows)
    elapsed = time.time() - start_t
    
    n_pos = feat_df["is_match"].sum() if "is_match" in feat_df.columns else 0
    logger.info(
        f"[FEATURES] Extracted {len(FEATURE_COLUMNS)} features for {len(feat_df):,} pairs "
        f"({n_pos:,} positives) in {elapsed:.2f}s."
    )
    return feat_df
