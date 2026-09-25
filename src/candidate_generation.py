"""
Candidate Generation Module for Business Entity Resolution.
Combines multiple retrieval channels into a unified candidate set per query record.
Supports fast scalable retrieval for multi-million query test sets.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any
import logging

from src.retrieval import BM25Retriever, CharTFIDFRetriever
from src.normalization import create_normalized_features

logger = logging.getLogger(__name__)


def generate_candidate_union(
    s1_df: pd.DataFrame,
    query_df: pd.DataFrame,
    k_name: int = 20,
    k_address: int = 15,
    k_combined: int = 20,
    k_char: int = 20
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates candidates for query_df against reference s1_df using multiple retrieval channels.
    
    Returns:
        candidate_pairs_df: DataFrame with query_id, s1_id, and retrieval diagnostic flags/ranks/scores.
        stats: Diagnostic statistics on retrieval performance.
    """
    logger.info("=" * 60)
    logger.info(f"Starting Candidate Generation for {len(query_df):,} queries against {len(s1_df):,} S1 reference records")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Ensure normalized features exist
    if "name_normalized" not in s1_df.columns:
        s1_df = create_normalized_features(s1_df)
    if "name_normalized" not in query_df.columns:
        query_df = create_normalized_features(query_df)
        
    s1_ids = s1_df["entity_id"].tolist()
    query_ids = query_df["entity_id"].tolist()
    
    # Map normalized S1 names/addresses to set of S1 IDs for exact match indexing
    exact_name_map: Dict[str, Set[str]] = {}
    exact_addr_map: Dict[str, Set[str]] = {}
    
    for s1_id, name, addr in zip(s1_ids, s1_df["name_normalized"], s1_df["address_normalized"]):
        if name:
            exact_name_map.setdefault(name, set()).add(s1_id)
        if addr:
            exact_addr_map.setdefault(addr, set()).add(s1_id)
            
    is_large_scale = len(query_df) > 500000
    
    # 1. Fit Retrievers
    # Char-TFIDF Name & Address (Ultra-fast GPU/sparse dot product)
    char_name = CharTFIDFRetriever()
    char_name.fit(s1_df["name_normalized"].tolist(), s1_ids)
    
    char_addr = CharTFIDFRetriever()
    char_addr.fit(s1_df["address_normalized"].tolist(), s1_ids)
    
    char_name_res = char_name.retrieve_top_k(query_df["name_normalized"].tolist(), top_k=k_char)
    char_addr_res = char_addr.retrieve_top_k(query_df["address_normalized"].tolist(), top_k=k_address)
    
    bm25_name_res = [[] for _ in range(len(query_df))]
    bm25_comb_res = [[] for _ in range(len(query_df))]
    
    if not is_large_scale:
        # On smaller training/validation sets, also run BM25 for extra coverage
        bm25_name = BM25Retriever()
        bm25_name.fit(s1_df["name_normalized"].tolist(), s1_ids)
        
        bm25_combined = BM25Retriever()
        bm25_combined.fit(s1_df["combined_normalized"].tolist(), s1_ids)
        
        bm25_name_res = bm25_name.retrieve_top_k(query_df["name_normalized"].tolist(), top_k=k_name)
        bm25_comb_res = bm25_combined.retrieve_top_k(query_df["combined_normalized"].tolist(), top_k=k_combined)
    else:
        logger.info("Large query dataset detected (>500k records). Utilizing Ultra-Fast Char-TFIDF & Exact Match retrieval.")

    # 3. Candidate Assembly
    candidate_records = []
    
    for idx, (q_id, q_name, q_addr) in enumerate(zip(query_ids, query_df["name_normalized"], query_df["address_normalized"])):
        candidates: Dict[str, Dict[str, Any]] = {}
        
        def get_cand(s1_id: str) -> Dict[str, Any]:
            if s1_id not in candidates:
                candidates[s1_id] = {
                    "query_id": q_id,
                    "s1_id": s1_id,
                    "by_exact_name": 0,
                    "by_exact_address": 0,
                    "by_bm25_name": 0,
                    "bm25_name_score": 0.0,
                    "bm25_name_rank": 999,
                    "by_bm25_combined": 0,
                    "bm25_comb_score": 0.0,
                    "bm25_comb_rank": 999,
                    "by_char_name": 0,
                    "char_name_score": 0.0,
                    "char_name_rank": 999,
                    "by_char_address": 0,
                    "char_addr_score": 0.0,
                    "char_addr_rank": 999,
                }
            return candidates[s1_id]

        # Exact Name Matches
        if q_name and q_name in exact_name_map:
            for s1_id in exact_name_map[q_name]:
                c = get_cand(s1_id)
                c["by_exact_name"] = 1
                
        # Exact Address Matches
        if q_addr and q_addr in exact_addr_map:
            for s1_id in exact_addr_map[q_addr]:
                c = get_cand(s1_id)
                c["by_exact_address"] = 1
                
        # BM25 Name
        for s1_id, score, rank in bm25_name_res[idx]:
            c = get_cand(s1_id)
            c["by_bm25_name"] = 1
            c["bm25_name_score"] = score
            c["bm25_name_rank"] = rank
            
        # BM25 Combined
        for s1_id, score, rank in bm25_comb_res[idx]:
            c = get_cand(s1_id)
            c["by_bm25_combined"] = 1
            c["bm25_comb_score"] = score
            c["bm25_comb_rank"] = rank
            
        # Char TF-IDF Name
        for s1_id, score, rank in char_name_res[idx]:
            c = get_cand(s1_id)
            c["by_char_name"] = 1
            c["char_name_score"] = score
            c["char_name_rank"] = rank
            
        # Char TF-IDF Address
        for s1_id, score, rank in char_addr_res[idx]:
            c = get_cand(s1_id)
            c["by_char_address"] = 1
            c["char_addr_score"] = score
            c["char_addr_rank"] = rank
            
        # Compute agreement features
        for s1_id, c in candidates.items():
            agree_cnt = (
                c["by_exact_name"] + c["by_exact_address"] +
                c["by_bm25_name"] + c["by_bm25_combined"] +
                c["by_char_name"] + c["by_char_address"]
            )
            c["retrieval_agreement_count"] = agree_cnt
            c["best_retrieval_rank"] = min(
                c["bm25_name_rank"], c["bm25_comb_rank"], c["char_name_rank"], c["char_addr_rank"]
            )
            candidate_records.append(c)
            
    candidate_df = pd.DataFrame(candidate_records)
    elapsed = time.time() - start_time
    
    cand_counts = candidate_df.groupby("query_id").size() if not candidate_df.empty else pd.Series(dtype=int)
    avg_cands = float(cand_counts.mean()) if not cand_counts.empty else 0.0
    median_cands = float(cand_counts.median()) if not cand_counts.empty else 0.0
    
    stats = {
        "num_queries": len(query_df),
        "total_candidate_pairs": len(candidate_df),
        "avg_candidates_per_query": round(avg_cands, 2),
        "median_candidates_per_query": round(median_cands, 2),
        "elapsed_seconds": round(elapsed, 2),
    }
    
    logger.info(f"Generated {len(candidate_df):,} candidate pairs (avg {avg_cands:.1f}/query) in {elapsed:.2f}s.")
    return candidate_df, stats
