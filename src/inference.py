"""
Inference Pipeline Module for Business Entity Resolution.
Optimized for large-scale test datasets (9M+ queries).
Uses CharTFIDF + Exact Match retrieval (skips BM25 for speed).
"""

import time
import gc
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple, Any, Optional
import logging

from src.config import (
    TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH,
    SUBMISSION_MATCHING_PATH, SUBMISSION_CANDIDATE_PATH
)
from src.data_loader import load_source_tsv
from src.normalization import create_normalized_features
from src.retrieval import CharTFIDFRetriever
from src.features import extract_candidate_features
from src.ranking import EntityMatcherModel
from src.thresholding import apply_decision_rules

logger = logging.getLogger(__name__)


def _build_candidates_fast(
    s1_df: pd.DataFrame,
    query_df: pd.DataFrame,
    k_char_name: int = 20,
    k_char_addr: int = 15,
    chunk_size: int = 500000
) -> pd.DataFrame:
    """
    Fast candidate generation using CharTFIDF + Exact Match only.
    Processes queries in chunks to manage memory on large datasets.
    """
    logger.info(f"Fast Candidate Generation: {len(query_df):,} queries vs {len(s1_df):,} S1 records")
    start_t = time.time()
    
    s1_ids = s1_df["entity_id"].tolist()
    
    # Build exact match indices
    exact_name_map: Dict[str, Set[str]] = {}
    exact_addr_map: Dict[str, Set[str]] = {}
    for s1_id, name, addr in zip(s1_ids, s1_df["name_normalized"], s1_df["address_normalized"]):
        if name:
            exact_name_map.setdefault(name, set()).add(s1_id)
        if addr:
            exact_addr_map.setdefault(addr, set()).add(s1_id)
    
    # Fit CharTFIDF retrievers on S1 corpus
    char_name = CharTFIDFRetriever(use_gpu=True)
    char_name.fit(s1_df["name_normalized"].tolist(), s1_ids)
    
    char_addr = CharTFIDFRetriever(use_gpu=True)
    char_addr.fit(s1_df["address_normalized"].tolist(), s1_ids)
    
    all_candidate_records = []
    num_queries = len(query_df)
    
    for chunk_start in range(0, num_queries, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_queries)
        q_chunk = query_df.iloc[chunk_start:chunk_end]
        
        logger.info(f"Processing query chunk {chunk_start:,}-{chunk_end:,} / {num_queries:,}")
        
        q_ids = q_chunk["entity_id"].tolist()
        q_names = q_chunk["name_normalized"].tolist()
        q_addrs = q_chunk["address_normalized"].tolist()
        
        # Retrieve candidates
        char_name_res = char_name.retrieve_top_k(q_names, top_k=k_char_name)
        char_addr_res = char_addr.retrieve_top_k(q_addrs, top_k=k_char_addr)
        
        for idx in range(len(q_ids)):
            q_id = q_ids[idx]
            q_name = q_names[idx]
            q_addr = q_addrs[idx]
            
            candidates: Dict[str, Dict[str, Any]] = {}
            
            def get_cand(s1_id):
                if s1_id not in candidates:
                    candidates[s1_id] = {
                        "query_id": q_id, "s1_id": s1_id,
                        "by_exact_name": 0, "by_exact_address": 0,
                        "by_bm25_name": 0, "bm25_name_score": 0.0, "bm25_name_rank": 999,
                        "by_bm25_combined": 0, "bm25_comb_score": 0.0, "bm25_comb_rank": 999,
                        "by_char_name": 0, "char_name_score": 0.0, "char_name_rank": 999,
                        "by_char_address": 0, "char_addr_score": 0.0, "char_addr_rank": 999,
                    }
                return candidates[s1_id]
            
            if q_name and q_name in exact_name_map:
                for s1_id in exact_name_map[q_name]:
                    get_cand(s1_id)["by_exact_name"] = 1
            if q_addr and q_addr in exact_addr_map:
                for s1_id in exact_addr_map[q_addr]:
                    get_cand(s1_id)["by_exact_address"] = 1
            
            for s1_id, score, rank in char_name_res[idx]:
                c = get_cand(s1_id)
                c["by_char_name"] = 1
                c["char_name_score"] = score
                c["char_name_rank"] = rank
            
            for s1_id, score, rank in char_addr_res[idx]:
                c = get_cand(s1_id)
                c["by_char_address"] = 1
                c["char_addr_score"] = score
                c["char_addr_rank"] = rank
            
            for s1_id, c in candidates.items():
                c["retrieval_agreement_count"] = (
                    c["by_exact_name"] + c["by_exact_address"] +
                    c["by_char_name"] + c["by_char_address"]
                )
                c["best_retrieval_rank"] = min(c["char_name_rank"], c["char_addr_rank"])
                all_candidate_records.append(c)
        
        gc.collect()
    
    candidate_df = pd.DataFrame(all_candidate_records)
    elapsed = time.time() - start_t
    logger.info(f"Fast Candidate Generation completed: {len(candidate_df):,} pairs in {elapsed:.2f}s")
    return candidate_df


def generate_submission_files(
    model: EntityMatcherModel,
    test_dir: Path,
    output_matching_path: Path = SUBMISSION_MATCHING_PATH,
    output_candidate_path: Path = SUBMISSION_CANDIDATE_PATH,
    abs_threshold: float = 0.50,
    margin_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Runs full inference pipeline on test dataset using fast CharTFIDF + Exact Match retrieval.
    """
    logger.info("=" * 60)
    logger.info("STARTING INFERENCE PIPELINE ON TEST DATASET")
    logger.info("=" * 60)
    
    start_t = time.time()
    
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"
    
    # 1. Load Data
    s1_df = load_source_tsv(s1_path)
    s2_df = load_source_tsv(s2_path)
    s3_df = load_source_tsv(s3_path)
    
    query_df = pd.concat([s2_df, s3_df], ignore_index=True)
    logger.info(f"Combined {len(query_df):,} query records (S2: {len(s2_df):,}, S3: {len(s3_df):,})")
    del s2_df, s3_df
    gc.collect()
    
    # 2. Normalize
    s1_df = create_normalized_features(s1_df)
    query_df = create_normalized_features(query_df)
    
    # 3. Fast Candidate Generation (CharTFIDF + Exact Match, no BM25)
    cand_df = _build_candidates_fast(s1_df, query_df, k_char_name=20, k_char_addr=15)
    
    # 4. Feature Extraction
    feat_df = extract_candidate_features(
        candidate_df=cand_df, s1_df=s1_df, query_df=query_df, s1_to_matches=None
    )
    
    # 5. Model Scoring
    probs = model.predict_proba(feat_df)
    feat_df["pred_score"] = probs
    
    # 6. Apply Decision Rule
    s1_to_pred_matches = apply_decision_rules(
        cand_df_with_probs=feat_df,
        abs_threshold=abs_threshold,
        margin_threshold=margin_threshold
    )
    
    # Build candidate mapping
    s1_to_candidates: Dict[str, Set[str]] = {}
    all_s1_test_ids = s1_df["entity_id"].tolist()
    
    for row in cand_df.itertuples():
        s1_to_candidates.setdefault(row.s1_id, set()).add(row.query_id)
    
    for s1_id in all_s1_test_ids:
        s1_to_candidates.setdefault(s1_id, set())
        s1_to_pred_matches.setdefault(s1_id, set())
    
    # 7. Write candidate_pairs.tsv
    output_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_test_ids:
            cands = ",".join(sorted(s1_to_candidates[s1_id]))
            f.write(f"{s1_id}\t{cands}\n")
    logger.info(f"Wrote candidate_pairs.tsv to {output_candidate_path}")
    
    # 8. Write matching_results.tsv
    output_matching_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_test_ids:
            matches = ",".join(sorted(s1_to_pred_matches[s1_id]))
            f.write(f"{s1_id}\t{matches}\n")
    logger.info(f"Wrote matching_results.tsv to {output_matching_path}")
    
    elapsed = time.time() - start_t
    
    summary = {
        "num_test_s1_entities": len(all_s1_test_ids),
        "num_test_queries": len(query_df),
        "total_candidates_generated": len(cand_df),
        "s1_with_zero_matches": sum(1 for s in s1_to_pred_matches.values() if len(s) == 0),
        "s1_with_one_match": sum(1 for s in s1_to_pred_matches.values() if len(s) == 1),
        "s1_with_multi_matches": sum(1 for s in s1_to_pred_matches.values() if len(s) > 1),
        "elapsed_seconds": round(elapsed, 2),
    }
    
    logger.info(f"INFERENCE PIPELINE COMPLETED in {elapsed:.0f}s.")
    return summary
