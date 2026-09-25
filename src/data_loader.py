"""
Data Loader Module for Business Entity Resolution.
Provides robust reading of TSV files and ground truth parsing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, Union, List
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_source_tsv(path: Union[Path, str]) -> pd.DataFrame:
    """
    Loads a source TSV file (entity_id, business_name, business_address, country).
    Fills missing string values with empty string.
    """
    path = Path(path)
    logger.info(f"Loading TSV file from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Source file not found at {path}")
    
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8"
    )
    
    expected_cols = ["entity_id", "business_name", "business_address", "country"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {path}, got {df.columns.tolist()}")
    
    # Clean whitespace and handle NaN/empty
    df["entity_id"] = df["entity_id"].astype(str).str.strip()
    df["business_name"] = df["business_name"].astype(str).str.strip()
    df["business_address"] = df["business_address"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    
    logger.info(f"Loaded {len(df):,} rows from {path.name}")
    return df


def load_ground_truth(path: Union[Path, str]) -> Tuple[pd.DataFrame, Dict[str, Set[str]], Dict[str, str]]:
    """
    Loads train_ground_truth.tsv.
    Returns:
        gt_df: DataFrame with source1_entity_id and matched_entity_ids
        s1_to_matches: Dict mapping S1 ID -> set of matched S2/S3 IDs
        match_to_s1: Dict mapping S2/S3 ID -> S1 ID
    """
    path = Path(path)
    logger.info(f"Loading Ground Truth from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found at {path}")

    gt_df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8"
    )
    
    s1_to_matches: Dict[str, Set[str]] = {}
    match_to_s1: Dict[str, str] = {}
    
    s1_col = gt_df["source1_entity_id"].astype(str).str.strip()
    match_col = gt_df["matched_entity_ids"].astype(str).str.strip()
    
    for s1_id, matches_str in zip(s1_col, match_col):
        if matches_str:
            matched_set = set(m.strip() for m in matches_str.split(",") if m.strip())
        else:
            matched_set = set()
        
        s1_to_matches[s1_id] = matched_set
        for m in matched_set:
            match_to_s1[m] = s1_id
            
    logger.info(f"Loaded {len(s1_to_matches):,} ground truth S1 entities ({len(match_to_s1):,} total match mappings)")
    return gt_df, s1_to_matches, match_to_s1


def load_coherent_training_sample(
    s1_path: Union[Path, str],
    gt_path: Union[Path, str],
    s2_path: Union[Path, str],
    s3_path: Union[Path, str],
    sample_s1_rows: int = 25000,
    max_active_queries: Optional[int] = 25000,
    num_unmatched_queries: int = 2000,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """
    Loads a coherent, leak-free training sample where queries in query_df
    genuinely correspond to the selected S1 entities in s1_df.
    
    Fixes the query mismatch bug where queries were loaded from head(N) without
    matching the sampled S1 entity IDs.
    
    Returns:
        s1_df: DataFrame of sampled S1 entities
        query_df: DataFrame of matching queries + controlled unmatched negatives
        sample_gt: Dict mapping sampled S1 entity_id -> set of true matching query IDs
    """
    s1_path = Path(s1_path)
    gt_path = Path(gt_path)
    s2_path = Path(s2_path)
    s3_path = Path(s3_path)
    
    logger.info(f"[DATA] Loading coherent training sample (S1 rows: {sample_s1_rows:,}, Queries max: {max_active_queries})...")
    
    # 1. Load S1 Sample
    s1_df = pd.read_csv(s1_path, sep="\t", nrows=sample_s1_rows, dtype=str, keep_default_na=False)
    for col in ["entity_id", "business_name", "business_address", "country"]:
        s1_df[col] = s1_df[col].astype(str).str.strip()
    sample_s1_ids = set(s1_df["entity_id"])
    
    # 2. Extract ground truth for sampled S1
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    sample_gt: Dict[str, Set[str]] = {}
    target_q_ids: Set[str] = set()
    
    s1_col = gt_df["source1_entity_id"].astype(str).str.strip()
    match_col = gt_df["matched_entity_ids"].astype(str).str.strip()
    
    for s1_id, matches_str in zip(s1_col, match_col):
        if s1_id in sample_s1_ids:
            if matches_str:
                q_set = set(m.strip() for m in matches_str.split(",") if m.strip())
            else:
                q_set = set()
            sample_gt[s1_id] = q_set
            target_q_ids.update(q_set)
            
    logger.info(f"[DATA] Selected {len(s1_df):,} S1 entities ({len(target_q_ids):,} total target query IDs across dataset)")
    
    # If target_q_ids exceeds max_active_queries, sample a deterministic subset
    rng = np.random.RandomState(random_seed)
    active_target_q = target_q_ids
    if max_active_queries and len(target_q_ids) > max_active_queries:
        active_target_list = sorted(list(target_q_ids))
        rng.shuffle(active_target_list)
        active_target_q = set(active_target_list[:max_active_queries])
        # Update sample_gt to only retain active target queries
        sample_gt = {s1: (q_set & active_target_q) for s1, q_set in sample_gt.items()}
        logger.info(f"[DATA] Capped active target queries to {len(active_target_q):,} for balanced training")

    # 3. Stream S2 and S3 to load active target queries + genuine negative queries
    matched_dfs: List[pd.DataFrame] = []
    unmatched_dfs: List[pd.DataFrame] = []
    unmatched_collected = 0

    for src_path in [s2_path, s3_path]:
        if not src_path.exists():
            continue
        for chunk in pd.read_csv(src_path, sep="\t", chunksize=200000, dtype=str, keep_default_na=False):
            for col in ["entity_id", "business_name", "business_address", "country"]:
                chunk[col] = chunk[col].astype(str).str.strip()
            
            is_target = chunk["entity_id"].isin(active_target_q)
            if is_target.any():
                matched_dfs.append(chunk[is_target])
                
            if unmatched_collected < num_unmatched_queries:
                needed = num_unmatched_queries - unmatched_collected
                neg_sample = chunk[~is_target & ~chunk["entity_id"].isin(target_q_ids)].head(needed)
                if not neg_sample.empty:
                    unmatched_dfs.append(neg_sample)
                    unmatched_collected += len(neg_sample)

    query_parts = matched_dfs + unmatched_dfs
    query_df = pd.concat(query_parts, ignore_index=True).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    
    total_true = sum(len(q) for q in sample_gt.values())
    logger.info(f"[DATA] Coherent sample ready:")
    logger.info(f"  Reference S1: {len(s1_df):,}")
    logger.info(f"  Queries:      {len(query_df):,} (Matched: {len(query_df) - unmatched_collected:,}, Unmatched: {unmatched_collected:,})")
    logger.info(f"  Active True Links: {total_true:,}")
    
    return s1_df, query_df, sample_gt
