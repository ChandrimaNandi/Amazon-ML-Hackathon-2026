"""
Coherent & Leak-Free Data Loader Module for Fast Hybrid ER.
Loads representative reference S1 entities, matching queries from S2/S3, and parsed ground truth.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, Union, List
import logging

logger = logging.getLogger("FastHybridER.DataLoader")


def load_coherent_training_sample(
    s1_path: Union[Path, str],
    gt_path: Union[Path, str],
    s2_path: Union[Path, str],
    s3_path: Union[Path, str],
    sample_s1_rows: int = 15000,
    max_active_queries: Optional[int] = 15000,
    num_unmatched_queries: int = 1500,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """
    Loads a coherent training sample where queries in query_df genuinely correspond
    to the selected reference S1 entities in s1_df.
    """
    s1_path = Path(s1_path)
    gt_path = Path(gt_path)
    s2_path = Path(s2_path)
    s3_path = Path(s3_path)

    logger.info(f"[DATA] Loading coherent sample (S1: {sample_s1_rows:,}, Queries max: {max_active_queries})...")

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

    logger.info(f"[DATA] Selected {len(s1_df):,} S1 entities ({len(target_q_ids):,} query links across dataset)")

    # Sample subset of target queries if needed
    rng = np.random.RandomState(random_seed)
    active_target_q = target_q_ids
    if max_active_queries and len(target_q_ids) > max_active_queries:
        active_target_list = sorted(list(target_q_ids))
        rng.shuffle(active_target_list)
        active_target_q = set(active_target_list[:max_active_queries])
        sample_gt = {s1: (q_set & active_target_q) for s1, q_set in sample_gt.items()}

    # 3. Stream S2 and S3 to load active target queries with prefix-aware early exit
    target_by_prefix = {
        "s2": {q for q in active_target_q if q.startswith("S2-")},
        "s3": {q for q in active_target_q if q.startswith("S3-")}
    }

    matched_dfs: List[pd.DataFrame] = []
    unmatched_dfs: List[pd.DataFrame] = []
    unmatched_collected = 0

    for src_path in [s2_path, s3_path]:
        if not src_path.exists():
            continue
        pfx = "s2" if "source2" in src_path.name.lower() else "s3"
        needed_for_src = target_by_prefix[pfx]
        matched_in_src = 0

        for chunk in pd.read_csv(src_path, sep="\t", chunksize=200000, dtype=str, keep_default_na=False):
            is_target = chunk["entity_id"].isin(needed_for_src)
            if is_target.any():
                matched_rows = chunk[is_target].copy()
                for col in ["entity_id", "business_name", "business_address", "country"]:
                    matched_rows[col] = matched_rows[col].astype(str).str.strip()
                matched_dfs.append(matched_rows)
                matched_in_src += len(matched_rows)

            if unmatched_collected < num_unmatched_queries:
                needed = num_unmatched_queries - unmatched_collected
                neg_sample = chunk[~is_target & ~chunk["entity_id"].isin(target_q_ids)].head(needed).copy()
                if not neg_sample.empty:
                    for col in ["entity_id", "business_name", "business_address", "country"]:
                        neg_sample[col] = neg_sample[col].astype(str).str.strip()
                    unmatched_dfs.append(neg_sample)
                    unmatched_collected += len(neg_sample)

            if len(needed_for_src) > 0 and matched_in_src >= len(needed_for_src) and unmatched_collected >= num_unmatched_queries:
                break

    query_parts = matched_dfs + unmatched_dfs
    if query_parts:
        query_df = pd.concat(query_parts, ignore_index=True).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    else:
        query_df = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    total_true = sum(len(q) for q in sample_gt.values())
    logger.info(f"[DATA] Coherent sample ready: S1={len(s1_df):,}, Queries={len(query_df):,}, True links={total_true:,}")

    return s1_df, query_df, sample_gt
