"""
Controlled Multi-Category Negative Sampling Module for Entity Resolution.

Constructs balanced, high-signal training pairs from multiple negative categories:
1. Retrieval Hard Negatives: High BM25 / Char-TFIDF retrieval scores but incorrect entity.
2. Near-Duplicate / Lexical Hard Negatives: High name similarity (Jaro-Winkler >= 0.75) but different entity.
3. Address Hard Negatives: Similar address (Jaro-Winkler >= 0.70) but different business.
4. Same-Country Negatives: Same country but different entity.
5. Random Negatives: Disjoint pairs with low textual similarity.

Enforces strictly controlled sampling without uncontrolled duplication or data leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


def categorize_negative(row: Any) -> str:
    """Classifies a non-matching candidate pair into its primary negative category."""
    # 1. Near duplicate / Lexical hard negative
    if getattr(row, "name_jaro_winkler", 0.0) >= 0.78 or getattr(row, "name_fuzz_ratio", 0.0) >= 0.75:
        return "hard_lexical_near_duplicate"
    
    # 2. Address hard negative
    if getattr(row, "addr_jaro_winkler", 0.0) >= 0.72 or getattr(row, "addr_fuzz_ratio", 0.0) >= 0.70:
        return "hard_address_collision"
        
    # 3. Retrieval hard negative (high retrieval rank/score)
    if (getattr(row, "tfidf_name_score", 0.0) >= 0.35 or
        getattr(row, "bm25_name_score", 0.0) >= 2.0 or
        getattr(row, "best_retrieval_rank", 999) <= 3):
        return "retrieval_hard_negative"
        
    # 4. Same country negative
    if getattr(row, "country_exact_match", 0) == 1:
        return "same_country_negative"
        
    # 5. Generic / Random negative
    return "random_negative"


def build_controlled_training_pairs(
    candidate_feat_df: pd.DataFrame,
    max_negatives_per_positive: int = 8,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Constructs a controlled, high-signal training set from candidate pairs.
    
    Args:
        candidate_feat_df: Feature DataFrame with 'is_match' target column.
        max_negatives_per_positive: Maximum negative pairs per positive pair.
        random_state: Reproducibility seed.
        
    Returns:
        train_df: Balanced training DataFrame.
        dist_summary: Distribution dictionary of positive and negative categories.
    """
    if candidate_feat_df.empty or "is_match" not in candidate_feat_df.columns:
        raise ValueError("candidate_feat_df must contain 'is_match' target column.")
        
    rng = np.random.RandomState(random_state)
    
    # Separate positives and negatives
    pos_df = candidate_feat_df[candidate_feat_df["is_match"] == 1].copy()
    neg_df = candidate_feat_df[candidate_feat_df["is_match"] == 0].copy()
    
    n_pos = len(pos_df)
    n_neg_total = len(neg_df)
    
    logger.info(f"[NEGATIVE SAMPLING] Total Candidate Pairs: {len(candidate_feat_df):,} (Positives: {n_pos:,}, Negatives: {n_neg_total:,})")
    
    if n_pos == 0:
        logger.warning("[NEGATIVE SAMPLING] No positive pairs found in candidates.")
        return candidate_feat_df, {"total": len(candidate_feat_df), "positives": 0}
        
    # Assign negative categories
    neg_categories = []
    for row in neg_df.itertuples():
        neg_categories.append(categorize_negative(row))
    neg_df["neg_category"] = neg_categories
    
    target_negatives = min(n_neg_total, n_pos * max_negatives_per_positive)
    
    # Stratified negative sampling across categories prioritizing hard negatives
    category_quotas = {
        "hard_lexical_near_duplicate": 0.35,
        "hard_address_collision": 0.25,
        "retrieval_hard_negative": 0.20,
        "same_country_negative": 0.10,
        "random_negative": 0.10,
    }
    
    sampled_neg_dfs = []
    category_counts = {}
    
    grouped = neg_df.groupby("neg_category")
    
    # First pass: Sample according to target quotas
    remaining_budget = target_negatives
    for cat, quota_frac in category_quotas.items():
        if cat in grouped.groups:
            cat_group = grouped.get_group(cat)
            desired = int(target_negatives * quota_frac)
            sample_n = min(len(cat_group), desired)
            if sample_n > 0:
                sampled = cat_group.sample(n=sample_n, random_state=rng)
                sampled_neg_dfs.append(sampled)
                category_counts[cat] = sample_n
                remaining_budget -= sample_n
            else:
                category_counts[cat] = 0
        else:
            category_counts[cat] = 0
            
    # Second pass: If budget remains, fill from any remaining hard negatives
    if remaining_budget > 0:
        sampled_indices = set()
        for df in sampled_neg_dfs:
            sampled_indices.update(df.index)
        leftover = neg_df.loc[~neg_df.index.isin(sampled_indices)]
        if len(leftover) > 0:
            extra = leftover.sample(n=min(len(leftover), remaining_budget), random_state=rng)
            sampled_neg_dfs.append(extra)
            for cat, count in extra["neg_category"].value_counts().items():
                category_counts[cat] = category_counts.get(cat, 0) + count
                
    sampled_neg_df = pd.concat(sampled_neg_dfs, ignore_index=True) if sampled_neg_dfs else pd.DataFrame()
    pos_df["neg_category"] = "true_positive"
    
    train_df = pd.concat([pos_df, sampled_neg_df], ignore_index=True)
    # Shuffle training set
    train_df = train_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    
    dist_summary = {
        "total_training_pairs": len(train_df),
        "positive_pairs": n_pos,
        "positive_ratio": round(n_pos / max(len(train_df), 1), 4),
        "total_negative_pairs": len(sampled_neg_df),
        "negative_categories": category_counts
    }
    
    logger.info("=" * 60)
    logger.info("[NEGATIVE SAMPLING] Training Distribution Summary:")
    logger.info(f"  Total Pairs: {dist_summary['total_training_pairs']:,}")
    logger.info(f"  Positive Pairs: {dist_summary['positive_pairs']:,} ({dist_summary['positive_ratio']*100:.1f}%)")
    logger.info(f"  Negative Pairs: {dist_summary['total_negative_pairs']:,}")
    for cat, cnt in category_counts.items():
        logger.info(f"    - {cat}: {cnt:,} ({cnt/max(len(sampled_neg_df), 1)*100:.1f}%)")
    logger.info("=" * 60)
    
    return train_df, dist_summary
