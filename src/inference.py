"""
Inference Pipeline Module for Business Entity Resolution.
Runs candidate generation, feature extraction, model scoring, and submission TSV generation.
"""

import time
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
from src.candidate_generation import generate_candidate_union
from src.features import extract_candidate_features
from src.ranking import EntityMatcherModel
from src.thresholding import apply_decision_rules

logger = logging.getLogger(__name__)


def generate_submission_files(
    model: EntityMatcherModel,
    test_dir: Path,
    output_matching_path: Path = SUBMISSION_MATCHING_PATH,
    output_candidate_path: Path = SUBMISSION_CANDIDATE_PATH,
    abs_threshold: float = 0.50,
    margin_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Runs full inference pipeline on test dataset and creates:
    - matching_results.tsv
    - candidate_pairs.tsv
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
    
    # Combine S2 and S3 queries
    query_df = pd.concat([s2_df, s3_df], ignore_index=True)
    logger.info(f"Combined {len(query_df):,} query records (S2: {len(s2_df):,}, S3: {len(s3_df):,})")
    
    # 2. Normalize
    s1_df = create_normalized_features(s1_df)
    query_df = create_normalized_features(query_df)
    
    # 3. Candidate Generation
    cand_df, cand_stats = generate_candidate_union(
        s1_df=s1_df,
        query_df=query_df,
        k_name=30,
        k_address=20,
        k_combined=30,
        k_char=30
    )
    
    # 4. Feature Extraction
    feat_df = extract_candidate_features(
        candidate_df=cand_df,
        s1_df=s1_df,
        query_df=query_df,
        s1_to_matches=None
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
    
    # Build candidate mapping per S1 entity: s1_id -> set of candidate query_ids
    s1_to_candidates: Dict[str, Set[str]] = {}
    all_s1_test_ids = s1_df["entity_id"].tolist()
    
    for row in cand_df.itertuples():
        s1_to_candidates.setdefault(row.s1_id, set()).add(row.query_id)
        
    # Ensure every S1 ID is present in outputs (even if empty)
    for s1_id in all_s1_test_ids:
        if s1_id not in s1_to_candidates:
            s1_to_candidates[s1_id] = set()
        if s1_id not in s1_to_pred_matches:
            s1_to_pred_matches[s1_id] = set()
            
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
    
    logger.info("INFERENCE PIPELINE COMPLETED SUCCESSFULLY.")
    return summary
