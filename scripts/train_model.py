"""
Script to train Entity Matching Model using strict entity-level split and controlled negative sampling.
Run from project root: python3 scripts/train_model.py [--sample-s1 25000] [--sample-queries 10000]
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GROUND_TRUTH_PATH, RESULTS_DIR
)
from src.data_loader import load_source_tsv, load_ground_truth
from src.normalization import create_normalized_features
from src.candidate_generation import CandidateGenerator
from src.features import extract_candidate_features
from src.negative_sampling import build_controlled_training_pairs
from src.ranking import EntityMatcherModel
from src.experiments import create_entity_level_split


def main():
    parser = argparse.ArgumentParser(description="Train Entity Matching Model")
    parser.add_argument("--sample-s1", type=int, default=25000, help="S1 reference entities sample size")
    parser.add_argument("--sample-queries", type=int, default=10000, help="Query records sample size")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation S1 entity ratio")
    parser.add_argument("--max-negatives", type=int, default=8, help="Max negatives per positive")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"[TRAIN MODEL] Entity-Level Split Training (S1: {args.sample_s1:,}, Queries: {args.sample_queries:,})")
    print("=" * 60)
    
    # 1. Load data
    s1_df = pd.read_csv(TRAIN_S1_PATH, sep="\t", nrows=args.sample_s1, keep_default_na=False, dtype=str)
    gt_df, s1_to_matches, _ = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)
    
    s1_sample_ids = set(s1_df["entity_id"])
    active_gt = {s1: s1_to_matches.get(s1, set()) for s1 in s1_sample_ids}
    needed_q_ids = set()
    for q_set in active_gt.values():
        needed_q_ids.update(q_set)
        
    s2_df = pd.read_csv(TRAIN_S2_PATH, sep="\t", nrows=args.sample_queries * 2, keep_default_na=False, dtype=str)
    s3_df = pd.read_csv(TRAIN_S3_PATH, sep="\t", nrows=args.sample_queries * 2, keep_default_na=False, dtype=str)
    query_df = pd.concat([s2_df, s3_df], ignore_index=True)
    query_df = query_df[query_df["entity_id"].isin(needed_q_ids) | (query_df.index < args.sample_queries)].head(args.sample_queries).reset_index(drop=True)
    
    # 2. Normalize
    s1_df = create_normalized_features(s1_df)
    query_df = create_normalized_features(query_df)
    
    # 3. Entity-level split
    train_s1, val_s1, train_q, val_q, train_gt, val_gt = create_entity_level_split(
        s1_df, query_df, active_gt, val_ratio=args.val_ratio, random_seed=42
    )
    
    # 4. Candidate generation on train and val separately
    gen_train = CandidateGenerator()
    gen_train.fit(train_s1)
    train_cands, _ = gen_train.generate_candidates(train_q)
    
    gen_val = CandidateGenerator()
    gen_val.fit(val_s1)
    val_cands, _ = gen_val.generate_candidates(val_q)
    
    # 5. Feature extraction
    train_feat = extract_candidate_features(train_cands, train_s1, train_q, train_gt)
    val_feat = extract_candidate_features(val_cands, val_s1, val_q, val_gt)
    
    # 6. Controlled negative sampling on train
    balanced_train, neg_dist = build_controlled_training_pairs(
        train_feat, max_negatives_per_positive=args.max_negatives, random_state=42
    )
    
    # 7. Model fitting with validation early stopping
    model = EntityMatcherModel()
    train_summary = model.fit(balanced_train, val_df=val_feat, early_stopping_rounds=40)
    
    # 8. Feature importances
    importances = model.get_feature_importances()
    print("\nTop 15 Feature Importances:")
    print(importances.head(15).to_string(index=False))
    
    # Save model and artifacts
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(RESULTS_DIR / "matcher_model.pkl")
    importances.to_csv(RESULTS_DIR / "feature_importances.csv", index=False)
    pd.DataFrame([train_summary]).to_csv(RESULTS_DIR / "training_summary.csv", index=False)
    
    print("\n[DONE] Model Training & Serialization Complete.")


if __name__ == "__main__":
    main()
