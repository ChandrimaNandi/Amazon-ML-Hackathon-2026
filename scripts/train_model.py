"""
Script to train entity resolution model.
Run from project root: python3 scripts/train_model.py [--sample-size 50000]
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
from src.candidate_generation import generate_candidate_union
from src.features import extract_candidate_features
from src.ranking import EntityMatcherModel, mine_hard_negatives


def main():
    parser = argparse.ArgumentParser(description="Train Entity Matching Model")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of query samples for fast training")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"[TRAIN MODEL] Loading datasets (sample size: {args.sample_size:,})...")
    print("=" * 60)
    
    s1_df = load_source_tsv(TRAIN_S1_PATH)
    s2_df = load_source_tsv(TRAIN_S2_PATH)
    s3_df = load_source_tsv(TRAIN_S3_PATH)
    _, s1_to_matches, _ = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)
    
    query_df = pd.concat([s2_df, s3_df], ignore_index=True)
    if args.sample_size and args.sample_size < len(query_df):
        query_df = query_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(query_df):,} query records for model training.")
        
    s1_df = create_normalized_features(s1_df)
    query_df = create_normalized_features(query_df)
    
    # Generate candidates
    cand_df, cand_stats = generate_candidate_union(s1_df, query_df, k_name=20, k_address=15)
    
    # Feature extraction
    feat_df = extract_candidate_features(cand_df, s1_df, query_df, s1_to_matches)
    
    # Train initial model
    model = EntityMatcherModel()
    model.fit(feat_df)
    
    # Mine hard negatives
    hard_negs = mine_hard_negatives(model, feat_df, threshold=0.25)
    
    if not hard_negs.empty:
        print(f"Retraining model with {len(hard_negs):,} hard negatives...")
        retrain_df = pd.concat([feat_df, hard_negs], ignore_index=True)
        model.fit(retrain_df)
        
    importances = model.get_feature_importances()
    print("\nFeature Importances:")
    print(importances.head(10).to_string(index=False))
    
    imp_csv = RESULTS_DIR / "feature_importances.csv"
    importances.to_csv(imp_csv, index=False)
    print(f"\nSaved feature importances to {imp_csv}")
    print("\n[DONE] Model Training Complete.")


if __name__ == "__main__":
    main()
