"""
Script to train entity resolution model with multi-epoch iterative HNM.
Run from project root: python3 scripts/train_model.py [--sample-size 50000] [--epochs 5]
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
from src.ranking import EntityMatcherModel


def main():
    parser = argparse.ArgumentParser(description="Train Entity Matching Model with Multi-Epoch Iterative HNM")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of query samples for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--hnm-every", type=int, default=2, help="HNM frequency (e.g. every 2 epochs)")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"[TRAIN MODEL] Multi-Epoch Iterative Training (Epochs: {args.epochs}, HNM Every: {args.hnm_every})...")
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
    
    # Multi-Epoch Iterative HNM Training
    model = EntityMatcherModel()
    stats = model.fit_iterative_hnm(
        candidate_feat_df=feat_df,
        num_epochs=args.epochs,
        hnm_every=args.hnm_every,
        initial_threshold=0.25
    )
    
    importances = model.get_feature_importances()
    print("\nTop 10 Feature Importances:")
    print(importances.head(10).to_string(index=False))
    
    imp_csv = RESULTS_DIR / "feature_importances.csv"
    importances.to_csv(imp_csv, index=False)
    print(f"\nSaved feature importances to {imp_csv}")
    print("\n[DONE] Multi-Epoch Iterative Model Training Complete.")


if __name__ == "__main__":
    main()
