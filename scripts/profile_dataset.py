"""
Script to profile train and test datasets.
Run from project root: python3 scripts/profile_dataset.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GROUND_TRUTH_PATH,
    TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH, RESULTS_DIR
)
from src.data_loader import load_source_tsv, load_ground_truth
from src.profiling import profile_dataframe, profile_ground_truth


def main():
    print("=" * 60)
    print("[1/10] PROFILING DATASETS")
    print("=" * 60)
    
    # Train Datasets
    print("\n--- Loading Train Datasets ---")
    train_s1 = load_source_tsv(TRAIN_S1_PATH)
    train_s2 = load_source_tsv(TRAIN_S2_PATH)
    train_s3 = load_source_tsv(TRAIN_S3_PATH)
    gt_df, s1_to_matches, _ = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)
    
    # Test Datasets
    print("\n--- Loading Test Datasets ---")
    test_s1 = load_source_tsv(TEST_S1_PATH)
    test_s2 = load_source_tsv(TEST_S2_PATH)
    test_s3 = load_source_tsv(TEST_S3_PATH)
    
    print("\n" + "=" * 60)
    print("DATASET SHAPES & BASIC SUMMARY")
    print("=" * 60)
    print(f"Train Source 1: {train_s1.shape[0]:,} rows, {train_s1['entity_id'].nunique():,} unique IDs")
    print(f"Train Source 2: {train_s2.shape[0]:,} rows, {train_s2['entity_id'].nunique():,} unique IDs")
    print(f"Train Source 3: {train_s3.shape[0]:,} rows, {train_s3['entity_id'].nunique():,} unique IDs")
    print(f"Train Ground Truth: {gt_df.shape[0]:,} S1 entities")
    print()
    print(f"Test Source 1: {test_s1.shape[0]:,} rows, {test_s1['entity_id'].nunique():,} unique IDs")
    print(f"Test Source 2: {test_s2.shape[0]:,} rows, {test_s2['entity_id'].nunique():,} unique IDs")
    print(f"Test Source 3: {test_s3.shape[0]:,} rows, {test_s3['entity_id'].nunique():,} unique IDs")
    
    # Ground Truth Profiling
    print("\n" + "=" * 60)
    print("GROUND TRUTH MATCH CARDINALITY")
    print("=" * 60)
    gt_stats = profile_ground_truth(gt_df, s1_to_matches)
    for k, v in gt_stats.items():
        print(f"  {k}: {v}")
        
    # Script & Country Profiling on Train
    print("\n" + "=" * 60)
    print("TRAIN SOURCE 1 PROFILING")
    print("=" * 60)
    p_s1 = profile_dataframe(train_s1, "Train Source 1")
    print(f"  Countries: {p_s1['countries']}")
    print(f"  Name Scripts: {p_s1['name_scripts']}")
    print(f"  Address Scripts: {p_s1['address_scripts']}")
    
    print("\n" + "=" * 60)
    print("TEST SOURCE 1 PROFILING")
    print("=" * 60)
    p_ts1 = profile_dataframe(test_s1, "Test Source 1")
    print(f"  Countries: {p_ts1['countries']}")
    print(f"  Name Scripts: {p_ts1['name_scripts']}")
    print(f"  Address Scripts: {p_ts1['address_scripts']}")
    
    # Save profiling output
    profiling_csv = RESULTS_DIR / "profiling_summary.csv"
    summary_data = [
        {"Dataset": "Train S1", "Rows": len(train_s1), "Countries": str(p_s1['countries'])},
        {"Dataset": "Train S2", "Rows": len(train_s2), "Countries": str(train_s2['country'].value_counts().to_dict())},
        {"Dataset": "Train S3", "Rows": len(train_s3), "Countries": str(train_s3['country'].value_counts().to_dict())},
        {"Dataset": "Test S1", "Rows": len(test_s1), "Countries": str(p_ts1['countries'])},
        {"Dataset": "Test S2", "Rows": len(test_s2), "Countries": str(test_s2['country'].value_counts().to_dict())},
        {"Dataset": "Test S3", "Rows": len(test_s3), "Countries": str(test_s3['country'].value_counts().to_dict())},
    ]
    pd.DataFrame(summary_data).to_csv(profiling_csv, index=False)
    print(f"\nSaved profiling summary to {profiling_csv}")
    print("\n[DONE] Dataset Profiling Complete.")


if __name__ == "__main__":
    main()
