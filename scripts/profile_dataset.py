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
    print("[1/5] PROFILING DATASETS")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_data = []
    
    # 1. Train Reference S1
    print("\n--- Profiling Train Source 1 ---")
    train_s1 = load_source_tsv(TRAIN_S1_PATH)
    p_s1 = profile_dataframe(train_s1, "Train Source 1")
    print(f"  Rows: {p_s1['num_rows']:,}, Unique IDs: {p_s1['unique_ids']:,}")
    print(f"  Missing Names: {p_s1['missing_name_count']} ({p_s1['missing_name_pct']}%)")
    print(f"  Missing Addresses: {p_s1['missing_address_count']} ({p_s1['missing_address_pct']}%)")
    print(f"  Countries: {p_s1['countries']}")
    print(f"  Name Scripts: {p_s1['name_scripts']}")
    summary_data.append({"Dataset": "Train S1", "Rows": p_s1['num_rows'], "Countries": str(p_s1['countries'])})
    del train_s1
    
    # 2. Ground Truth
    print("\n--- Profiling Train Ground Truth ---")
    gt_df, s1_to_matches, match_to_s1 = load_ground_truth(TRAIN_GROUND_TRUTH_PATH)
    gt_stats = profile_ground_truth(gt_df, s1_to_matches)
    for k, v in gt_stats.items():
        print(f"  {k}: {v}")
    del gt_df, s1_to_matches, match_to_s1
    
    # 3. Test Reference S1
    print("\n--- Profiling Test Source 1 ---")
    test_s1 = load_source_tsv(TEST_S1_PATH)
    p_ts1 = profile_dataframe(test_s1, "Test Source 1")
    print(f"  Rows: {p_ts1['num_rows']:,}, Unique IDs: {p_ts1['unique_ids']:,}")
    print(f"  Countries: {p_ts1['countries']}")
    print(f"  Name Scripts: {p_ts1['name_scripts']}")
    summary_data.append({"Dataset": "Test S1", "Rows": p_ts1['num_rows'], "Countries": str(p_ts1['countries'])})
    del test_s1
    
    # Save profiling output
    profiling_csv = RESULTS_DIR / "profiling_summary.csv"
    pd.DataFrame(summary_data).to_csv(profiling_csv, index=False)
    print(f"\nSaved profiling summary to {profiling_csv}")
    print("\n[DONE] Dataset Profiling Complete.")


if __name__ == "__main__":
    main()
